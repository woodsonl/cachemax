//! The canonical ledger: the exact request the proxy forwarded and the exact
//! response the provider returned, per turn.
//!
//! Repair's core invariant lives here. A provider caches by the exact tokens
//! of what it received, and the only serialization cachemax can promise to
//! extend is its own — so instead of predicting the provider's tokenizer, the
//! ledger simply remembers what we sent. Per session and model it stores:
//!
//! - `request_messages` — the `messages` array exactly as forwarded upstream
//!   (after any proxy-side injection, before the provider saw it);
//! - `response_messages` — the assistant message(s) exactly as the provider
//!   returned them, reassembled from the stream when the turn streamed, in
//!   the client's dialect — i.e. the element a compliant client re-sends as
//!   history next turn.
//!
//! Later batches compare the next request against this chain and rewrite
//! drifted history to prefix-extend it. Token counts never enter that
//! decision; the tokenizer stays a reporting device.
//!
//! The ledger is **content-bearing by design and stays local**. Memory keeps
//! only the latest turn per session+model — the cumulative `request_messages`
//! of the latest turn already contains the whole chain, so nothing
//! repair-relevant is lost to that bound. Disk (when enabled) keeps the full
//! audit trail as JSONL, one file per session, one line per complete turn.
//! Export and finalize logs never read from here; they stay metrics-only.
//!
//! Incomplete turns never enter the chain: the client did not receive a
//! complete assistant message, so there is nothing canonical to remember. A
//! retried request then matches the pre-failure chain cleanly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Cap on distinct sessions held in memory. The in-memory ledger keeps one
/// turn (the latest, per model) per session, so the bound is on *sessions*,
/// not turns. When a capped reload must evict, eviction follows the
/// directory's file-name order.
const SESSIONS_CAP: usize = 64;

/// Cap on a retained non-streaming response body. A whole-JSON response is
/// buffered to extract the assistant message; beyond this the capture
/// degrades to "no canonical response" rather than buffering without bound.
/// Streaming responses never hit this — they are reassembled incrementally.
const BODY_CAP: usize = 16 * 1024 * 1024;

/// Cap on a single pending SSE line. Real `data:` events are small; a line
/// larger than this means the stream is not the dialect we know, and the
/// capture degrades instead of growing.
const LINE_CAP: usize = 1024 * 1024;

/// One complete turn, exactly as forwarded and received.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalTurn {
    /// Turn index within the session, as planned. Gaps are possible: an
    /// incomplete attempt consumes a turn number in the record store but
    /// never enters the ledger. The chain, not the numbering, is authoritative.
    pub turn: u32,
    /// Model this turn ran on. The canonical chain is per session+model;
    /// a model switch starts a fresh chain rather than splicing.
    pub model: String,
    /// The `messages` array exactly as forwarded upstream this turn.
    /// Key order is preserved (`serde_json` `preserve_order`), and
    /// serialization of this value is deterministic, so "extend the bytes we
    /// forwarded" reduces to "extend this value's serialization".
    pub request_messages: serde_json::Value,
    /// The assistant message(s) exactly as the provider returned them, in the
    /// client's dialect — the element a compliant client re-sends next turn.
    /// Empty when the response could not be reassembled (unknown dialect,
    /// degraded capture); repair then falls back to pass-through.
    pub response_messages: Vec<serde_json::Value>,
    /// The request's cumulative prefix-hash sequence, as resolved for session
    /// continuity. A cross-check between the session store and the ledger;
    /// the ledger's authority is the messages themselves, never the hashes.
    pub prefix_hashes: Vec<u64>,
    /// Breakpoints the proxy placed on the request as sent (0 when the
    /// client manages its own, or the feature is off). A compliant client
    /// echoes those bytes back, so the next turn tells its own hints from
    /// the client's by this count.
    #[serde(default)]
    pub breakpoints: u64,
}

/// The on-disk line: the turn plus the session it belongs to. One JSON object
/// per line in `<ledger-dir>/<session>.jsonl`.
#[derive(Serialize, Deserialize)]
struct LedgerLine {
    session_id: u64,
    #[serde(flatten)]
    turn: CanonicalTurn,
}

/// The ledger: latest canonical turn per session+model in memory, full audit
/// trail on disk. Disk writes are best-effort — a ledger failure must never
/// fail a response the proxy has already started forwarding — and are logged
/// (metadata only) when they fail.
pub struct Ledger {
    sessions: HashMap<u64, HashMap<String, CanonicalTurn>>,
    /// Session ids in first-append order, for eviction.
    order: Vec<u64>,
    /// (session, model) pairs in append order, most recent last. Lets a
    /// forked session (a truncated or re-based request) find the chain it
    /// actually extends, keyed by model.
    recent: Vec<(u64, String)>,
    dir: Option<PathBuf>,
}

impl Ledger {
    /// A ledger that never touches disk. What tests and `--no-ledger` run
    /// with: full in-memory behavior, no persistence.
    pub fn in_memory() -> Self {
        Self {
            sessions: HashMap::new(),
            order: Vec::new(),
            recent: Vec::new(),
            dir: None,
        }
    }

    /// A ledger persisted as JSONL under `dir`, reloading what is already
    /// there. Torn trailing lines (a crash mid-append) are skipped, not
    /// fatal: a lost line loses one turn of audit, never correctness.
    /// Files are read in file-name order, so a capped reload's eviction
    /// is deterministic for a given directory listing.
    pub fn on_disk(dir: PathBuf) -> std::io::Result<Self> {
        let mut ledger = Self {
            sessions: HashMap::new(),
            order: Vec::new(),
            recent: Vec::new(),
            dir: Some(dir.clone()),
        };
        std::fs::create_dir_all(&dir)?;
        let mut files: Vec<String> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".jsonl"))
            .collect();
        files.sort();
        for name in files {
            let path = dir.join(name);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<LedgerLine>(line) else {
                    // A torn or corrupt line (crash mid-append): skip it. The
                    // in-memory chain keeps the newest parseable turn; the
                    // audit trail may miss one turn, the behavior may not.
                    continue;
                };
                ledger.remember(entry.session_id, entry.turn);
            }
        }
        Ok(ledger)
    }

    /// Record one complete turn. Memory always; disk when persistent. A turn
    /// numbered at or below the remembered turn for the same session+model is
    /// ignored — an out-of-order finalize (concurrent requests that planned
    /// the same turn) must not regress the chain.
    pub fn append(&mut self, session_id: u64, turn: CanonicalTurn) {
        if self.remember(session_id, turn.clone()) {
            self.flush_line(session_id, &turn);
        }
    }

    /// Memory-side insert with the eviction bound applied. Returns whether
    /// the turn was accepted (false when it would regress the chain).
    fn remember(&mut self, session_id: u64, turn: CanonicalTurn) -> bool {
        let per_model = self.sessions.entry(session_id).or_insert_with(|| {
            self.order.push(session_id);
            HashMap::new()
        });
        let regresses = per_model
            .get(&turn.model)
            .is_some_and(|prev| prev.turn > turn.turn);
        if regresses {
            return false;
        }
        self.recent
            .retain(|(s, m)| *s != session_id || *m != turn.model);
        self.recent.push((session_id, turn.model.clone()));
        per_model.insert(turn.model.clone(), turn);
        // Evict oldest-created sessions beyond the cap. First-append order is
        // creation order; refreshing on activity would require a tick
        // bookkeeping the bound does not need at single-user scale.
        while self.order.len() > SESSIONS_CAP {
            let victim = self.order.remove(0);
            self.sessions.remove(&victim);
        }
        true
    }

    /// Append one complete JSON line to the session's file. A single
    /// `write_all` of the full line keeps the file line-atomic under
    /// O_APPEND: a reader sees the whole line or none of it.
    ///
    /// Blocking IO, called synchronously from the finalize path: once per
    /// turn, after the upstream stream has ended (or on disconnect, where
    /// `Drop` cannot `await`), a single small append on local disk. A slow
    /// disk delays that response's close, never its bytes; the client stream
    /// itself is never touched by this write.
    fn flush_line(&self, session_id: u64, turn: &CanonicalTurn) {
        let Some(dir) = &self.dir else { return };
        let line = serde_json::to_vec(&LedgerLine {
            session_id,
            turn: turn.clone(),
        });
        let line = match line {
            Ok(mut l) => {
                l.push(b'\n');
                l
            }
            // A turn that cannot serialize cannot be remembered on disk; the
            // in-memory chain still holds it. Log metadata only, never the
            // content that failed.
            Err(e) => {
                tracing::warn!(
                    target: "cachemax_ledger",
                    session = session_id,
                    turn = turn.turn,
                    error = %e,
                    "ledger line serialization failed"
                );
                return;
            }
        };
        let write = || -> std::io::Result<()> {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(dir.join(format!("{session_id}.jsonl")))?;
            f.write_all(&line)?;
            f.flush()
        };
        if let Err(e) = write() {
            tracing::warn!(
                target: "cachemax_ledger",
                session = session_id,
                turn = turn.turn,
                error = %e,
                "ledger flush failed; chain kept in memory"
            );
        }
    }

    /// Forget everything in memory (the purge operation; the on-disk trail
    /// is the caller's to delete). Subsequent requests classify as first
    /// turns until new chains form.
    pub fn clear(&mut self) {
        self.sessions.clear();
        self.order.clear();
        self.recent.clear();
    }

    /// The latest remembered turn for a session+model.
    pub fn last_turn(&self, session_id: u64, model: &str) -> Option<&CanonicalTurn> {
        self.sessions.get(&session_id)?.get(model)
    }

    /// Whether the session has canonical turns under any model. Distinguishes
    /// a model switch (chains exist, none for this model) from a first turn.
    pub fn session_has_chains(&self, session_id: u64) -> bool {
        self.sessions
            .get(&session_id)
            .is_some_and(|per_model| !per_model.is_empty())
    }

    /// Replay a recorded ledger directory: for every session+model chain on
    /// disk, emit the request body that extends that chain (the canonical
    /// request plus a fresh user turn), in original turn order. The bench
    /// A/B driver builds both variants from these: the drifted form (what
    /// a re-serializing client sends) and the canonical form (what repair
    /// would forward).
    ///
    /// Untrusted disk: a chain whose stored messages are not an array is
    /// skipped, never emitted — a pair must be a real request the provider
    /// could have cached.
    pub fn replay_requests(dir: &std::path::Path) -> std::io::Result<Vec<ReplayRequest>> {
        let ledger = Ledger::on_disk(dir.to_path_buf())?;
        let mut out: Vec<ReplayRequest> = ledger
            .sessions
            .iter()
            .flat_map(|(session_id, per_model)| {
                per_model.iter().filter_map(move |(model, turn)| {
                    // A chain only extends a request that is a message
                    // array; anything else on disk is not a chain.
                    turn.request_messages.as_array()?;
                    let mut chain = turn.request_messages.clone();
                    if let Some(arr) = chain.as_array_mut() {
                        arr.push(serde_json::json!(
                            {"role": "user", "content": "Continue."}
                        ));
                    }
                    Some(ReplayRequest {
                        session_id: *session_id,
                        turn: turn.turn,
                        model: model.clone(),
                        messages: chain,
                    })
                })
            })
            .collect();
        // A total order: two models can share a session+turn, and the
        // output must be identical across runs.
        out.sort_by_key(|r| (r.session_id, r.turn, r.model.clone()));
        Ok(out)
    }

    /// The most recently appended-to session that has a chain for `model`.
    /// A request whose leading history was truncated or re-based resolves
    /// to a *new* session (no shared prefix-hash), and this finds the
    /// conversation it actually extends — same model only.
    pub fn most_recent_chain_session(&self, model: &str) -> Option<u64> {
        self.recent
            .iter()
            .rev()
            .find(|(_, m)| m == model)
            .map(|(s, _)| *s)
    }

    /// The canonical message chain for a session+model: the messages of the
    /// latest turn exactly as forwarded, extended by the assistant message(s)
    /// exactly as received. This is what the next request should
    /// prefix-extend. `None` when the session+model has no remembered turn or
    /// the stored messages are not an array (never the case via the proxy,
    /// which validates the shape before forwarding).
    pub fn canonical_messages(
        &self,
        session_id: u64,
        model: &str,
    ) -> Option<Vec<serde_json::Value>> {
        let last = self.last_turn(session_id, model)?;
        let request = last.request_messages.as_array()?;
        let mut chain: Vec<serde_json::Value> = request.clone();
        chain.extend(last.response_messages.iter().cloned());
        Some(chain)
    }
}

/// Thread-safe wrapper, mirroring [`crate::sessions::SharedSessions`]: the
/// lock is held only for the duration of a store call and never across an
/// `.await`; poisoning is recovered from because the ledger has no invariant
/// a panicked handler could break.
pub struct SharedLedger(pub Mutex<Ledger>);

/// One replayable request reconstructed from the on-disk ledger: the
/// canonical chain plus a fresh tail. The replay bench drives a
/// stub/provider with the drifted and canonical serializations of this
/// body to measure the repair delta.
#[derive(Debug)]
pub struct ReplayRequest {
    pub session_id: u64,
    pub turn: u32,
    pub model: String,
    /// The request body's `messages`: the chain as forwarded, extended by
    /// a fresh user turn.
    pub messages: serde_json::Value,
}

impl SharedLedger {
    /// An in-memory ledger (no disk). The default for tests and `--no-ledger`.
    pub fn new() -> Self {
        Self(Mutex::new(Ledger::in_memory()))
    }

    /// A disk-persisted ledger under `dir`, reloading existing turns.
    pub fn on_disk(dir: PathBuf) -> std::io::Result<Self> {
        Ok(Self(Mutex::new(Ledger::on_disk(dir)?)))
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for SharedLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// The response dialect an assembler works in. Decides how a response is
/// reassembled into the assistant element the client will re-send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// OpenAI `chat/completions` — also OpenRouter, llama.cpp, vLLM, mlx-lm
    /// (the OpenAI-compatible wire shape).
    OpenAi,
    /// Anthropic Messages API (native blocks).
    Anthropic,
}

impl Dialect {
    /// Map a backend adapter name to its response dialect. Everything but
    /// Anthropic speaks the OpenAI-compatible shape on this proxy.
    pub fn from_backend(name: &str) -> Self {
        match name {
            "anthropic" => Dialect::Anthropic,
            _ => Dialect::OpenAi,
        }
    }
}

/// Incremental reassembly of the assistant message(s) from a response, fed
/// the same untouched chunks the client receives.
///
/// Transport is auto-detected from the first meaningful bytes: a body that
/// begins `data:` / `event:` / `:` is SSE (reassembled line by line as events
/// arrive — never buffered whole), anything else is a whole-JSON document
/// (held, capped, and parsed once). Capture is *degradation-only*: on
/// anything unexpected (a line or body over cap, an unparseable dialect) the
/// assembler stops and [`ResponseAssembler::finish`] reports no canonical
/// response, which later repair treats as pass-through, never as invention.
pub struct ResponseAssembler {
    dialect: Dialect,
    mode: Mode,
    pending: Vec<u8>,
    body: Vec<u8>,
    oa: OpenAiAcc,
    an: AnthropicAcc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Not yet decided; the first meaningful bytes decide.
    Undecided,
    /// Line-oriented `data:` events.
    Sse,
    /// One whole JSON document.
    JsonBody,
    /// Capture abandoned (cap exceeded). Finish reports nothing.
    Degraded,
}

impl ResponseAssembler {
    pub fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            mode: Mode::Undecided,
            pending: Vec::new(),
            body: Vec::new(),
            oa: OpenAiAcc::default(),
            an: AnthropicAcc::default(),
        }
    }

    /// Observe one forwarded chunk, untouched. Cheap by construction: only
    /// complete `data:` lines that could carry content are parsed, and the
    /// retained state is the size of the message being assembled, not the
    /// stream.
    pub fn on_chunk(&mut self, chunk: &[u8]) {
        if self.mode == Mode::Degraded {
            return;
        }
        if self.mode == Mode::Undecided {
            // The transport is decided by the first meaningful bytes, which
            // may be split across chunks: a JSON document announces itself
            // with `{`/`[` immediately; an SSE stream announces itself with
            // a complete first line that is not JSON. Never guess early —
            // deciding on a partial first line would misroute the stream.
            self.pending.extend_from_slice(chunk);
            let meaningful = self
                .pending
                .iter()
                .position(|b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n'));
            let Some(start) = meaningful else {
                return; // only framing whitespace so far
            };
            let head = &self.pending[start..];
            if head[0] == b'{' || head[0] == b'[' {
                self.mode = Mode::JsonBody;
                let buffered = std::mem::take(&mut self.pending);
                self.body = buffered;
                if self.body.len() > BODY_CAP {
                    self.degrade();
                }
                return;
            }
            if head.contains(&b'\n') {
                // A complete non-JSON first line: line-oriented framing.
                self.mode = Mode::Sse;
                self.drain_pending_lines();
                return;
            }
            if self.pending.len() > LINE_CAP {
                self.degrade();
            }
            return;
        }
        match self.mode {
            Mode::Sse => {
                self.pending.extend_from_slice(chunk);
                self.drain_pending_lines();
                if self.pending.len() > LINE_CAP {
                    self.degrade();
                }
            }
            Mode::JsonBody => {
                self.body.extend_from_slice(chunk);
                if self.body.len() > BODY_CAP {
                    self.degrade();
                }
            }
            Mode::Undecided | Mode::Degraded => unreachable!("decided above"),
        }
    }

    /// Split `pending` into complete lines and feed each to the SSE handler.
    fn drain_pending_lines(&mut self) {
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=pos).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            self.on_sse_line(line);
            if self.mode == Mode::Degraded {
                return;
            }
        }
    }

    /// Process one SSE `data:` line (already `\n`/`\r`-stripped).
    fn on_sse_line(&mut self, line: &[u8]) {
        let Some(payload) = line
            .strip_prefix(b"data: ")
            .or_else(|| line.strip_prefix(b"data:"))
        else {
            return; // `event:`/comment/id lines carry no content
        };
        if payload == b"[DONE]" {
            return;
        }
        // Prefilter: content-bearing events mention deltas, blocks, or
        // choices. Usage-only chunks and anything else skip the parse. Not
        // valid UTF-8? Then not JSON; skip the parse too.
        let Ok(text) = std::str::from_utf8(payload) else {
            return;
        };
        if !(text.contains("delta") || text.contains("block") || text.contains("choices")) {
            return;
        }
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(payload) else {
            return;
        };
        match self.dialect {
            Dialect::OpenAi => self.oa.on_event(&v),
            Dialect::Anthropic => self.an.on_event(&v),
        }
    }

    /// The reassembled assistant message(s), exactly as the provider
    /// returned them — the element a compliant client re-sends next turn.
    /// Empty when nothing could be reassembled (empty stream, unknown shape,
    /// degraded capture): absence is a fact repair must respect, not fill.
    pub fn finish(&self) -> Vec<serde_json::Value> {
        if self.mode == Mode::Degraded {
            return Vec::new();
        }
        match self.mode {
            Mode::Sse => match self.dialect {
                Dialect::OpenAi => self.oa.finish(),
                Dialect::Anthropic => self.an.finish(),
            },
            Mode::JsonBody => match self.dialect {
                Dialect::OpenAi => self.oa.finish_body(&self.body),
                Dialect::Anthropic => self.an.finish_body(&self.body),
            },
            Mode::Undecided | Mode::Degraded => Vec::new(),
        }
    }

    /// The whole retained JSON body (non-streaming responses), for usage
    /// extraction. `None` on SSE responses and degraded captures.
    pub fn full_body(&self) -> Option<&[u8]> {
        match self.mode {
            Mode::JsonBody => Some(&self.body),
            _ => None,
        }
    }

    fn degrade(&mut self) {
        self.mode = Mode::Degraded;
        self.pending = Vec::new();
        self.body = Vec::new();
        self.oa = OpenAiAcc::default();
        self.an = AnthropicAcc::default();
    }
}

/// OpenAI-dialect accumulation: role/content deltas plus tool calls merged by
/// index (argument fragments arrive as separate string deltas).
#[derive(Default)]
struct OpenAiAcc {
    role: Option<String>,
    content: String,
    saw_content: bool,
    tool_calls: Vec<OpenAiToolCall>,
}

#[derive(Default)]
struct OpenAiToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAiAcc {
    fn on_event(&mut self, v: &serde_json::Value) {
        let Some(delta) = v.pointer("/choices/0/delta") else {
            return; // usage-only terminal chunk, etc.
        };
        if let Some(role) = delta.get("role").and_then(|r| r.as_str()) {
            self.role = Some(role.to_string());
        }
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            self.saw_content = true;
            self.content.push_str(content);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for call in calls {
                let index = call
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .unwrap_or(self.tool_calls.len().max(1) as u64 - 1)
                    as usize;
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(OpenAiToolCall::default());
                }
                let slot = &mut self.tool_calls[index];
                if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                    slot.id = Some(id.to_string());
                }
                if let Some(name) = call.pointer("/function/name").and_then(|n| n.as_str()) {
                    slot.name = Some(name.to_string());
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(|a| a.as_str()) {
                    slot.arguments.push_str(args);
                }
            }
        }
    }

    fn started(&self) -> bool {
        self.role.is_some() || self.saw_content || !self.tool_calls.is_empty()
    }

    fn finish(&self) -> Vec<serde_json::Value> {
        if !self.started() {
            return Vec::new();
        }
        let mut message = serde_json::json!({
            "role": self.role.clone().unwrap_or_else(|| "assistant".to_string()),
            // Mirrors the non-streaming shape: `content` is present (null)
            // for a pure tool-call turn, absent-content never fabricates "".
            "content": if self.saw_content {
                serde_json::Value::String(self.content.clone())
            } else {
                serde_json::Value::Null
            },
        });
        if !self.tool_calls.is_empty() {
            message["tool_calls"] = serde_json::Value::Array(
                self.tool_calls
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id.clone().unwrap_or_default(),
                            "type": "function",
                            "function": {
                                "name": c.name.clone().unwrap_or_default(),
                                "arguments": c.arguments.clone(),
                            },
                        })
                    })
                    .collect(),
            );
        }
        vec![message]
    }

    /// Non-streaming: the message object is in the body verbatim.
    fn finish_body(&self, body: &[u8]) -> Vec<serde_json::Value> {
        serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.pointer("/choices/0/message").cloned())
            .map(|m| vec![m])
            .unwrap_or_default()
    }
}

/// Anthropic-dialect accumulation: content blocks assembled from
/// `content_block_start` + typed deltas, in block order.
#[derive(Default)]
struct AnthropicAcc {
    blocks: Vec<AnthropicBlock>,
}

#[derive(Default)]
struct AnthropicBlock {
    kind: Option<String>,
    text: String,
    thinking: String,
    signature: String,
    id: Option<String>,
    name: Option<String>,
    input_json: String,
    /// Blocks of unknown type, kept verbatim from `content_block_start`.
    verbatim: Option<serde_json::Value>,
}

impl AnthropicAcc {
    fn on_event(&mut self, v: &serde_json::Value) {
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "content_block_start" => {
                let Some(block) = v.pointer("/content_block") else {
                    return;
                };
                let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                while self.blocks.len() <= index {
                    self.blocks.push(AnthropicBlock::default());
                }
                let slot = &mut self.blocks[index];
                slot.kind = block.get("type").and_then(|t| t.as_str()).map(String::from);
                match slot.kind.as_deref() {
                    Some("text") => {
                        slot.text = block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                    }
                    Some("tool_use") => {
                        slot.id = block.get("id").and_then(|i| i.as_str()).map(String::from);
                        slot.name = block.get("name").and_then(|n| n.as_str()).map(String::from);
                    }
                    _ => {
                        slot.verbatim = Some(block.clone());
                    }
                }
            }
            "content_block_delta" => {
                let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let Some(delta) = v.get("delta") else { return };
                let Some(slot) = self.blocks.get_mut(index) else {
                    return;
                };
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => {
                        if let Some(t) = delta.get("text").and_then(|t| t.as_str()) {
                            slot.text.push_str(t);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(j) = delta.get("partial_json").and_then(|j| j.as_str()) {
                            slot.input_json.push_str(j);
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(t) = delta.get("thinking").and_then(|t| t.as_str()) {
                            slot.thinking.push_str(t);
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(s) = delta.get("signature").and_then(|s| s.as_str()) {
                            slot.signature.push_str(s);
                        }
                    }
                    _ => {}
                }
            }
            _ => {} // message_start/message_delta/message_stop: usage, not content
        }
    }

    fn finish(&self) -> Vec<serde_json::Value> {
        if self.blocks.is_empty() {
            return Vec::new();
        }
        let content: Vec<serde_json::Value> = self
            .blocks
            .iter()
            .map(|b| match b.kind.as_deref() {
                Some("text") => serde_json::json!({"type": "text", "text": b.text}),
                Some("thinking") => {
                    let mut o = serde_json::json!({
                        "type": "thinking",
                        "thinking": b.thinking,
                    });
                    if !b.signature.is_empty() {
                        o["signature"] = serde_json::Value::String(b.signature.clone());
                    }
                    o
                }
                Some("tool_use") => serde_json::json!({
                    "type": "tool_use",
                    "id": b.id.clone().unwrap_or_default(),
                    "name": b.name.clone().unwrap_or_default(),
                    // The non-streaming shape carries `input` as an object;
                    // an empty accumulation is `{}`, an unparseable one stays
                    // the string the provider sent — never invented.
                    "input": parse_input(&b.input_json),
                }),
                _ => b.verbatim.clone().unwrap_or_else(
                    || serde_json::json!({"type": b.kind.clone().unwrap_or_default()}),
                ),
            })
            .collect();
        vec![serde_json::json!({"role": "assistant", "content": content})]
    }

    /// Non-streaming: the content array is in the body verbatim.
    fn finish_body(&self, body: &[u8]) -> Vec<serde_json::Value> {
        serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.get("content").and_then(|c| c.as_array()).cloned())
            .map(|content| vec![serde_json::json!({"role": "assistant", "content": content})])
            .unwrap_or_default()
    }
}

/// Parse accumulated tool-use input into the object the non-streaming shape
/// carries. Empty → `{}`; unparseable → the raw string, flagged by shape.
fn parse_input(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn turn(seq: u32, marker: &str) -> CanonicalTurn {
        CanonicalTurn {
            turn: seq,
            model: "gpt-4o".into(),
            request_messages: json!([{"role": "user", "content": marker}]),
            response_messages: vec![json!({"role": "assistant", "content": marker})],
            prefix_hashes: vec![seq as u64],
            breakpoints: 0,
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("cachemax-ledger-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn turns_append_in_order_and_late_older_turns_do_not_regress() {
        let dir = temp_dir("order");
        let mut ledger = Ledger::on_disk(dir.clone()).unwrap();
        ledger.append(1, turn(0, "t0"));
        ledger.append(1, turn(1, "t1"));
        ledger.append(1, turn(2, "t2"));
        // A late finalize of turn 1 (concurrent requests planned the same
        // turn) must not unseat the remembered turn 2 — in memory or on disk.
        ledger.append(1, turn(1, "late"));
        let last = ledger.last_turn(1, "gpt-4o").unwrap();
        assert_eq!(last.turn, 2);
        assert_eq!(
            last.request_messages,
            json!([{"role": "user", "content": "t2"}])
        );
        let disk = std::fs::read_to_string(dir.join("1.jsonl")).unwrap();
        assert_eq!(disk.lines().count(), 3, "a regressing turn is not audited");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn canonical_chain_extends_the_request_with_the_response() {
        let mut ledger = Ledger::in_memory();
        ledger.append(7, turn(3, "hi"));
        let chain = ledger.canonical_messages(7, "gpt-4o").unwrap();
        assert_eq!(
            chain,
            vec![
                json!({"role": "user", "content": "hi"}),
                json!({"role": "assistant", "content": "hi"}),
            ]
        );
        // Unknown session/model: nothing canonical, never a guess.
        assert!(ledger.canonical_messages(8, "gpt-4o").is_none());
        assert!(ledger.canonical_messages(7, "other-model").is_none());
    }

    #[test]
    fn a_model_switch_starts_an_independent_chain() {
        let mut ledger = Ledger::in_memory();
        let mut a = turn(0, "on-a");
        a.model = "model-a".into();
        let mut b = turn(0, "on-b");
        b.model = "model-b".into();
        ledger.append(1, a);
        ledger.append(1, b);
        assert_eq!(
            ledger.canonical_messages(1, "model-a").unwrap()[0],
            json!({"role": "user", "content": "on-a"})
        );
        assert_eq!(
            ledger.canonical_messages(1, "model-b").unwrap()[0],
            json!({"role": "user", "content": "on-b"})
        );
    }

    #[test]
    fn flush_is_line_atomic_and_turn_ordered_on_disk() {
        let dir = temp_dir("flush");
        let mut ledger = Ledger::on_disk(dir.clone()).unwrap();
        for i in 0..200 {
            ledger.append(3, turn(i, &format!("m{i}")));
        }
        let text = std::fs::read_to_string(dir.join("3.jsonl")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 200, "one complete line per turn");
        for (i, line) in lines.iter().enumerate() {
            let entry: LedgerLine = serde_json::from_str(line).unwrap();
            assert_eq!(entry.session_id, 3);
            assert_eq!(entry.turn.turn, i as u32, "disk order matches turn order");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn concurrent_appends_to_one_session_stay_line_atomic() {
        // Same-session concurrent finalizations — two in-flight requests that
        // planned the same turn — interleave on one file (the mutex
        // serializes them, but in arbitrary order). Every append must land
        // as one whole line, and the memory chain holds one of the racers.
        // Turn numbers per session only ever repeat or rise in reality (they
        // come from the record counter), so a strictly-decreasing interleave
        // is not a case to model here.
        let dir = temp_dir("concurrent");
        let ledger = std::sync::Arc::new(SharedLedger::on_disk(dir.clone()).unwrap());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let ledger = ledger.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    ledger.lock().append(42, turn(7, &format!("c{t}")));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let text = std::fs::read_to_string(dir.join("42.jsonl")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 400, "every append landed as one whole line");
        for line in &lines {
            let entry: LedgerLine = serde_json::from_str(line).unwrap();
            assert_eq!(entry.session_id, 42);
            assert_eq!(entry.turn.turn, 7);
        }
        let last = ledger.lock().last_turn(42, "gpt-4o").unwrap().clone();
        assert_eq!(last.turn, 7);
        assert!(last.request_messages[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("c"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_session_bound_evicts_oldest_first() {
        let mut ledger = Ledger::in_memory();
        let total = SESSIONS_CAP as u64 + 8;
        for s in 1..=total {
            ledger.append(s, turn(0, "x"));
        }
        // Exactly the newest SESSIONS_CAP sessions survive; the first eight
        // created are gone.
        for s in 1..=total {
            assert_eq!(
                ledger.last_turn(s, "gpt-4o").is_some(),
                s > total - SESSIONS_CAP as u64,
                "session {s} eviction state wrong"
            );
        }
    }

    #[test]
    fn reload_after_restart_recovers_the_chain_and_skips_torn_lines() {
        let dir = temp_dir("reload");
        {
            let mut ledger = Ledger::on_disk(dir.clone()).unwrap();
            ledger.append(5, turn(0, "a"));
            ledger.append(5, turn(1, "b"));
        }
        // Simulate a crash mid-append: half a line at the tail.
        let path = dir.join("5.jsonl");
        let mut torn = std::fs::read_to_string(&path).unwrap();
        torn.push_str("{\"session_id\":5,\"turn\":2,\"mod");
        std::fs::write(&path, torn).unwrap();
        // "Restart": a fresh ledger over the same directory.
        let ledger = Ledger::on_disk(dir.clone()).unwrap();
        let chain = ledger.canonical_messages(5, "gpt-4o").unwrap();
        assert_eq!(
            chain,
            vec![
                json!({"role": "user", "content": "b"}),
                json!({"role": "assistant", "content": "b"}),
            ],
            "the newest complete turn is the chain"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disk_failures_never_panic_and_keep_the_memory_chain() {
        let dir = temp_dir("diskfail");
        let mut ledger = Ledger::on_disk(dir.clone()).unwrap();
        // Make every subsequent open fail: replace the directory with a file.
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::write(&dir, b"now a file").unwrap();
        ledger.append(1, turn(0, "x"));
        assert!(
            ledger.last_turn(1, "gpt-4o").is_some(),
            "the chain lives on in memory"
        );
        std::fs::remove_file(&dir).ok();
    }

    // ---- Response assembler: OpenAI dialect --------------------------------

    fn oa_stream_chunks() -> Vec<Vec<u8>> {
        vec![
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n".to_vec(),
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n".to_vec(),
            // A usage-only terminal chunk (include_usage) must not disturb assembly.
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n".to_vec(),
            b"data: [DONE]\n\n".to_vec(),
        ]
    }

    #[test]
    fn openai_streamed_text_reassembles_to_the_message_object() {
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        for c in oa_stream_chunks() {
            a.on_chunk(&c);
        }
        assert_eq!(
            a.finish(),
            vec![json!({"role": "assistant", "content": "Hello"})]
        );
    }

    #[test]
    fn openai_events_split_across_chunks_reassemble_identically() {
        let whole: Vec<u8> = oa_stream_chunks().concat();
        // Feed in odd-sized slices so `data:` lines split across chunks.
        for size in [1usize, 3, 7, 29] {
            let mut a = ResponseAssembler::new(Dialect::OpenAi);
            for chunk in whole.chunks(size) {
                a.on_chunk(chunk);
            }
            assert_eq!(
                a.finish(),
                vec![json!({"role": "assistant", "content": "Hello"})],
                "split size {size} must not change the assembly"
            );
        }
    }

    #[test]
    fn openai_streamed_tool_calls_merge_by_index() {
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        for ev in [
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}]}}]}"#,
        ] {
            a.on_chunk(format!("data: {ev}\n\n").as_bytes());
        }
        assert_eq!(
            a.finish(),
            vec![json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "get_time", "arguments": "{}"}},
                ],
            })]
        );
    }

    #[test]
    fn openai_non_streaming_body_returns_the_message_verbatim() {
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        a.on_chunk(br#"{"choices":[{"message":{"role":"assistant","content":"Hi","refusal":null}}],"usage":{"prompt_tokens":9}}"#);
        assert_eq!(
            a.finish(),
            vec![json!({"role": "assistant", "content": "Hi", "refusal": null})]
        );
        assert!(a.full_body().is_some(), "usage can be read from the body");
    }

    // ---- Response assembler: Anthropic dialect -----------------------------

    fn an_stream_chunks() -> Vec<Vec<u8>> {
        let sse = |v: serde_json::Value| format!("data: {v}\n\n").into_bytes();
        vec![
            sse(json!({"type": "message_start", "message": {"usage": {"input_tokens": 10}}})),
            sse(
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ),
            sse(
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bon"}}),
            ),
            sse(
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "jour"}}),
            ),
            sse(json!({"type": "content_block_stop", "index": 0})),
            sse(
                json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}}),
            ),
            sse(
                json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"q\":"}}),
            ),
            sse(
                json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"x\"}"}}),
            ),
            sse(json!({"type": "content_block_stop", "index": 1})),
            sse(
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
            ),
            sse(json!({"type": "message_stop"})),
        ]
    }

    #[test]
    fn anthropic_streamed_blocks_reassemble_to_the_assistant_element() {
        let mut a = ResponseAssembler::new(Dialect::Anthropic);
        for c in an_stream_chunks() {
            a.on_chunk(&c);
        }
        assert_eq!(
            a.finish(),
            vec![json!({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Bonjour"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
                ],
            })]
        );
    }

    #[test]
    fn anthropic_thinking_blocks_carry_signature_deltas() {
        let mut a = ResponseAssembler::new(Dialect::Anthropic);
        for ev in [
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ponder"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}"#,
        ] {
            a.on_chunk(format!("event: content_block_delta\ndata: {ev}\n\n").as_bytes());
        }
        assert_eq!(
            a.finish(),
            vec![json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "ponder", "signature": "sig123"},
                    {"type": "text", "text": "ok"},
                ],
            })]
        );
    }

    #[test]
    fn anthropic_non_streaming_body_returns_content_verbatim() {
        let mut a = ResponseAssembler::new(Dialect::Anthropic);
        a.on_chunk(br#"{"content":[{"type":"text","text":"Hi"}],"usage":{"input_tokens":5}}"#);
        assert_eq!(
            a.finish(),
            vec![json!({"role": "assistant", "content": [{"type": "text", "text": "Hi"}]})]
        );
    }

    // ---- Degradation --------------------------------------------------------

    #[test]
    fn oversized_sse_lines_degrade_to_no_canonical_response() {
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        a.on_chunk(b"data: ");
        a.on_chunk(&vec![b'x'; LINE_CAP + 1]);
        assert!(a.finish().is_empty(), "degraded capture reports nothing");
    }

    #[test]
    fn an_empty_stream_has_no_canonical_response() {
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        a.on_chunk(b"data: [DONE]\n\n");
        assert!(a.finish().is_empty());
    }

    #[test]
    fn leading_blank_sse_frames_still_detect_as_sse() {
        // Some providers open with an empty keep-alive frame before the first
        // event; detection must wait for meaningful bytes, not decide "JSON".
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        a.on_chunk(b"\n\n");
        a.on_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        assert_eq!(
            a.finish(),
            vec![json!({"role": "assistant", "content": "hi"})]
        );
    }

    #[test]
    fn dialect_maps_every_backend() {
        assert_eq!(Dialect::from_backend("anthropic"), Dialect::Anthropic);
        for openai_shaped in ["openai", "vllm", "llamacpp", "mlxlm"] {
            assert_eq!(Dialect::from_backend(openai_shaped), Dialect::OpenAi);
        }
    }

    #[test]
    fn crlf_lines_and_data_without_space_reassemble() {
        // SSE permits `data:` without the space and `\r\n` line endings; both
        // tolerance branches must survive the assembler.
        let events = [
            r#"{"choices":[{"delta":{"role":"assistant","content":"a"}}]}"#,
            r#"{"choices":[{"delta":{"content":"b"}}]}"#,
        ];
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        a.on_chunk(format!("data: {}\r\n\r\n", events[0]).as_bytes());
        a.on_chunk(format!("data:{}\r\n", events[1]).as_bytes());
        a.on_chunk(b"data: [DONE]\r\n\r\n");
        assert_eq!(
            a.finish(),
            vec![json!({"role": "assistant", "content": "ab"})]
        );
    }

    #[test]
    fn multibyte_utf8_content_split_across_chunks_reassembles() {
        // Chunk boundaries that cut a codepoint in half must still reassemble:
        // lines are only ever split on `\n`, which never splits a codepoint,
        // so a complete line is always valid UTF-8.
        let payload = "héllo 🌍 ok";
        let event = format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content": payload}}]})
        );
        for size in [1usize, 2, 5, 13] {
            let mut a = ResponseAssembler::new(Dialect::OpenAi);
            for chunk in event.as_bytes().chunks(size) {
                a.on_chunk(chunk);
            }
            assert_eq!(
                a.finish(),
                vec![json!({"role": "assistant", "content": payload})],
                "split size {size} must not corrupt multibyte content"
            );
        }
    }

    #[test]
    fn canonical_messages_is_none_for_non_array_request_messages() {
        // Via the proxy this cannot happen (the 400 gate runs first), but the
        // ledger's contract is `None`, never a guessed chain.
        let mut ledger = Ledger::in_memory();
        let mut t = turn(0, "x");
        t.request_messages = json!("not-an-array");
        ledger.append(1, t);
        assert!(ledger.canonical_messages(1, "gpt-4o").is_none());
    }

    #[test]
    fn reload_with_duplicate_turn_numbers_takes_the_last_line() {
        let dir = temp_dir("dupturn");
        let path = dir.join("9.jsonl");
        let line = |marker: &str| {
            serde_json::to_string(&LedgerLine {
                session_id: 9,
                turn: turn(4, marker),
            })
            .unwrap()
        };
        std::fs::write(&path, format!("{}\n{}\n", line("first"), line("second"))).unwrap();
        let ledger = Ledger::on_disk(dir.clone()).unwrap();
        let chain = ledger.canonical_messages(9, "gpt-4o").unwrap();
        assert_eq!(
            chain[0],
            json!({"role": "user", "content": "second"}),
            "the last line for a turn wins on reload"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reload_applies_the_session_cap() {
        let dir = temp_dir("capreload");
        for s in 1..=(SESSIONS_CAP as u64 + 6) {
            let line = serde_json::to_string(&LedgerLine {
                session_id: s,
                turn: turn(0, "x"),
            })
            .unwrap();
            std::fs::write(dir.join(format!("{s}.jsonl")), format!("{line}\n")).unwrap();
        }
        let ledger = Ledger::on_disk(dir.clone()).unwrap();
        let surviving = (1..=(SESSIONS_CAP as u64 + 6))
            .filter(|s| ledger.last_turn(*s, "gpt-4o").is_some())
            .count();
        assert_eq!(surviving, SESSIONS_CAP, "reload respects the memory bound");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tool_call_delta_without_index_merges_into_the_last_slot() {
        // Real providers omit `index` on single-call streams after the first
        // fragment; the fallback must continue the last slot, not drop the
        // fragment or open a phantom one.
        let mut a = ResponseAssembler::new(Dialect::OpenAi);
        for ev in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"}"}}]}}]}"#,
        ] {
            a.on_chunk(format!("data: {ev}\n\n").as_bytes());
        }
        assert_eq!(
            a.finish(),
            vec![json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                ],
            })]
        );
    }

    #[test]
    fn a_parsed_body_round_trips_byte_identically_under_preserve_order() {
        // The ledger's authority is "the Value we forwarded serializes the
        // way we forwarded it". Compact JSON must round-trip byte-identically
        // (a fixpoint on re-parse), or the invariant is broken.
        let bytes = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi","extra":{"z":1,"a":[true,null]}}],"z":2,"a":1}"#;
        let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let once = serde_json::to_vec(&v).unwrap();
        assert_eq!(&once[..], &bytes[..], "document order is preserved");
        let again =
            serde_json::to_vec(&serde_json::from_slice::<serde_json::Value>(&once).unwrap())
                .unwrap();
        assert_eq!(once, again, "serialization is a fixpoint");
    }
}
