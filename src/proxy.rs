//! Proxy core: forward first, unbuffered SSE passthrough, observe while
//! streaming, finalize one record per request.
//!
//! The design rule (spec §Streaming): the SSE bytes the client receives are the
//! bytes we observed; the stored record must byte-match reassembly of the stream
//! by concatenation. So the hot path never parses-to-forward — it forwards
//! chunks untouched and runs observation on a side copy. Only at finalize do we
//! look at the buffered copy (bounded: usage arrives in the terminal SSE event).
//!
//! The network wiring is in [`serve`]; the pure seams ([`build_record`],
//! [`observe`], [`finalize`], [`RequestPlan`]) are tested without a socket.

use crate::adapters::Adapter;
use crate::ledger::{CanonicalTurn, Dialect, ResponseAssembler, SharedLedger};
use crate::rates::Rates;
use crate::record::{Record, SourceLabel, Status};
use crate::repair::{self, DriftReport, RepairMode};
use crate::sessions::{SessionStore, SharedSessions};
use crate::tokenize::{Message, Tokenizer};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

/// What the proxy knows about a request *before* forwarding: which session it
/// extends, its turn index, and the binding denominator.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestPlan {
    pub session_id: u64,
    pub turn: u32,
    /// `resent_history_tokens`: token count of system + all prior messages,
    /// excluding this turn's new content.
    pub resent_history_tokens: u64,
    /// This request broke a tracked session's prefix (measured as a miss).
    pub broke_prefix: bool,
    /// The request's cumulative prefix-hash sequence, as resolved for session
    /// continuity. The ledger records it per turn as a cross-check; the
    /// canonical chain's authority is the messages themselves.
    pub prefix_hashes: Vec<u64>,
}

/// Compute the plan for an incoming request from its messages.
///
/// `prefix_hashes` is this request's cumulative per-message hash sequence. The
/// matched session tells us the turn index (its record count) and the longest
/// shared prefix; everything past that shared prefix is this turn's new
/// content, so the denominator is the token count up to the shared boundary.
///
/// Token counts are measured locally here only to size the denominator. On the
/// cloud path the *provider's* figure stays authoritative for `cached_tokens`;
/// this local count is the re-sent-history measure the dashboard divides by.
pub fn plan_request(
    store: &mut SessionStore,
    tokenizer: &Tokenizer,
    messages: &[Message],
) -> RequestPlan {
    let hashes = tokenizer.prefix_hashes(messages);
    let resolution = store.resolve(&hashes);
    let session_id = resolution.session_id;
    let session = store.session(session_id);
    // Turn index = records finalized so far. Appends happen when a stream ends,
    // so genuinely concurrent requests sharing one prefix can plan the same
    // turn. Aggregate integrity is unaffected (appends stay atomic); turn
    // numbering is advisory and single-client sequential in practice.
    let turn = session.map(|s| s.records.len() as u32).unwrap_or(0);
    // Binding denominator = the re-sent history: system + all prior user,
    // assistant, and tool messages, excluding this turn's new content. This
    // turn's new content is the final message, so history is every message
    // before it. Turn 0 is cold and has no history.
    let resent_history_tokens = if turn == 0 || messages.len() <= 1 {
        0
    } else {
        tokenizer.count_messages(&messages[..messages.len() - 1]) as u64
    };

    RequestPlan {
        session_id,
        turn,
        resent_history_tokens,
        broke_prefix: resolution.broke_prefix,
        prefix_hashes: hashes,
    }
}

/// Stamp a record with the drift claim computed on the request path, plus
/// the rewrite outcome when one was applied (the actual canonicalized
/// amount in place of the estimate). Every path that records an examined
/// turn stamps the same claim, so a send failure and a clean finalize
/// agree on what repair did.
fn apply_drift_claim(record: &mut Record, report: &DriftReport, rewrite: Option<&repair::Rewrite>) {
    record.repair_mode = report.mode;
    record.matches_canonical = if report.mode == RepairMode::Off {
        None
    } else {
        Some(report.matches_canonical)
    };
    record.drift_kind = report.drift_kind;
    record.canonicalized_tokens = report.tokens_at_risk;
    if let Some(rw) = rewrite {
        record.repaired = true;
        record.canonicalized_tokens = rw.canonicalized_tokens;
    }
}

/// The raw observed figures for one turn, before cost is applied.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Observation {
    pub ttft_ms: Option<f64>,
    pub cached_tokens: u64,
    pub cache_written_tokens: u64,
    pub billed_input_tokens: u64,
}

/// Build a record from the observed response. The pure seam C1/C2 test against,
/// independent of the network. `rates` prices the no-cache counterfactual when
/// the model is known; unknown models simply carry no cost.
pub fn build_record(
    plan: &RequestPlan,
    obs: Observation,
    model: &str,
    rates: &Rates,
    source: SourceLabel,
    complete: bool,
) -> Record {
    // Local engines report cached tokens in the engine's own token space, which
    // can drift from our tokenizer; clamp to the history span so the ratio never
    // exceeds 100%. Cloud counts are provider-reported and shown as-is.
    let cached = match source {
        SourceLabel::EngineMeasured => obs.cached_tokens.min(plan.resent_history_tokens),
        SourceLabel::ProviderReported => obs.cached_tokens,
        SourceLabel::NoCacheTruth => 0,
    };
    // An incomplete turn (cut stream, non-2xx) has partial usage; a cost derived
    // from it would be a fabricated bill. Report no cost rather than a wrong one.
    let (cost_usd, cost_saved_usd) = if complete {
        (
            rates
                .lookup(model)
                .map(|r| r.input_cost(obs.billed_input_tokens)),
            rates.cost_saved(
                model,
                cached,
                plan.resent_history_tokens,
                obs.cache_written_tokens,
            ),
        )
    } else {
        (None, None)
    };
    Record {
        session_id: plan.session_id,
        turn: plan.turn,
        status: if complete {
            Status::Complete
        } else {
            Status::Incomplete
        },
        source,
        ttft_ms: obs.ttft_ms,
        cached_tokens: cached,
        cache_written_tokens: obs.cache_written_tokens,
        resent_history_tokens: plan.resent_history_tokens,
        billed_input_tokens: obs.billed_input_tokens,
        broke_prefix: plan.broke_prefix,
        cost_usd,
        cost_saved_usd,
        // The pure seam carries no repair claim; the finalizer patches these
        // from the drift report computed on the request path.
        repair_mode: RepairMode::Off,
        repaired: false,
        matches_canonical: None,
        drift_kind: None,
        canonicalized_tokens: 0,
        // Patched by the finalizer when breakpoint management ran.
        breakpoint_count: None,
    }
}

/// The cache signal an adapter reads from a response. `response_body` may be a
/// single JSON document (non-streaming) or a buffered SSE tail; the
/// usage-bearing event wins. Callers that also need the reduced document (for
/// billing) reduce the tail once and call [`observe_doc`] directly.
pub fn observe<A: Adapter>(adapter: &A, response_body: &[u8]) -> (u64, u64, SourceLabel) {
    observe_doc(adapter, &last_json_event(response_body))
}

/// [`observe`] over an already-reduced JSON document.
fn observe_doc<A: Adapter>(adapter: &A, doc: &[u8]) -> (u64, u64, SourceLabel) {
    let sig = adapter.cache_signal(doc);
    (
        sig.cached_tokens,
        sig.written_tokens,
        sig.source.unwrap_or_else(|| adapter.source()),
    )
}

/// Reduce a body to the most informative JSON document: the whole body if it is
/// one JSON object, else the single most informative SSE event.
///
/// Selection order matters per provider:
/// - OpenAI's cache/billing figures arrive in the *terminal* `usage` chunk, so
///   the last event carrying `usage` wins.
/// - Anthropic's cache figures arrive on the *first* event (`message_start`,
///   under `message.usage`); its terminal `message_delta` carries only
///   `output_tokens`. So an event carrying cache fields is preferred regardless
///   of position.
///
/// Precedence: a whole JSON body; else the last event with a cache field; else
/// the last event with `usage`; else the last parseable event.
fn last_json_event(body: &[u8]) -> Vec<u8> {
    if body.starts_with(b"{") && serde_json::from_slice::<serde_json::Value>(body).is_ok() {
        return body.to_vec();
    }
    let mut cache_event: Option<Vec<u8>> = None;
    let mut usage_event: Option<Vec<u8>> = None;
    let mut fallback: Option<Vec<u8>> = None;
    for line in body.split(|&b| b == b'\n').rev() {
        // SSE allows `data:` with or without the space.
        let line = line
            .strip_prefix(b"data: ")
            .or_else(|| line.strip_prefix(b"data:"))
            .unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line == b"[DONE]" || line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<serde_json::Value>(line) {
            Ok(v) => {
                if has_cache_field(&v) {
                    cache_event.get_or_insert_with(|| line.to_vec());
                }
                if v.get("usage").is_some() || v.pointer("/message/usage").is_some() {
                    usage_event.get_or_insert_with(|| line.to_vec());
                }
                if fallback.is_none() {
                    fallback = Some(line.to_vec());
                }
            }
            Err(_) => continue,
        }
    }
    cache_event.or(usage_event).or(fallback).unwrap_or_default()
}

/// Whether a reduced event exposes any cache figure, in either provider's shape
/// (`usage.*` or Anthropic's nested `message.usage.*`).
fn has_cache_field(v: &serde_json::Value) -> bool {
    let paths = [
        "/usage/cache_read_input_tokens",
        "/usage/cache_creation_input_tokens",
        "/usage/prompt_tokens_details/cached_tokens",
        "/message/usage/cache_read_input_tokens",
        "/message/usage/cache_creation_input_tokens",
    ];
    paths
        .iter()
        .any(|p| v.pointer(p).map(|n| !n.is_null()).unwrap_or(false))
}

/// Incremental stream observer. Fed each forwarded chunk in order, untouched;
/// tracks TTFT and accumulates a bounded copy for the finalize parse.
///
/// The client stream is *never* delayed by this: chunk bytes are cloned for
/// observation and the originals are forwarded as-is. See [`serve`].
pub struct StreamObserver {
    started: Instant,
    ttft_ms: Option<f64>,
    saw_first_byte: bool,
    tail: Vec<u8>,
    /// The most informative usage-bearing event seen so far, retained across
    /// tail eviction. Usage can arrive early (Anthropic `message_start`) while
    /// large content follows, so it must not be lost to the tail cap.
    usage_doc: Option<Vec<u8>>,
    /// Byte offset up to which `tail` has already been scanned for usage.
    scanned: usize,
    /// Cap the retained copy; the tail is a fallback for content-free streams.
    cap: usize,
    /// Incremental reassembly of the assistant message(s) for the ledger.
    /// Retains only the message being assembled (never the whole stream), so
    /// observation stays bounded no matter how long the response runs.
    assembler: ResponseAssembler,
}

impl StreamObserver {
    pub fn new() -> Self {
        Self::for_dialect(Dialect::OpenAi)
    }

    /// An observer whose response capture speaks `dialect` (the upstream
    /// backend's wire shape; see [`Dialect::from_backend`]).
    pub fn for_dialect(dialect: Dialect) -> Self {
        Self {
            started: Instant::now(),
            ttft_ms: None,
            saw_first_byte: false,
            tail: Vec::new(),
            usage_doc: None,
            scanned: 0,
            cap: 64 * 1024,
            assembler: ResponseAssembler::new(dialect),
        }
    }

    /// Observe one forwarded chunk. Called with the exact bytes being sent to
    /// the client; must not mutate them.
    pub fn on_chunk(&mut self, chunk: &[u8]) {
        if !self.saw_first_byte && !chunk.is_empty() {
            self.saw_first_byte = true;
            self.ttft_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
        }
        self.assembler.on_chunk(chunk);
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > self.cap {
            let drop = self.tail.len() - self.cap;
            self.tail.drain(..drop);
            self.scanned = self.scanned.saturating_sub(drop);
        }
        // Capture any usage-bearing event as it passes, before tail eviction can
        // drop it. Scan only the newly appended bytes (plus a 256B overlap for
        // events split across chunk boundaries), not the whole growing tail.
        if has_usage_or_cache_in(&self.tail[self.scanned.saturating_sub(256)..]) {
            self.usage_doc = Some(last_json_event(&self.tail));
        }
        self.scanned = self.tail.len();
    }

    /// The reassembled assistant message(s) exactly as the provider returned
    /// them — what the ledger remembers for this turn. Empty when the
    /// response could not be reassembled; absence is a fact, never filled.
    pub fn response_messages(&self) -> Vec<serde_json::Value> {
        self.assembler.finish()
    }

    /// Finalize the observation into a record. `complete` is false when the
    /// stream ended early (client disconnect, upstream error mid-stream).
    /// `engine_cached` is a per-turn cache figure measured out-of-band (the
    /// vLLM `/metrics` delta); when present it overrides the body-derived
    /// count and labels the source `EngineMeasured`.
    pub fn finalize<A: Adapter>(
        &self,
        plan: &RequestPlan,
        adapter: &A,
        model: &str,
        rates: &Rates,
        complete: bool,
        engine_cached: Option<u64>,
    ) -> Record {
        // Prefer the retained usage event (SSE); else the whole retained JSON
        // body (non-streaming responses are held in full by the assembler);
        // else the bounded tail.
        let doc = self
            .usage_doc
            .clone()
            .or_else(|| self.assembler.full_body().map(<[u8]>::to_vec))
            .unwrap_or_else(|| last_json_event(&self.tail));
        let (mut cached_tokens, cache_written_tokens, mut source) = observe_doc(adapter, &doc);
        if let Some(n) = engine_cached {
            cached_tokens = n;
            source = SourceLabel::EngineMeasured;
        }
        let billed = billed_from_doc(&doc).unwrap_or(plan.resent_history_tokens);
        let obs = Observation {
            ttft_ms: self.ttft_ms,
            cached_tokens,
            cache_written_tokens,
            billed_input_tokens: billed,
        };
        build_record(plan, obs, model, rates, source, complete)
    }
}

/// Whether the accumulated buffer contains any usage- or cache-bearing `data:`
/// event. Cheap substring check; the exact event is chosen by `last_json_event`.
fn has_usage_or_cache_in(buf: &[u8]) -> bool {
    let needle = |n: &[u8]| buf.windows(n.len()).any(|w| w == n);
    needle(b"\"usage\"")
        || needle(b"cache_read_input_tokens")
        || needle(b"cache_creation_input_tokens")
        || needle(b"cached_tokens")
}

/// Records exactly one turn, at whichever end comes first: the upstream stream
/// finishing (`finish`) or the response body being dropped when the client
/// disconnects mid-stream (`Drop`). A finalize written only after the read loop
/// would never run on a dropped stream, losing the turn. `done` makes it
/// idempotent so the explicit finish and the drop never double-append.
struct Finalizer<A: Adapter> {
    observer: StreamObserver,
    plan: RequestPlan,
    adapter: Arc<A>,
    model: String,
    rates: Rates,
    sessions: Arc<SharedSessions>,
    /// The canonical ledger: complete turns are remembered here, exactly as
    /// forwarded and received.
    ledger: Arc<SharedLedger>,
    /// The `messages` array exactly as forwarded upstream this turn (already
    /// reflecting any proxy-side injection). Captured before forwarding.
    as_sent_messages: serde_json::Value,
    /// The drift report computed on the request path. Patched into the
    /// record at finalize; dry-run never lets it touch the request.
    drift_report: DriftReport,
    /// The rewrite that was applied (`on` mode, drift present, chain
    /// extendable). `None` means the request went out untouched.
    rewrite: Option<repair::Rewrite>,
    /// Breakpoint management's outcome (`None` when the feature is off or
    /// the backend is not Anthropic). Patches the record and the ledger's
    /// echo-detection count.
    breakpoints: Option<crate::breakpoints::Managed>,
    /// False once an upstream read error was seen.
    complete: bool,
    done: bool,
    /// vLLM metrics sampling: the client, the `/metrics` URL, and the counter
    /// snapshot taken just before the request. `None` for other backends.
    metrics: Option<(reqwest::Client, String, crate::adapters::vllm::PromCounters)>,
}

impl<A: Adapter> Finalizer<A> {
    /// Record the turn once. `complete` is true only when the upstream stream
    /// finished cleanly and answered 2xx.
    fn finish(&mut self, complete: bool) {
        self.record(complete, None);
    }

    /// Record a vLLM turn, using the `/metrics` counter delta as the cache
    /// figure when one is available. The delta is the only per-turn cache
    /// measurement vLLM exposes (its response body carries none). Async because
    /// it takes the "after" scrape; only called on the clean-completion path,
    /// where the engine has finished updating its counters. A cut stream is an
    /// Incomplete turn and takes no delta.
    async fn finish_with_metrics(&mut self, complete: bool) {
        let delta = match &self.metrics {
            Some((client, url, before)) if complete => sample_prom(client, url)
                .await
                .map(|after| crate::adapters::vllm::PromCounters::delta_hits(*before, after)),
            _ => None,
        };
        self.record(complete, delta);
    }

    fn record(&mut self, complete: bool, engine_cached: Option<u64>) {
        if self.done {
            return;
        }
        self.done = true;
        let mut record = self.observer.finalize(
            &self.plan,
            self.adapter.as_ref(),
            &self.model,
            &self.rates,
            complete,
            engine_cached,
        );
        apply_drift_claim(&mut record, &self.drift_report, self.rewrite.as_ref());
        if let Some(managed) = self.breakpoints {
            record.breakpoint_count = Some(managed.total as u64);
        }
        if complete {
            // The canonical turn: the messages exactly as forwarded, extended
            // by the assistant message(s) exactly as received. Only complete
            // turns enter the chain — an incomplete turn's partial response
            // was never a message the client could re-send.
            let turn = CanonicalTurn {
                turn: self.plan.turn,
                model: self.model.clone(),
                request_messages: self.as_sent_messages.clone(),
                response_messages: self.observer.response_messages(),
                prefix_hashes: self.plan.prefix_hashes.clone(),
                // What WE placed (a declined pass touched nothing and must
                // not claim the client's hints as ours: echoed counts are
                // how the next turn tells the two apart).
                breakpoints: match self.breakpoints {
                    Some(m) if !m.declined => m.total as u64,
                    _ => 0,
                },
            };
            self.ledger.lock().append(self.plan.session_id, turn);
        }
        crate::export::log_finalize(&record);
        self.sessions.lock().append(record);
    }
}

impl<A: Adapter> Drop for Finalizer<A> {
    fn drop(&mut self) {
        // The generator was dropped mid-suspension: either the read loop
        // finished and called `finish` (then `done` short-circuits) or the
        // client disconnected before the stream ended. A disconnect is not a
        // completed turn — record it Incomplete so a partial stream never
        // counts as a measured turn.
        self.finish(false);
    }
}

impl Default for StreamObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Read `billed_input_tokens` from an already-reduced JSON document, across
/// dialects: OpenAI reports `usage.prompt_tokens`; Anthropic reports
/// `usage.input_tokens` (flat for a whole message, or under `message.usage`
/// for a streamed `message_start`). The provider's own count is authoritative;
/// the caller falls back to the local denominator only when none is present.
fn billed_from_doc(doc: &[u8]) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(doc).ok()?;
    v.pointer("/usage/prompt_tokens")
        .or_else(|| v.pointer("/usage/input_tokens"))
        .or_else(|| v.pointer("/message/usage/input_tokens"))
        .and_then(|n| n.as_u64())
}

/// State shared by every request handler.
pub struct AppState<A: Adapter> {
    pub adapter: Arc<A>,
    pub tokenizer: Tokenizer,
    pub sessions: Arc<SharedSessions>,
    /// The canonical ledger (see [`crate::ledger`]). Content-bearing by
    /// design; local-only. `--no-ledger` runs it in memory only.
    pub ledger: Arc<SharedLedger>,
    pub rates: Rates,
    pub upstream_url: String,
    pub client: reqwest::Client,
    /// Set `stream_options.include_usage` on OpenAI-dialect streaming requests
    /// so the terminal usage chunk (cache figures) is emitted. On by default;
    /// disable with `--no-inject-usage` for strict pass-through.
    pub inject_usage: bool,
    /// The repair mode. Dry-run is the product default; `off` restores the
    /// pure-measurement behavior. `on` (rewriting) lands with the next batch.
    pub repair: RepairMode,
    /// Anthropic-only, opt-in (`--manage-breakpoints`): place
    /// `cache_control` breakpoints per the incremental-breakpoint guidance.
    pub manage_breakpoints: bool,
    /// With `--manage-breakpoints`: re-derive breakpoints even over
    /// client-placed ones (`--force-breakpoints`).
    pub force_breakpoints: bool,
}

/// The proxy's operating configuration (everything the CLI tunes).
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Inject `stream_options.include_usage` on OpenAI-dialect streams.
    pub inject_usage: bool,
    /// The repair mode (dry-run is the product default).
    pub repair: RepairMode,
    /// Anthropic breakpoint management (opt-in; see
    /// [`crate::breakpoints`]).
    pub manage_breakpoints: bool,
    /// Override client-placed breakpoints (requires `manage_breakpoints`).
    pub force_breakpoints: bool,
}

/// Run the proxy. Forwards to the upstream, streams the response through
/// unbuffered, and finalizes one record per request.
pub async fn serve<A: Adapter + 'static>(
    adapter: A,
    tokenizer: Tokenizer,
    rates: Rates,
    ledger: Arc<SharedLedger>,
    upstream_url: String,
    options: ServeOptions,
    listener: tokio::net::TcpListener,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = adapter.name();
    let tokenizer_label = tokenizer.label().to_string();
    let state = Arc::new(AppState {
        adapter: Arc::new(adapter),
        tokenizer,
        sessions: Arc::new(SharedSessions::new()),
        ledger,
        rates,
        upstream_url,
        repair: options.repair,
        inject_usage: options.inject_usage,
        manage_breakpoints: options.manage_breakpoints,
        force_breakpoints: options.force_breakpoints,
        // A connect timeout fails fast on an unreachable upstream without
        // capping a legitimate long-lived SSE stream. No total request timeout:
        // streams are open-ended by design.
        client: reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("reqwest client"),
    });

    let listener = listener;
    tracing::info!(
        addr = %listener.local_addr()?,
        backend,
        tokenizer = %tokenizer_label,
        "cachemax listening"
    );
    axum::serve(listener, router(state)).await?;
    Ok(())
}

/// The proxy's axum router. Shared by [`serve`] and integration tests so both
/// exercise one construction path. The dashboard (`/`) and its data endpoints
/// live on the same server as the API.
pub fn router<A: Adapter + 'static>(state: Arc<AppState<A>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat::<A>))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route("/", axum::routing::get(serve_dashboard))
        .route("/api/state", axum::routing::get(dashboard_state::<A>))
        .route("/api/export", axum::routing::get(export_session::<A>))
        // Axum's default 2 MB body cap silently 413s a legitimately large
        // long-context prompt before the handler runs, so the turn is never
        // measured. Raise it well past real context sizes. A prompt larger than
        // this, or one the upstream rejects, still records Incomplete.
        .layer(axum::extract::DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state)
}

/// Upper bound on an accepted request body. Long-context prompts run to many
/// megabytes; 32 MB leaves headroom without unbounded buffering.
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Serve the embedded single-file dashboard.
async fn serve_dashboard() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(crate::dashboard::DASHBOARD_HTML))
        .unwrap()
}

/// The live dashboard snapshot: the most recently active session's view.
async fn dashboard_state<A: Adapter + 'static>(State(state): State<Arc<AppState<A>>>) -> Response {
    let (records, session_count, live) = {
        let guard = state.sessions.lock();
        let live = guard
            .most_recent()
            .map(|s| !s.records.is_empty())
            .unwrap_or(false);
        let records = guard
            .most_recent()
            .map(|s| s.records.clone())
            .unwrap_or_default();
        (records, guard.len(), live)
    };
    let view = crate::dashboard::view(&records, live, session_count);
    match serde_json::to_vec(&view) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(bytes))
            .unwrap(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Export the most recently active session as metrics-only JSONL.
async fn export_session<A: Adapter + 'static>(State(state): State<Arc<AppState<A>>>) -> Response {
    let (records, id) = {
        let guard = state.sessions.lock();
        match guard.most_recent() {
            Some(s) => (s.records.clone(), s.id),
            None => (Vec::new(), 0),
        }
    };
    match crate::export::to_jsonl(&records) {
        Ok(jsonl) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-ndjson")
            .header(
                "content-disposition",
                format!("attachment; filename=\"cachemax-{id}.jsonl\""),
            )
            .body(Body::from(jsonl))
            .unwrap(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Request headers forwarded to the upstream verbatim. Auth must pass through
/// or every cloud call 401s; the rest are the provider-identification headers
/// OpenRouter and similar gateways read. Everything else is dropped.
const FORWARD_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "anthropic-version",
    "openai-organization",
    "openai-project",
    "http-referer",
    "x-title",
];

/// The versioned API base for `base`. `base` is the API base and may already
/// end in `/v1` (the documented form: `https://api.openai.com/v1`, OpenRouter's
/// `.../api/v1`) or omit it (a bare host). Append `/v1` only when missing, so a
/// base that already carries it is not doubled. Shared by `serve` and `check`
/// so both agree on the version segment.
pub fn versioned_base(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1") {
        base.to_string()
    } else {
        format!("{base}/v1")
    }
}

/// The `chat/completions` URL for `base`. See [`versioned_base`].
fn upstream_chat_url(base: &str) -> String {
    format!("{}/chat/completions", versioned_base(base))
}

/// The engine's Prometheus endpoint, for backends that expose one (`vllm`).
/// `/metrics` sits at the server origin, not under the OpenAI `/v1` prefix.
fn upstream_metrics_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    let origin = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
    format!("{origin}/metrics")
}

/// Sample the engine's Prometheus counters once. `None` on any failure; a
/// missing scrape is a missing measurement, never a wrong one.
async fn sample_prom(
    client: &reqwest::Client,
    url: &str,
) -> Option<crate::adapters::vllm::PromCounters> {
    let text = client.get(url).send().await.ok()?.text().await.ok()?;
    Some(crate::adapters::vllm::PromCounters::parse(&text))
}

/// Ensure an OpenAI-dialect streaming request reports usage.
///
/// OpenAI only emits the terminal `usage` chunk (where `cached_tokens` lives)
/// when the request sets `stream_options.include_usage`. The proxy measures
/// cache reuse, so it asks for usage on the client's behalf: if the body is a
/// JSON object with `"stream": true` and no `stream_options.include_usage`, set
/// it. Non-streaming bodies and bodies already opting in are left unchanged.
///
/// Mutates `doc` in place and returns whether anything changed. The caller
/// serializes once, after every mutation (injection, repair rewrite) has
/// applied — never once per mutation.
fn ensure_usage_requested(doc: &mut serde_json::Value) -> bool {
    if doc.get("stream").and_then(|s| s.as_bool()) != Some(true) {
        return false;
    }
    if doc
        .pointer("/stream_options/include_usage")
        .and_then(|b| b.as_bool())
        .unwrap_or(false)
    {
        return false;
    }
    if !doc
        .get("stream_options")
        .map(|s| s.is_object())
        .unwrap_or(false)
    {
        doc["stream_options"] = serde_json::json!({});
    }
    doc["stream_options"]["include_usage"] = serde_json::Value::Bool(true);
    true
}

/// The request handler: plan, forward, stream through, observe, finalize.
pub async fn handle_chat<A: Adapter + 'static>(
    State(state): State<Arc<AppState<A>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // One parse of the request body. The same document serves session
    // planning (flattened messages), the canonical ledger (the as-forwarded
    // `messages` value), and usage injection (which mutates it before the
    // forwarded bytes are serialized from it).
    let Ok(mut doc) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (StatusCode::BAD_REQUEST, "could not parse messages").into_response();
    };
    let Some(messages) = messages_from_doc(&doc) else {
        return (StatusCode::BAD_REQUEST, "could not parse messages").into_response();
    };
    let model = model_from_doc(&doc);
    // Counted before any proxy mutation: what the CLIENT carried. Breakpoint
    // management compares this against what the proxy placed last turn to
    // tell client-authored hints from its own echoed back.
    let client_breakpoints = crate::breakpoints::count(&doc);

    let mut plan = {
        let mut guard = state.sessions.lock();
        plan_request(&mut guard, &state.tokenizer, &messages)
    };

    // The repair stage: classify the re-sent history against the canonical
    // chain, then — only in `on` mode — rewrite drifted elements to the
    // canonical serialization. Dry-run (the default) and off never touch a
    // byte. The per-request header overrides the configured mode.
    let effective_mode = match headers
        .get("x-cachemax-repair")
        .and_then(|v| v.to_str().ok())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("on") => RepairMode::On,
        Some("off") => RepairMode::Off,
        Some(other) => {
            tracing::warn!(value = %other, "ignoring unknown x-cachemax-repair override");
            state.repair
        }
        None => state.repair,
    };
    let (drift_report, mut rewrite) = if effective_mode == RepairMode::Off {
        (DriftReport::unexamined(RepairMode::Off), None)
    } else {
        // Which chain does this request extend? Its own session's, or — when
        // the session forked (truncated or re-based leading history shares
        // no prefix-hash, so the store allocated a new session) — a probed
        // chain: the most recently written one for this model, accepted
        // below only when the client's history actually reads as a re-send
        // of it. A model switch gets no cross-chain fallback: chains are
        // per model by design.
        let (chain, model_switched, probed) = {
            let ledger = state.ledger.lock();
            match ledger.canonical_messages(plan.session_id, &model) {
                Some(c) => (Some(c), false, false),
                None if ledger.session_has_chains(plan.session_id) => (None, true, false),
                None => (
                    ledger
                        .most_recent_chain_session(&model)
                        .and_then(|s| ledger.canonical_messages(s, &model)),
                    false,
                    true,
                ),
            }
        };
        let mut client_values = doc
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let mut classification = repair::classify_turn(
            &client_values,
            chain.as_deref(),
            model_switched,
            &state.tokenizer,
        );
        if probed && !(classification.semantic_break.is_none() && classification.equivalent_run > 0)
        {
            // The probed chain is only a candidate — the most recently
            // written chain *for the model*, nothing more. It stands for
            // this conversation only when the client's history reads as a
            // re-send of it: every element with a canonical counterpart
            // aligns, and at least one does. A request that merely shares
            // a leading span with a foreign conversation (the same
            // framework system prompt is the norm) breaks somewhere inside
            // — that is a new conversation's first turn, honestly reported
            // and never rewritten.
            classification = repair::classify_turn(&client_values, None, false, &state.tokenizer);
        }
        let report = repair::report(&classification, effective_mode);
        let mut rewrite = None;
        if effective_mode == RepairMode::On {
            if let Some(chain) = chain.as_deref() {
                if let Some(rw) = repair::apply_canonical(
                    &mut client_values,
                    &classification,
                    chain,
                    &state.tokenizer,
                ) {
                    // The rewrite log (trust): what, why, how much. Metadata
                    // only — the content stays in the local ledger.
                    tracing::info!(
                        target: "cachemax_repair",
                        session = plan.session_id,
                        turn = plan.turn,
                        kind = ?report.drift_kind,
                        elements = rw.elements_replaced,
                        tokens = rw.canonicalized_tokens,
                        "rewrote drifted history to canonical"
                    );
                    doc["messages"] = serde_json::Value::Array(client_values);
                    rewrite = Some(rw);
                }
            }
        }
        (report, rewrite)
    };

    // Anthropic breakpoint management (opt-in): the last body mutation
    // before the single serialization. Hints go on the last system block
    // and the last user/tool-result blocks; the ledger remembers what went
    // out, so echoed-back placements re-derive instead of reading as
    // client-managed. Other backends never take this path.
    let breakpoints =
        (state.manage_breakpoints && state.adapter.name() == "anthropic").then(|| {
            let ours_last_turn = state
                .ledger
                .lock()
                .last_turn(plan.session_id, &model)
                .map_or(0, |t| t.breakpoints);
            crate::breakpoints::manage(
                &mut doc,
                state.force_breakpoints,
                client_breakpoints,
                ours_last_turn as usize,
            )
        });
    // A declined pass touched nothing; a managed pass re-serializes.
    let breakpoints_changed = breakpoints.is_some_and(|m| !m.declined);

    // Forward first. The request body is passed through untouched unless
    // usage injection applies; auth and provider-identification headers are
    // forwarded so cloud keys keep working.
    let url = upstream_chat_url(&state.upstream_url);
    // vLLM exposes no per-request cache figure in the response; the only
    // measurement is the delta of its `/metrics` counters across the request.
    // Snapshot before we forward, for backends that have the endpoint.
    let metrics = if state.adapter.name() == "vllm" {
        let url = upstream_metrics_url(&state.upstream_url);
        sample_prom(&state.client, &url)
            .await
            .map(|before| (state.client.clone(), url, before))
    } else {
        None
    };
    let mut req = state
        .client
        .post(&url)
        .header("content-type", "application/json");
    for name in FORWARD_HEADERS {
        if let Some(value) = headers.get(*name) {
            req = req.header(*name, value);
        }
    }
    // Only the OpenAI dialect understands `stream_options`; Anthropic's
    // Messages API would reject it, so never inject there.
    let inject = state.inject_usage && matches!(state.adapter.name(), "openai" | "vllm");
    // Every mutation applies before one serialization: usage injection, a
    // repair rewrite, and breakpoint management each may change the
    // document, and none pays for a second pass.
    let usage_mutated = inject && ensure_usage_requested(&mut doc);
    let (forwarded, as_sent_messages) = if usage_mutated || rewrite.is_some() || breakpoints_changed
    {
        match serde_json::to_vec(&doc) {
            Ok(bytes) => (
                Bytes::from(bytes),
                doc.get("messages")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            ),
            Err(_) => {
                // Unreachable with a document parsed from the request, but
                // degrade honestly: forward the client's bytes untouched,
                // claim no rewrite whose forwarded form could not be
                // produced, and record the messages that actually went
                // out — read back from the client's own bytes.
                rewrite = None;
                let as_sent = serde_json::from_slice::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|d| d.get("messages").cloned())
                    .unwrap_or_default();
                (body.clone(), as_sent)
            }
        }
    } else {
        (
            body.clone(),
            doc.get("messages")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
    };
    if rewrite.is_some() {
        // The canonical turn's prefix hashes must describe what was
        // forwarded, not what the client sent: a rewrite changes the
        // flattened history that turn claims to extend, and its hashes
        // must agree with its (rewritten) request messages.
        if let Some(sent) = messages_from_doc(&doc) {
            plan.prefix_hashes = state.tokenizer.prefix_hashes(&sent);
        }
    }
    let upstream = match req.body(forwarded).send().await {
        Ok(r) => r,
        Err(e) => {
            let mut record = build_record(
                &plan,
                Observation::default(),
                &model,
                &state.rates,
                state.adapter.source(),
                false,
            );
            // The turn was examined; the failed-forward record still carries
            // the drift claim — and the rewrite claim, when one was applied
            // before the send failed — exactly as a clean finalize would
            // (it stays excluded from aggregates as Incomplete).
            apply_drift_claim(&mut record, &drift_report, rewrite.as_ref());
            if let Some(managed) = breakpoints {
                record.breakpoint_count = Some(managed.total as u64);
            }
            crate::export::log_finalize(&record);
            state.sessions.lock().append(record);
            return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response();
        }
    };
    let status = upstream.status();
    // A non-2xx upstream answer is not a measured turn: recording it would
    // fabricate a billed count for a rejected/failed request. Mark it
    // incomplete so it is excluded from the aggregate.
    let upstream_ok = status.is_success();
    // Forward the upstream's content type (SSE or JSON); fall back to SSE only
    // when the upstream did not name one.
    let content_type = upstream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();
    let mut upstream_stream = upstream.bytes_stream();

    // Unbuffered passthrough: each chunk is forwarded as it arrives; a clone is
    // handed to the observer, which never blocks the forward path.
    let adapter = state.adapter.clone();
    let sessions = state.sessions.clone();
    let ledger = state.ledger.clone();
    let dialect = Dialect::from_backend(state.adapter.name());
    let rates = state.rates.clone();
    let stream = async_stream::stream! {
        // The finalizer records exactly once, whichever comes first: the end of
        // the upstream stream (complete = upstream finished and was 2xx) or the
        // generator being dropped when the client disconnects mid-stream. A
        // plain tail after the loop would never run on disconnect, losing the
        // turn entirely; the guard's Drop finalizes it as Incomplete instead.
        let mut fin = Finalizer {
            observer: StreamObserver::for_dialect(dialect),
            plan,
            adapter,
            model,
            rates,
            sessions,
            ledger,
            as_sent_messages,
            drift_report,
            rewrite,
            breakpoints,
            complete: true,
            done: false,
            metrics,
        };
        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    fin.observer.on_chunk(&bytes);
                    yield Ok::<Bytes, std::io::Error>(bytes);
                }
                Err(_) => {
                    fin.complete = false;
                    break;
                }
            }
        }
        let complete = fin.complete && upstream_ok;
        fin.finish_with_metrics(complete).await;
    };

    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Extract messages from an already-parsed OpenAI-dialect request body.
/// Dialect-neutral output.
fn messages_from_doc(doc: &serde_json::Value) -> Option<Vec<Message>> {
    let arr = doc.get("messages")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for m in arr {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let text = flatten_content(m.get("content"));
        out.push(Message {
            role: role.to_string(),
            text,
        });
    }
    Some(out)
}

/// Extract the model name from an already-parsed request body (for rate
/// lookup and the ledger's per-model chain).
fn model_from_doc(doc: &serde_json::Value) -> String {
    doc.get("model")
        .and_then(|m| m.as_str())
        .map(String::from)
        .unwrap_or_default()
}

/// Flatten OpenAI content — a string, or an array of `{type,text}` parts.
/// Shared with the repair classifier (content-shape equivalence).
pub(crate) fn flatten_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::openai::OpenAiAdapter;

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            text: text.into(),
        }
    }

    #[test]
    fn anthropic_streaming_picks_the_event_with_cache_fields() {
        // Real Anthropic SSE: cache fields live on message_start (nested under
        // message.usage); the terminal message_delta has only output_tokens.
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":2100,\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":300,\"output_tokens\":1}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let picked = last_json_event(body);
        let (read, creation) =
            crate::adapters::anthropic::AnthropicAdapter::split(&picked).expect("cache fields");
        assert_eq!((read, creation), (900, 300));
    }

    #[test]
    fn openai_streaming_still_picks_the_terminal_usage_chunk() {
        // OpenAI's cache figure is in the last usage-bearing chunk; unchanged.
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":2140,\"prompt_tokens_details\":{\"cached_tokens\":1455}}}\n\ndata: [DONE]\n\n";
        let picked = last_json_event(body);
        assert_eq!(billed_from_doc(&picked), Some(2140));
    }

    #[test]
    fn anthropic_billed_input_reads_input_tokens_both_shapes() {
        let flat = br#"{"usage":{"input_tokens":2100,"cache_read_input_tokens":900}}"#;
        assert_eq!(billed_from_doc(flat), Some(2100));
        let nested = br#"{"type":"message_start","message":{"usage":{"input_tokens":2100}}}"#;
        assert_eq!(billed_from_doc(nested), Some(2100));
    }

    #[test]
    fn usage_is_requested_on_openai_streaming_only() {
        let run = |body: &'static [u8]| {
            let mut doc: serde_json::Value = serde_json::from_slice(body).unwrap();
            let mutated = ensure_usage_requested(&mut doc);
            (mutated, serde_json::to_vec(&doc).unwrap())
        };
        let (mutated, out) = run(br#"{"model":"gpt-4o","stream":true,"messages":[]}"#);
        assert!(mutated);
        // Injection preserves the rest of the document's key order, and the
        // caller's single serialization carries it.
        assert_eq!(
            out,
            br#"{"model":"gpt-4o","stream":true,"messages":[],"stream_options":{"include_usage":true}}"#.to_vec()
        );

        // Non-streaming untouched.
        let (mutated, out) = run(br#"{"model":"gpt-4o","messages":[]}"#);
        assert!(!mutated);
        assert_eq!(out, br#"{"model":"gpt-4o","messages":[]}"#.to_vec());

        // Already opted in: unchanged, not duplicated.
        let (mutated, out) = run(br#"{"stream":true,"stream_options":{"include_usage":true}}"#);
        assert!(!mutated);
        assert_eq!(
            out,
            br#"{"stream":true,"stream_options":{"include_usage":true}}"#.to_vec()
        );
    }

    #[test]
    fn upstream_url_never_doubles_the_version_segment() {
        // The documented form already includes /v1; appending the full path
        // would yield /v1/v1/chat/completions and 404 against a real server.
        assert_eq!(
            upstream_chat_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            upstream_chat_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        // A bare host still gets the version segment.
        assert_eq!(
            upstream_chat_url("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
        assert_eq!(
            upstream_chat_url("http://127.0.0.1:8080/"),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
    }

    #[test]
    fn versioned_base_matches_between_serve_and_check() {
        // `serve` (chat/completions) and `check` (/models) must agree on /v1.
        for base in ["https://api.openai.com/v1", "http://127.0.0.1:8080"] {
            assert!(versioned_base(base).ends_with("/v1"));
            assert!(!versioned_base(base).ends_with("/v1/v1"));
        }
        assert_eq!(versioned_base("http://h/v1"), "http://h/v1");
        assert_eq!(versioned_base("http://h"), "http://h/v1");
    }

    #[test]
    fn a_dropped_stream_finalizes_incomplete() {
        let plan = RequestPlan {
            session_id: 1,
            turn: 1,
            resent_history_tokens: 1550,
            broke_prefix: false,
            prefix_hashes: Vec::new(),
        };
        let obs = Observation {
            ttft_ms: Some(120.0),
            ..Default::default()
        };
        let r = build_record(
            &plan,
            obs,
            "gpt-4o",
            &Rates::builtin(),
            SourceLabel::ProviderReported,
            false,
        );
        assert_eq!(r.status, Status::Incomplete);
    }

    #[test]
    fn observe_routes_through_the_adapter() {
        let body = br#"{"usage":{"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        let (cached, written, source) = observe(&OpenAiAdapter, body);
        assert_eq!(cached, 1455);
        assert_eq!(written, 0);
        assert_eq!(source, SourceLabel::ProviderReported);
    }

    #[test]
    fn observer_records_ttft_at_first_nonempty_chunk() {
        let mut o = StreamObserver::new();
        assert!(o.ttft_ms.is_none());
        o.on_chunk(b"");
        assert!(o.ttft_ms.is_none(), "empty chunk is not first byte");
        o.on_chunk(b"data: {}\n\n");
        assert!(o.ttft_ms.is_some(), "first nonempty chunk sets TTFT");
    }

    #[test]
    fn observer_finalizes_cached_and_billed_from_the_tail() {
        let plan = RequestPlan {
            session_id: 7,
            turn: 2,
            resent_history_tokens: 1810,
            broke_prefix: false,
            prefix_hashes: Vec::new(),
        };
        let mut o = StreamObserver::new();
        o.on_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n");
        o.on_chunk(
            b"data: {\"usage\":{\"prompt_tokens\":2140,\"prompt_tokens_details\":{\"cached_tokens\":1455}}}\n\n",
        );
        let r = o.finalize(
            &plan,
            &OpenAiAdapter,
            "gpt-4o",
            &Rates::builtin(),
            true,
            None,
        );
        assert_eq!(r.cached_tokens, 1455);
        assert_eq!(r.billed_input_tokens, 2140);
        assert_eq!(r.resent_history_tokens, 1810);
        assert_eq!(r.turn, 2);
        assert!(r.cost_usd.is_some(), "known model carries cost");
    }

    #[test]
    fn tail_is_bounded() {
        let mut o = StreamObserver::new();
        for _ in 0..2000 {
            o.on_chunk(&[b'x'; 100]);
        }
        assert!(o.tail.len() <= o.cap, "tail must not grow unbounded");
    }

    #[test]
    fn plan_tracks_turns_and_denominator_across_a_session() {
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        let conv1 = vec![msg("system", "You are helpful."), msg("user", "Hi")];
        let p0 = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &conv1);
        assert_eq!(p0.turn, 0, "first request is cold");
        assert_eq!(p0.resent_history_tokens, 0, "turn 0 has no history");

        // Simulate the finalized turn 0 so the next request sees turn 1.
        let r = build_record(
            &p0,
            Observation {
                ttft_ms: Some(100.0),
                billed_input_tokens: 10,
                ..Default::default()
            },
            "gpt-4o",
            &Rates::builtin(),
            SourceLabel::ProviderReported,
            true,
        );
        store.0.lock().unwrap().append(r);

        let conv2 = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "More"),
        ];
        let p1 = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &conv2);
        assert_eq!(p1.session_id, p0.session_id, "same session");
        assert_eq!(p1.turn, 1);
        assert!(
            p1.resent_history_tokens > 0,
            "history excludes this turn's new content but includes the prefix"
        );
    }

    #[test]
    fn anthropic_write_split_lands_in_the_record() {
        use crate::adapters::anthropic::AnthropicAdapter;
        let plan = RequestPlan {
            session_id: 1,
            turn: 1,
            resent_history_tokens: 2000,
            broke_prefix: false,
            prefix_hashes: Vec::new(),
        };
        let mut o = StreamObserver::new();
        o.on_chunk(
            b"data: {\"usage\":{\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":300,\"input_tokens\":2100}}\n\n",
        );
        let r = o.finalize(
            &plan,
            &AnthropicAdapter,
            "claude-3-5-sonnet",
            &Rates::builtin(),
            true,
            None,
        );
        assert_eq!(r.cached_tokens, 900);
        assert_eq!(r.cache_written_tokens, 300, "creation split recorded");
        assert!(r.cost_saved_usd.is_some(), "anthropic rates known");
    }

    #[test]
    fn denominator_grows_with_each_turn() {
        // Regression: the re-sent history must grow as the conversation does
        // (system + prior messages), not stay pinned at the first request's
        // span. Turn t's history is every message before the final one.
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        let mut histories = Vec::new();
        let mut convo: Vec<Message> = vec![msg("system", "You are a helpful assistant.")];
        for turn in 0..4 {
            convo.push(msg("user", &format!("Question number {turn} please")));
            // resolve/append so turn indexing advances
            let p = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &convo);
            let r = build_record(
                &p,
                Observation {
                    billed_input_tokens: 10,
                    ..Default::default()
                },
                "gpt-4o",
                &Rates::builtin(),
                SourceLabel::ProviderReported,
                true,
            );
            histories.push(p.resent_history_tokens);
            store.0.lock().unwrap().append(r);
            convo.push(msg("assistant", &format!("Answer number {turn} here")));
        }
        assert_eq!(histories[0], 0, "turn 0 is cold");
        // Each subsequent turn's history strictly grows.
        for w in histories.windows(2) {
            if w[0] != 0 {
                assert!(w[1] > w[0], "history must grow: {:?}", histories);
            }
        }
    }

    #[test]
    fn plan_flags_a_broken_prefix() {
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        // Seed a session with a two-message prefix.
        let seed = vec![msg("system", "You are helpful."), msg("user", "First")];
        let _ = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &seed);
        // A request that shares only the system message then diverges.
        let broken = vec![msg("system", "You are helpful."), msg("user", "Different")];
        let p = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &broken);
        assert!(p.broke_prefix, "a divergent prefix is flagged on the plan");
        assert_eq!(
            p.session_id, 1,
            "the break stays in the tracked session, not a new one"
        );
    }

    #[test]
    fn messages_from_flattens_string_and_parts() {
        let body = br#"{"messages":[
            {"role":"system","content":"sys"},
            {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}
        ]}"#;
        let doc: serde_json::Value = serde_json::from_slice(body).unwrap();
        let msgs = messages_from_doc(&doc).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "sys");
        assert_eq!(msgs[1].text, "ab");
    }
}
