//! Drift detection: does the client's re-sent history extend the canonical
//! chain, and if not, what kind of drift is it?
//!
//! The match model mirrors what a provider's cache actually sees. Providers
//! parse the request JSON and tokenize the *content* — so JSON envelope key
//! order never matters (object equality is order-insensitive here), while
//! string leaves always matter byte-for-byte. Drift that matters is therefore
//! drift inside strings: a tool-call `arguments` blob re-serialized with
//! different key order or spacing, text re-wrapped with different whitespace,
//! content reshaped between string and parts-array form. `cache_control`
//! hints are the one exception: the provider does not tokenize them as
//! content, so their presence and placement are ignored on both sides.
//!
//! The equivalence ladder, per element pair:
//! 1. **Exact** — `serde_json::Value` equality (semantic JSON equality:
//!    object order-insensitive, strings exact). The provider sees the same
//!    tokens. Not drift.
//! 2. **ToolArgReserialization** — equal after parsing tool-call `arguments`
//!    strings as JSON on both sides. Same call, different serialization:
//!    repairable, because rewriting to the canonical serialization changes
//!    no semantics the model sees.
//! 3. **TextNormalization** — equal after collapsing whitespace runs in
//!    certified prose positions: a message's `content` string, and the
//!    `text`/`thinking` field of an exactly-shaped content part. Repairable.
//! 4. **RoleContentReshaped** — same role, same flattened text, different
//!    content shape (string ↔ parts array). Repairable.
//! 5. Otherwise — **semantic inequality**: the history means something
//!    different. Never rewritten. Repair rewrites the prefix up to (not
//!    including) that element and passes the rest through untouched
//!    (plan §3.3), flagged as drift it could not fix.
//!
//! Hard stops (never classify further, never rewrite): system prompt
//! changed, model switched, first turn of a session (nothing canonical to
//! extend). Turn 0 of the *ledger* is not turn 0 of the record store: an
//! incomplete attempt consumes a record turn but never enters the chain, so
//! a retry classifies against the pre-failure chain.
//!
//! Token counts appear only as *quantification* (`tokens_at_risk`): an
//! estimate of what the drift endangers, for the dry-run annotation and the
//! dashboard. No repair decision ever consults them.

use crate::tokenize::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Whether repair touches traffic at all, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairMode {
    /// Today's behavior: no classification, no annotation, no rewriting.
    Off,
    /// DEFAULT. Classify and annotate every turn; never mutate the request.
    DryRun,
    /// Rewrite drifted history to the canonical serialization. (Lands with
    /// the repair batch; the classification here is the gate it trusts.)
    On,
}

impl Default for RepairMode {
    /// `Off` — the *record's* neutral default meaning "not examined". The
    /// proxy's *operating* default is `DryRun` (set in `main`); this default
    /// exists only so a `Record` can be constructed without a report.
    fn default() -> Self {
        RepairMode::Off
    }
}

/// The flavor of a repairable divergence between the client's re-sent
/// history and the canonical chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftKind {
    /// Tool-call `arguments` re-serialized (key order/whitespace) with the
    /// same parsed JSON. The classic agent-framework cache breaker.
    ToolArgReserialization,
    /// Text content differing only in whitespace runs.
    TextNormalization,
    /// The client dropped leading history (old turns) and re-sent the rest;
    /// the canonical prefix restarts deeper in the conversation.
    TruncatedHistory,
    /// Content shape changed (string ↔ parts array) with identical text.
    RoleContentReshaped,
    /// More than one of the above in one turn.
    Mixed,
}

/// Why repair refused to touch this turn at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unrepairable {
    /// The system message changed. Everything downstream re-bases; rewriting
    /// history under a new system prompt would fabricate consent.
    SystemPromptChanged,
    /// The session's canonical chain is per model; this request switched.
    ModelSwitched,
    /// Nothing canonical to extend yet (first turn, or a purged ledger).
    FirstTurn,
}

/// The dry-run report for one turn: what drifted, how much is at risk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftReport {
    pub mode: RepairMode,
    /// True when the whole re-sent span is exactly the canonical chain.
    pub matches_canonical: bool,
    pub drift_kind: Option<DriftKind>,
    /// Re-sent history elements not byte-equal to their canonical
    /// counterparts (dropped leading elements count too).
    pub turns_affected: usize,
    /// Estimated tokens the drift endangers: the canonical elements that
    /// would be canonicalized, the truncated prefix, and — when semantic
    /// inequality stopped the match — the residual span that will not be
    /// served from cache. An estimate for annotation; never a repair input.
    pub tokens_at_risk: u64,
    pub unrepairable: Option<Unrepairable>,
}

impl DriftReport {
    /// The report for a turn repair did not examine (mode off).
    pub fn unexamined(mode: RepairMode) -> Self {
        Self {
            mode,
            matches_canonical: false,
            drift_kind: None,
            turns_affected: 0,
            tokens_at_risk: 0,
            unrepairable: None,
        }
    }
}

/// Full classification result. Everything the dry-run report shows, plus the
/// alignment facts the rewriting path consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    pub report_matches: bool,
    pub drift_kind: Option<DriftKind>,
    pub turns_affected: usize,
    pub tokens_at_risk: u64,
    pub unrepairable: Option<Unrepairable>,
    /// Where the client's history starts inside the canonical chain.
    /// `> 0` is leading truncation (the client dropped old turns).
    pub canonical_offset: usize,
    /// How many leading client elements are exact-or-equivalent under that
    /// alignment. The rewriting path may canonicalize exactly these.
    pub equivalent_run: usize,
    /// Index into the client's messages where semantic inequality stopped
    /// the match. Elements from here on pass through untouched.
    pub semantic_break: Option<usize>,
}

/// Classify one incoming request against the session's canonical chain.
///
/// `chain` is the canonical message chain for (session, model) — `None`
/// when the ledger holds nothing for the session. `session_has_other_model`
/// distinguishes a model switch (chain exists under another model) from a
/// genuine first turn.
pub fn classify_turn(
    client: &[Value],
    chain: Option<&[Value]>,
    session_has_other_model: bool,
    tokenizer: &Tokenizer,
) -> Classification {
    let Some(canonical) = chain else {
        return Classification {
            report_matches: false,
            drift_kind: None,
            turns_affected: 0,
            tokens_at_risk: 0,
            unrepairable: Some(if session_has_other_model {
                Unrepairable::ModelSwitched
            } else {
                Unrepairable::FirstTurn
            }),
            canonical_offset: 0,
            equivalent_run: 0,
            semantic_break: None,
        };
    };
    if canonical.is_empty() || client.is_empty() {
        return Classification {
            report_matches: false,
            drift_kind: None,
            turns_affected: 0,
            tokens_at_risk: 0,
            unrepairable: Some(Unrepairable::FirstTurn),
            canonical_offset: 0,
            equivalent_run: 0,
            semantic_break: None,
        };
    }

    // Hard stop: a changed system message re-bases everything. Compare the
    // leading elements only when both are system-role messages.
    let both_system = client[0].get("role").and_then(Value::as_str) == Some("system")
        && canonical[0].get("role").and_then(Value::as_str) == Some("system");
    if both_system && relation(&client[0], &canonical[0]) == Rel::Different {
        return Classification {
            report_matches: false,
            drift_kind: None,
            turns_affected: 1,
            tokens_at_risk: tokens_of(tokenizer, canonical),
            unrepairable: Some(Unrepairable::SystemPromptChanged),
            canonical_offset: 0,
            equivalent_run: 0,
            semantic_break: Some(0),
        };
    }

    // Find the best alignment: the offset d into the canonical chain that
    // the client's history extends. The common clean case (d = 0) is tried
    // first and usually wins immediately; the offset scan only runs when the
    // head-on comparison breaks early (truncation or drift). Among equal
    // runs, a break-free alignment beats one with a semantic break: with
    // repeated identical elements both readings fit, and the break-free one
    // (history extends, possibly truncated) is the reading repair can act on.
    let clean =
        |a: &Aligned| a.run == canonical.len().min(client.len()) && a.semantic_break.is_none();
    let mut best = align_at(client, canonical, 0);
    if !clean(&best) {
        for d in 1..canonical.len() {
            let aligned = align_at(client, canonical, d);
            let better = aligned.run > best.run
                || (aligned.run == best.run
                    && best.semantic_break.is_some()
                    && aligned.semantic_break.is_none());
            if better {
                best = aligned;
            }
            if clean(&best) {
                break;
            }
        }
    }

    // Assemble the report from the winning alignment.
    let a = best;
    let compared = canonical.len().min(client.len());
    let mut kinds: Vec<DriftKind> = Vec::new();
    let mut nonexact = 0usize;
    let mut tokens_at_risk = tokens_of(tokenizer, &canonical[..a.offset]);
    for i in 0..a.run {
        match relation(&client[i], &canonical[a.offset + i]) {
            Rel::Exact => {}
            Rel::Equivalent(k) => {
                nonexact += 1;
                tokens_at_risk +=
                    tokens_of(tokenizer, std::slice::from_ref(&canonical[a.offset + i]));
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
            Rel::Different => unreachable!("run stops before Different"),
        }
    }
    if a.offset > 0 && !kinds.contains(&DriftKind::TruncatedHistory) {
        kinds.push(DriftKind::TruncatedHistory);
    }
    let drift_kind = match kinds.len() {
        0 => None,
        1 => Some(kinds[0]),
        _ => Some(DriftKind::Mixed),
    };

    // The residual span the semantic break leaves unmatchable: canonical
    // elements positionally claimed by client elements from the break on.
    let residual = match a.semantic_break {
        Some(k) => canonical
            .len()
            .saturating_sub(a.offset + k)
            .min(client.len() - k),
        None => 0,
    };
    let turns_affected = a.offset + nonexact + residual;
    if residual > 0 {
        let k = a.semantic_break.unwrap();
        tokens_at_risk += tokens_of(tokenizer, &canonical[a.offset + k..a.offset + k + residual]);
    }

    Classification {
        report_matches: a.offset == 0
            && nonexact == 0
            && a.semantic_break.is_none()
            && a.run == compared,
        drift_kind,
        turns_affected,
        tokens_at_risk,
        unrepairable: None,
        canonical_offset: a.offset,
        equivalent_run: a.run,
        semantic_break: a.semantic_break,
    }
}

/// Turn a classification into the record-facing report under `mode`.
pub fn report(classification: &Classification, mode: RepairMode) -> DriftReport {
    DriftReport {
        mode,
        matches_canonical: classification.unrepairable.is_none() && classification.report_matches,
        drift_kind: classification.drift_kind,
        turns_affected: classification.turns_affected,
        tokens_at_risk: classification.tokens_at_risk,
        unrepairable: classification.unrepairable,
    }
}

/// The outcome of a rewrite in `on` mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rewrite {
    /// How many elements of the re-sent history were replaced with their
    /// canonical counterparts.
    pub elements_replaced: usize,
    /// The canonical serialization's token count of the replaced elements —
    /// the actual canonicalized amount (the dry-run's `tokens_at_risk`
    /// estimate made real). Reporting only.
    pub canonicalized_tokens: u64,
}

/// Rewrite the client's drifted history to the canonical serialization:
/// replace every *equivalent-but-not-exact* element under the winning
/// alignment with its canonical counterpart, and report the outcome.
/// Never touches exact matches, the client's new tail, or anything at/after
/// a semantic break (plan §3.3: rewrite the prefix `[0..k)` only). Never
/// invents content: a truncated prefix stays truncated — the dropped
/// elements are not re-added, only the kept span is canonicalized.
///
/// Returns `None` when there is nothing a rewrite may do (no drift, an
/// unrepairable hard stop, or no chain); the request must then be forwarded
/// untouched.
pub fn apply_canonical(
    client: &mut [Value],
    classification: &Classification,
    canonical: &[Value],
    tokenizer: &Tokenizer,
) -> Option<Rewrite> {
    if classification.unrepairable.is_some() || client.is_empty() || canonical.is_empty() {
        return None;
    }
    let limit = classification
        .semantic_break
        .unwrap_or(classification.equivalent_run)
        .min(client.len());
    let mut replaced = 0usize;
    let mut tokens = 0u64;
    for (i, element) in client.iter_mut().enumerate().take(limit) {
        let Some(canonical_element) = canonical.get(classification.canonical_offset + i) else {
            break;
        };
        // Recompute the relation: only equivalent-but-not-exact elements are
        // replaced, and only with the byte-stable canonical form — its
        // content, never its cache hints. Hints are placement policy:
        // breakpoint management re-derives them, and under a client-managed
        // request, inheriting the canonical side's would push the total
        // past the provider's limit.
        match relation(element, canonical_element) {
            Rel::Equivalent(_) => {
                tokens += tokens_of(tokenizer, std::slice::from_ref(canonical_element));
                *element = strip_cache_control(canonical_element).into_owned();
                replaced += 1;
            }
            Rel::Exact | Rel::Different => {}
        }
    }
    (replaced > 0).then_some(Rewrite {
        elements_replaced: replaced,
        canonicalized_tokens: tokens,
    })
}

/// One alignment attempt: client history starting at `offset` in canonical.
#[derive(Debug, Clone, Copy)]
struct Aligned {
    offset: usize,
    run: usize,
    semantic_break: Option<usize>,
}

fn align_at(client: &[Value], canonical: &[Value], offset: usize) -> Aligned {
    let mut run = 0usize;
    let mut semantic_break = None;
    while offset + run < canonical.len() && run < client.len() {
        match relation(&client[run], &canonical[offset + run]) {
            Rel::Different => {
                semantic_break = Some(run);
                break;
            }
            _ => run += 1,
        }
    }
    Aligned {
        offset,
        run,
        semantic_break,
    }
}

/// The element-relation ladder (see module docs). Ordered cheapest-first;
/// the first equality that holds names the relation. Cache hints
/// (`cache_control`) are stripped on both sides first: the provider does
/// not tokenize them as content, and breakpoint management (the proxy's or
/// the client's) moves them without changing meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rel {
    Exact,
    Equivalent(DriftKind),
    Different,
}

fn relation(a: &Value, b: &Value) -> Rel {
    let a = strip_cache_control(a);
    let b = strip_cache_control(b);
    let (a, b): (&Value, &Value) = (&a, &b);
    if a == b {
        return Rel::Exact;
    }
    if tool_args_normalized(a) == tool_args_normalized(b) {
        return Rel::Equivalent(DriftKind::ToolArgReserialization);
    }
    if text_normalized(a) == text_normalized(b) {
        return Rel::Equivalent(DriftKind::TextNormalization);
    }
    if reshaped(a) == reshaped(b) {
        return Rel::Equivalent(DriftKind::RoleContentReshaped);
    }
    Rel::Different
}

/// A copy of `v` with content-block cache hints removed — an object that
/// names a `type` and carries an object-valued `cache_control`. Borrowed
/// when there is nothing to strip, so hint-free requests pay no
/// allocation. The in-place walk lives in [`crate::breakpoints`]; one
/// predicate, one home.
fn strip_cache_control(v: &Value) -> std::borrow::Cow<'_, Value> {
    fn carries_hint(v: &Value) -> bool {
        match v {
            Value::Object(map) => is_hint_shaped(map) || map.values().any(carries_hint),
            Value::Array(items) => items.iter().any(carries_hint),
            _ => false,
        }
    }
    fn is_hint_shaped(map: &serde_json::Map<String, Value>) -> bool {
        map.contains_key("type") && map.get("cache_control").is_some_and(Value::is_object)
    }
    if !carries_hint(v) {
        return std::borrow::Cow::Borrowed(v);
    }
    let mut owned = v.clone();
    crate::breakpoints::strip_hints_in_place(&mut owned);
    std::borrow::Cow::Owned(owned)
}

/// Rewrite every tool-call `arguments` string as its parsed JSON value, so
/// two serializations of the same arguments compare equal. Covers the
/// current OpenAI shape (`tool_calls[].function.arguments`) and the legacy
/// `function_call.arguments` — both are objects carrying a sibling `name`,
/// which gates the tolerance: a string field merely *named* `arguments`
/// elsewhere in the tree is user payload, not a tool call, and must compare
/// exactly. Anthropic's `tool_use.input` is already an object on the wire
/// and needs no tolerance.
fn tool_args_normalized(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, val) in map {
                out.insert(
                    k.clone(),
                    if k == "arguments" && val.is_string() && map.contains_key("name") {
                        parsed_or_self(val)
                    } else {
                        tool_args_normalized(val)
                    },
                );
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(tool_args_normalized).collect()),
        _ => v.clone(),
    }
}

/// A string treated as its parsed JSON when it parses, else itself.
fn parsed_or_self(v: &Value) -> Value {
    let s = v.as_str().unwrap_or_default();
    serde_json::from_str(s).unwrap_or_else(|_| v.clone())
}

/// Collapse whitespace runs in the *certified prose positions* of a message
/// element: the `content` string of an object carrying a `role` sibling (a
/// message), and the `text`/`thinking` field of an exactly-shaped content
/// part inside a content array. Every other string — ids, names,
/// signatures, tool argument blobs, nested user payload in any field named
/// `text` or `content` — is data, not prose, and compares exactly.
fn text_normalized(v: &Value) -> Value {
    let Value::Object(map) = v else {
        return v.clone();
    };
    let mut out = Map::new();
    for (k, val) in map {
        let normalized = if k == "content" && map.contains_key("role") {
            match val {
                Value::String(_) => collapse_ws(val),
                // A content array's parts are certified one level down, by
                // their own exact shape.
                Value::Array(parts) => {
                    Value::Array(parts.iter().map(part_text_normalized).collect())
                }
                _ => val.clone(),
            }
        } else {
            val.clone()
        };
        out.insert(k.clone(), normalized);
    }
    Value::Object(out)
}

/// Normalize the prose field of a content part: `{type, text}` or
/// `{type, thinking}` exactly — two keys, matching type. A part carrying
/// any other key (`annotations`, `signature`, …) or any other type is data:
/// returned verbatim, so it compares exactly and is never rewritten away.
fn part_text_normalized(part: &Value) -> Value {
    let Some(map) = part.as_object() else {
        return part.clone();
    };
    let prose_key = match map.get("type").and_then(Value::as_str) {
        Some("text") if map.len() == 2 && map.contains_key("text") => "text",
        Some("thinking") if map.len() == 2 && map.contains_key("thinking") => "thinking",
        _ => return part.clone(),
    };
    let mut out = Map::new();
    for (k, val) in map {
        out.insert(
            k.clone(),
            if k.as_str() == prose_key {
                collapse_ws(val)
            } else {
                val.clone()
            },
        );
    }
    Value::Object(out)
}

/// Collapse interior runs of spaces/tabs/CRs to a single space. Newline
/// structure and line-edge whitespace are preserved verbatim: indentation
/// and line breaks are semantic text (code, YAML, markdown), and rewriting
/// them would change what the model reads, not just its serialization.
fn collapse_ws(v: &Value) -> Value {
    let Some(s) = v.as_str() else {
        return v.clone();
    };
    let mut out = String::with_capacity(s.len());
    let mut pending = String::new();
    let mut line_started = false;
    for ch in s.chars() {
        match ch {
            ' ' | '\t' | '\r' => {
                if line_started {
                    pending.push(ch);
                } else {
                    out.push(ch); // leading indentation stays verbatim
                }
            }
            '\n' => {
                out.push_str(&pending);
                pending.clear();
                out.push('\n');
                line_started = false;
            }
            c => {
                if !pending.is_empty() {
                    out.push(' ');
                    pending.clear();
                }
                line_started = true;
                out.push(c);
            }
        }
    }
    out.push_str(&pending); // trailing whitespace stays verbatim
    Value::String(out)
}

/// Flatten `content` to its text on both sides, so a string and an
/// equivalent parts-array compare equal. Text must then match exactly.
/// Content carrying non-text parts (images, audio, tool results) is kept
/// verbatim — it is semantic and must compare exactly, so a message that
/// gained, lost, or swapped an image is never "reshaped".
fn reshaped(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, val) in map {
                out.insert(
                    k.clone(),
                    if k == "content" {
                        match all_text_flattened(val) {
                            Some(text) => Value::String(text),
                            None => val.clone(),
                        }
                    } else {
                        reshaped(val)
                    },
                );
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(reshaped).collect()),
        _ => v.clone(),
    }
}

/// The flattened text of a content value when it is a string or an array of
/// text-only parts. `None` when any part is not a text part: non-text
/// content is semantic and must compare exactly.
fn all_text_flattened(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                let map = p.as_object()?;
                // A text part carrying any sibling key (`annotations`,
                // `cache_control`, `signature`, …) is not proven equal to a
                // bare string: the siblings are semantic and must compare
                // exactly, so they block the reshape.
                if map.len() != 2
                    || map.get("type").and_then(Value::as_str) != Some("text")
                    || !map.contains_key("text")
                {
                    return None;
                }
                out.push_str(map.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            Some(out)
        }
        _ => None,
    }
}

/// Serialized token count of the given canonical elements — the at-risk
/// estimate. Reporting only.
fn tokens_of(tokenizer: &Tokenizer, elements: &[Value]) -> u64 {
    elements
        .iter()
        .filter_map(|e| serde_json::to_string(e).ok())
        .map(|s| tokenizer.count(&s) as u64)
        .sum()
}

/// The A/B replay pair for one recorded chain: the same request in the
/// form a re-serializing client drifts into, and in the canonical form
/// repair forwards. The drift applies the classifier's own tolerances in
/// reverse — reordered tool-argument keys, collapsed interior whitespace
/// runs — so both bodies are semantically identical and differ only in
/// the bytes a provider's cache keys on. That delta is what the A/B
/// measurement prices.
///
/// The drift is gated exactly where the classifier's tolerance is gated:
/// only tool-call `arguments` strings with a sibling `name`, and only
/// `content` strings on messages with a `role`. An ungated transformation
/// would produce a pair the classifier itself reads as semantically
/// different — a fabricated measurement.
pub fn replay_pair(request: &crate::ledger::ReplayRequest) -> Value {
    fn drift_value(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut out = Map::new();
                for (k, val) in map {
                    let value = drift_value(val);
                    out.insert(
                        k.clone(),
                        if k == "arguments" && val.is_string() && map.contains_key("name") {
                            match serde_json::from_str::<Value>(val.as_str().unwrap_or_default()) {
                                // Re-serialized by a different serializer:
                                // keys sorted, spacing compacted — the
                                // same parsed JSON, still a wire string.
                                Ok(parsed) => {
                                    let resorted = compact(&parsed);
                                    Value::String(
                                        serde_json::to_string(&resorted).unwrap_or_default(),
                                    )
                                }
                                Err(_) => value,
                            }
                        } else if k == "content" && val.is_string() && map.contains_key("role") {
                            collapse_ws(val)
                        } else {
                            value
                        },
                    );
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(drift_value).collect()),
            _ => v.clone(),
        }
    }
    // Compact re-serialization: sort object keys — the canonical
    // drift this proxy exists to repair.
    fn compact(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let sorted: std::collections::BTreeMap<&String, &Value> = map.iter().collect();
                Value::Object(
                    sorted
                        .into_iter()
                        .map(|(k, val)| (k.clone(), compact(val)))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(compact).collect()),
            _ => v.clone(),
        }
    }

    let canonical = serde_json::json!({
        "model": request.model,
        "messages": request.messages,
    });
    let drifted = drift_value(&canonical);
    serde_json::json!({
        "session_id": request.session_id,
        "turn": request.turn,
        "model": request.model,
        "a_drifted": drifted,
        "b_canonical": canonical,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tok() -> Tokenizer {
        Tokenizer::default_encoder().unwrap()
    }

    fn classify(client: &[Value], canonical: &[Value]) -> Classification {
        classify_turn(client, Some(canonical), false, &tok())
    }

    fn msgs(pairs: &[(&str, &str)]) -> Vec<Value> {
        pairs
            .iter()
            .map(|(role, text)| json!({"role": role, "content": text}))
            .collect()
    }

    #[test]
    fn a_clean_continuation_matches_exactly() {
        let canonical = msgs(&[
            ("system", "Be terse."),
            ("user", "Hi"),
            ("assistant", "Hello"),
        ]);
        let mut client = canonical.clone();
        client.push(json!({"role": "user", "content": "More"}));
        let c = classify(&client, &canonical);
        assert!(c.report_matches, "clean extension of the chain");
        assert_eq!(c.drift_kind, None);
        assert_eq!(c.tokens_at_risk, 0);
        assert_eq!(c.unrepairable, None);
        assert_eq!(c.canonical_offset, 0);
        assert_eq!(c.equivalent_run, 3, "the run covers the whole chain");
        assert_eq!(c.semantic_break, None);
    }

    #[test]
    fn a_regenerate_request_with_no_new_tail_still_matches() {
        let canonical = msgs(&[("system", "sys"), ("user", "Hi"), ("assistant", "Hello")]);
        let c = classify(&canonical, &canonical);
        assert!(c.report_matches, "resent == chain is a clean prefix");
    }

    #[test]
    fn tool_args_key_reorder_is_tool_arg_reserialization() {
        let canonical = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c1", "type": "function", "function":
                {"name": "f", "arguments": "{\"a\": 1, \"b\": 2}"}}],
        })];
        let client = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c1", "type": "function", "function":
                {"name": "f", "arguments": "{\"b\":2,\"a\":1}"}}],
        })];
        let c = classify(&client, &canonical);
        assert!(!c.report_matches);
        assert_eq!(c.drift_kind, Some(DriftKind::ToolArgReserialization));
        assert_eq!(c.turns_affected, 1);
        assert!(c.tokens_at_risk > 0, "the drift is quantified");
        assert_eq!(c.semantic_break, None, "equivalence holds — repairable");
    }

    #[test]
    fn a_changed_argument_value_is_semantic_inequality() {
        let canonical = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function":
                {"name": "f", "arguments": "{\"a\": 1}"}}],
        })];
        let client = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function":
                {"name": "f", "arguments": "{\"a\": 2}"}}],
        })];
        let c = classify(&client, &canonical);
        assert!(!c.report_matches);
        assert_eq!(c.semantic_break, Some(0), "equivalence fails at element 0");
        assert_eq!(
            c.drift_kind, None,
            "semantic inequality carries no repairable kind"
        );
        assert_eq!(c.turns_affected, 1);
        assert!(c.tokens_at_risk > 0, "the unmatchable span is quantified");
    }

    #[test]
    fn whitespace_normalization_is_detected() {
        let canonical = msgs(&[("user", "Please   summarize  the   results.")]);
        let client = msgs(&[("user", "Please summarize the results.")]);
        let c = classify(&client, &canonical);
        assert!(!c.report_matches);
        assert_eq!(c.drift_kind, Some(DriftKind::TextNormalization));
    }

    #[test]
    fn newline_structure_is_semantic_not_normalization() {
        // Line breaks are meaning (code, lists); collapsing them would
        // rewrite what the model reads. Such drift is inequality: flagged,
        // passed through, never "repaired".
        let canonical = msgs(&[("user", "if x:\n    return 1\nelse:\n    return 2")]);
        let client = msgs(&[("user", "if x: return 1 else: return 2")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn indentation_is_preserved_not_collapsed() {
        // Leading line whitespace stays verbatim; interior runs collapse.
        let canonical = msgs(&[("user", "line one\n    indented")]);
        let client = msgs(&[("user", "line one\n  indented")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0), "different indentation differs");
        // Same indentation, different interior run: normalization.
        let client2 = msgs(&[("user", "line  one\n    indented")]);
        let c2 = classify(&client2, &canonical);
        assert_eq!(c2.drift_kind, Some(DriftKind::TextNormalization));
    }

    #[test]
    fn non_text_content_parts_are_never_reshaped_away() {
        // A message that gained an image is a different message; a message
        // with a different image is too. Neither may be rewritten to the
        // text-only canonical form.
        let canonical = msgs(&[("user", "What is in this image?")]);
        let with_image = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        })];
        let c = classify(&with_image, &canonical);
        assert_eq!(c.semantic_break, Some(0), "image content is inequality");

        // Two different images are not equivalent to each other either.
        let other_image = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/dog.png"}},
            ],
        })];
        let c2 = classify(&other_image, with_image.as_slice());
        assert_eq!(c2.semantic_break, Some(0));
    }

    #[test]
    fn a_string_field_named_arguments_is_not_tool_args() {
        // Tool-arg tolerance is gated on the function-call structure (a
        // sibling `name`), not the bare field name: a user-authored
        // `arguments` payload must compare exactly.
        let canonical = vec![json!({
            "role": "user",
            "content": "Deploy this",
            "arguments": "{\"region\": \"us-east-1\", \"replicas\": 3}",
        })];
        let client = vec![json!({
            "role": "user",
            "content": "Deploy this",
            "arguments": "{\"replicas\":3,\"region\":\"us-east-1\"}",
        })];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0), "user payload is not a tool call");
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn duplicated_elements_prefer_the_break_free_alignment() {
        // With repeated identical elements two readings fit; the break-free
        // one (truncation) must win over the semantic-break reading.
        let canonical = msgs(&[("user", "continue"), ("user", "continue")]);
        let client = msgs(&[("user", "continue"), ("user", "what next?")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.canonical_offset, 1, "the dropped duplicate is truncation");
        assert_eq!(c.semantic_break, None);
        assert_eq!(c.drift_kind, Some(DriftKind::TruncatedHistory));
    }

    #[test]
    fn content_reshaped_between_string_and_parts_is_detected() {
        let canonical = msgs(&[("user", "hello world")]);
        let client = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}],
        })];
        let c = classify(&client, &canonical);
        assert!(!c.report_matches);
        assert_eq!(c.drift_kind, Some(DriftKind::RoleContentReshaped));
    }

    #[test]
    fn leading_truncation_aligns_deeper_and_is_truncated_history() {
        let canonical = msgs(&[
            ("system", "sys"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
        ]);
        // Client drops the first two turns and adds a new tail.
        let mut client = canonical[4..].to_vec();
        client.push(json!({"role": "user", "content": "q3"}));
        let c = classify(&client, &canonical);
        assert!(!c.report_matches);
        assert_eq!(c.drift_kind, Some(DriftKind::TruncatedHistory));
        assert_eq!(c.canonical_offset, 4, "alignment starts deeper");
        assert_eq!(c.equivalent_run, 1, "the kept assistant element matches");
        assert_eq!(c.turns_affected, 4, "the four dropped elements");
        assert!(c.tokens_at_risk > 0);
    }

    #[test]
    fn truncation_plus_tool_args_is_mixed() {
        let canonical = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1",
                "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1, \"b\": 2}"}}]}),
        ];
        let client = vec![
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1",
                "type": "function", "function": {"name": "f", "arguments": "{\"b\":2,\"a\":1}"}}]}),
            json!({"role": "user", "content": "next"}),
        ];
        let c = classify(&client, &canonical);
        assert_eq!(c.drift_kind, Some(DriftKind::Mixed));
        assert_eq!(
            c.canonical_offset, 2,
            "the leading system+user were dropped"
        );
    }

    #[test]
    fn a_changed_system_prompt_is_a_hard_stop() {
        let canonical = msgs(&[("system", "Be terse."), ("user", "Hi")]);
        let client = msgs(&[("system", "Be verbose."), ("user", "Hi")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.unrepairable, Some(Unrepairable::SystemPromptChanged));
        assert!(!c.report_matches);
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn an_equivalent_system_prompt_change_is_not_a_hard_stop() {
        // Whitespace-only system drift is repairable, not a re-base.
        let canonical = msgs(&[("system", "Be   terse."), ("user", "Hi")]);
        let client = msgs(&[("system", "Be terse."), ("user", "Hi")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.unrepairable, None);
        assert_eq!(c.drift_kind, Some(DriftKind::TextNormalization));
    }

    #[test]
    fn no_chain_is_first_turn_or_model_switch() {
        let c = classify_turn(&msgs(&[("user", "hi")]), None, false, &tok());
        assert_eq!(c.unrepairable, Some(Unrepairable::FirstTurn));
        let c = classify_turn(&msgs(&[("user", "hi")]), None, true, &tok());
        assert_eq!(c.unrepairable, Some(Unrepairable::ModelSwitched));
    }

    #[test]
    fn envelope_key_order_alone_is_not_drift() {
        // Providers tokenize parsed content, not envelope bytes: a key
        // reorder anywhere in the message object is exact, not drift.
        let canonical = vec![json!({"role": "user", "content": "hi", "extra": 1})];
        let client = vec![json!({"extra": 1, "content": "hi", "role": "user"})];
        let c = classify(&client, &canonical);
        assert!(c.report_matches);
    }

    #[test]
    fn a_changed_role_is_semantic_inequality() {
        let canonical = msgs(&[("user", "hi")]);
        let client = msgs(&[("assistant", "hi")]);
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn drift_after_a_clean_prefix_is_located_not_global() {
        let canonical = msgs(&[
            ("system", "sys"),
            ("user", "q1"),
            ("assistant", "a1  spaced"),
            ("user", "q2"),
        ]);
        let client = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "a1 spaced"}),
            json!({"role": "user", "content": "q2"}),
        ];
        let c = classify(&client, &canonical);
        assert_eq!(c.canonical_offset, 0);
        assert_eq!(c.equivalent_run, 4);
        assert_eq!(c.turns_affected, 1, "only the drifted element counts");
        assert_eq!(c.drift_kind, Some(DriftKind::TextNormalization));
    }

    #[test]
    fn semantic_inequality_mid_history_stops_the_run() {
        let canonical = msgs(&[
            ("system", "sys"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
        ]);
        // Element 1 differs semantically; element 2+ would have matched.
        let client = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "DIFFERENT"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
        ];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(1));
        assert_eq!(c.equivalent_run, 1, "rewrite prefix is [0..1) only");
        // The residual (client[1..] positionally claiming canonical[1..4])
        // is counted at risk.
        assert_eq!(c.turns_affected, 3);
    }

    #[test]
    fn report_carries_mode_and_matches() {
        // Interior-run whitespace drift (trailing spaces are line-edge and
        // stay verbatim under the conservative normalization).
        let canonical = msgs(&[("user", "hi  there"), ("assistant", "yo")]);
        let client = msgs(&[("user", "hi there"), ("assistant", "yo")]);
        let c = classify(&client, &canonical);
        let r = report(&c, RepairMode::DryRun);
        assert_eq!(r.mode, RepairMode::DryRun);
        assert!(!r.matches_canonical);
        assert_eq!(r.drift_kind, Some(DriftKind::TextNormalization));
        assert_eq!(r.tokens_at_risk, c.tokens_at_risk);
    }

    #[test]
    fn nested_text_keys_are_payload_not_prose() {
        // Whitespace tolerance is certified for message content and
        // exactly-shaped text parts — not for any string named `text`
        // anywhere in the tree. Nested user payload is data and must
        // compare exactly, or a rewrite would silently reformat it.
        let canonical = vec![json!({
            "role": "user",
            "content": "Deploy",
            "spec": {"text": "keep  double  spaces"},
        })];
        let client = vec![json!({
            "role": "user",
            "content": "Deploy",
            "spec": {"text": "keep double spaces"},
        })];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn annotated_text_parts_are_not_reshaped_away() {
        // A text part carrying sibling keys is not a bare string: the
        // siblings (annotations, cache_control) are semantic and must
        // compare exactly, never be rewritten away by a reshape.
        let canonical = msgs(&[("user", "hello world")]);
        let client = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello world",
                         "annotations": [{"type": "url_citation"}]}],
        })];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn a_signed_thinking_part_is_data_not_prose() {
        // A thinking part carrying a signature sibling is not the certified
        // two-key prose part: whitespace inside its text compares exactly.
        let canonical = vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm  let me think", "signature": "sig1"},
        ]})];
        let client = vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm let me think", "signature": "sig1"},
        ]})];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
        assert_eq!(c.drift_kind, None);
    }

    #[test]
    fn exactly_shaped_text_parts_still_normalize() {
        // The certified shape keeps the tolerance: two-key text parts are
        // prose, so an interior whitespace run alone is repairable drift.
        let canonical = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "summarize   this"},
        ]})];
        let client = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "summarize this"},
        ]})];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, None);
        assert_eq!(c.drift_kind, Some(DriftKind::TextNormalization));
    }

    #[test]
    fn cache_control_presence_and_placement_are_not_content() {
        // Breakpoint hints move as conversations grow (Anthropic's
        // incremental guidance); the provider does not tokenize them as
        // content. A hint present on the canonical side and absent (or
        // moved) on the client's is an exact match — never drift, never
        // rewritten for, never at risk.
        let canonical = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "world"},
        ]})];
        let without = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]})];
        let c = classify(&without, &canonical);
        assert!(c.report_matches, "hint presence alone is not drift");
        assert_eq!(c.tokens_at_risk, 0);

        let moved = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world", "cache_control": {"type": "ephemeral"}},
        ]})];
        let c2 = classify(&moved, &canonical);
        assert!(c2.report_matches, "hint placement alone is not drift");
        assert_eq!(c2.drift_kind, None);

        // A string ↔ hinted-parts reshape is still detected as a reshape:
        // the hint must not block the ladder's sibling-gated rungs (the
        // three-key part would read as non-flattenable without the strip).
        let as_string = vec![json!({"role": "user", "content": "hello world"})];
        let hinted_parts = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hello world", "cache_control": {"type": "ephemeral"}},
        ]})];
        let c3 = classify(&hinted_parts, &as_string);
        assert_eq!(c3.semantic_break, None);
        assert_eq!(
            c3.drift_kind,
            Some(DriftKind::RoleContentReshaped),
            "the shape difference is still repairable drift"
        );
    }

    #[test]
    fn a_cache_control_key_outside_block_shape_is_payload() {
        // `cache_control` with a non-object value, or on an object with no
        // `type` sibling, is user payload: compared exactly like any other
        // string leaf, never stripped.
        let canonical = vec![json!({
            "role": "user", "content": "x", "cache_control": 1,
        })];
        let client = vec![json!({
            "role": "user", "content": "x", "cache_control": 2,
        })];
        let c = classify(&client, &canonical);
        assert_eq!(c.semantic_break, Some(0));
    }

    #[test]
    fn a_rewrite_never_inherits_the_canonical_side_hints() {
        // The chain's element carries the proxy's breakpoints; the drifted
        // re-send is rewritten to its content, hint-free. Hints are
        // placement policy — re-derived by breakpoint management or the
        // client's own — and a rewrite smuggling them in could push a
        // client-managed request past the provider's block limit.
        let canonical = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "Please  summarize",
             "cache_control": {"type": "ephemeral"}},
        ]})];
        let mut client = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "Please summarize"},
        ]})];
        let c = classify(&client, &canonical);
        let rw = apply_canonical(&mut client, &c, &canonical, &tok()).unwrap();
        assert_eq!(rw.elements_replaced, 1);
        assert_eq!(
            client[0]["content"][0]["text"], "Please  summarize",
            "the canonical content went in"
        );
        assert!(
            client[0]["content"][0].get("cache_control").is_none(),
            "the canonical side's hint did not"
        );
    }

    #[test]
    fn hint_ttl_differences_are_not_content() {
        // A ttl changes retention policy, not the cached prefix's content;
        // the match model ignores hints entirely, so this is an exact
        // match. (Under --manage-breakpoints, ttl hints read as
        // client-managed and pass through untouched — see
        // crate::breakpoints.)
        let canonical = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "q", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        ]})];
        let client = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "q", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]})];
        let c = classify(&client, &canonical);
        assert!(c.report_matches);
        assert_eq!(c.tokens_at_risk, 0);
    }
}
