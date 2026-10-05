//! Anthropic breakpoint management (opt-in, `--manage-breakpoints`).
//!
//! Anthropic caches a prefix only up to a *breakpoint*: a
//! `cache_control: {"type": "ephemeral"}` marker on a content block. With
//! none, nothing is cached; with too many (provider limit: 4 per request),
//! the request is rejected. Good clients step breakpoints back as the
//! conversation grows — the incremental-breakpoint guidance — so each turn
//! both reads an older cached prefix and writes a newer one.
//!
//! When asked to, the proxy does exactly that, per request:
//!
//! - one breakpoint on the **last system block** (creating block form when
//!   the client sent the system prompt as a plain string), and
//! - one on the **last suitable block** (text or tool_result) of each of
//!   the **last three user messages**,
//!
//! which never exceeds the 4-block limit — minus any breakpoints the
//! request already carries on tool definitions, which are left alone and
//! counted toward it.
//!
//! Whose breakpoints are whose — the decline rule. The proxy only ever
//! writes bare `{"type": "ephemeral"}` markers, on a system last block or
//! a user message's last suitable block. A request whose hints are all of
//! that shape, and no more of them than the proxy placed last turn (the
//! ledger remembers the count it as-sent), reads as an echo of its own
//! work and is re-derived — which is what makes management idempotent and
//! lets stale echoed placements step forward. Anything else — a hint with
//! a `ttl`, a hint on an assistant message, one deeper in history than the
//! proxy would place, more than the proxy ever placed — is the client
//! managing its own breakpoints and passes through untouched, unless
//! `--force-breakpoints`. Placement always starts from a strip of
//! existing hints in `system`/`messages`, so a block never carries two
//! markers.
//!
//! Breakpoints are cache hints, not content: the repair match model
//! ignores `cache_control` entirely (see `crate::repair`), so a client
//! echoing a stale placement is drift-free, never "repaired" — and a
//! rewritten element never inherits the canonical side's hints (they are
//! re-derived here, or the client's own).

use serde_json::Value;

/// The provider's per-request breakpoint limit.
const LIMIT: usize = 4;

/// How many user-message blocks carry breakpoints (the system block takes
/// the fourth slot).
const USER_BLOCKS: usize = 3;

/// The marker the proxy places — the documented default hint.
fn ephemeral() -> Value {
    serde_json::json!({"type": "ephemeral"})
}

/// Whether an object is a content block carrying a cache hint: a
/// `cache_control` key whose value is an object, on an object that also
/// names a `type` — the shape Anthropic puts hints in. A user payload
/// with a same-named key is data and never counted or touched.
fn is_hint(block: &Value) -> bool {
    block.as_object().is_some_and(|m| {
        m.contains_key("type") && m.get("cache_control").is_some_and(Value::is_object)
    })
}

/// Whether a hint value is exactly what the proxy writes. Anything richer
/// (a `ttl`, an extension field) is a client's own policy.
fn is_ours(cc: &Value) -> bool {
    *cc == ephemeral()
}

/// What the client's request carries, for the decline decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Survey {
    /// Hints in `system`/`messages` — the growing-prefix placements this
    /// feature manages. Tool-definition hints are excluded: static, and
    /// never the proxy's to move.
    pub managing: usize,
    /// True when any of those hints is one the proxy's rule would never
    /// place: not a bare `{"type": "ephemeral"}`, on an assistant
    /// message, or on a system block other than the last.
    pub foreign: bool,
}

/// Read the request's hint placement: how many hints sit where this
/// feature manages them, and whether any has a shape the proxy would
/// never produce.
pub fn survey(doc: &Value) -> Survey {
    let mut s = Survey::default();
    // System: only the last block is ever ours.
    if let Some(Value::Array(blocks)) = doc.get("system") {
        for (i, block) in blocks.iter().enumerate() {
            if let Some(cc) = hint_value(block) {
                s.managing += 1;
                if i + 1 != blocks.len() || !is_ours(cc) {
                    s.foreign = true;
                }
            }
        }
    }
    // Messages: only user-role content is ever ours.
    if let Some(Value::Array(messages)) = doc.get("messages") {
        for msg in messages {
            let role = msg.get("role").and_then(Value::as_str);
            each_hint(msg, &mut |cc| {
                s.managing += 1;
                if role != Some("user") || !is_ours(cc) {
                    s.foreign = true;
                }
            });
        }
    }
    s
}

/// The object-valued `cache_control` of a typed block, if it is a hint.
fn hint_value(block: &Value) -> Option<&Value> {
    let map = block.as_object()?;
    if !map.contains_key("type") {
        return None;
    }
    map.get("cache_control").filter(|cc| cc.is_object())
}

/// Call `f` with the hint value of every typed content block in `v`.
fn each_hint<F: FnMut(&Value)>(v: &Value, f: &mut F) {
    match v {
        Value::Object(map) => {
            if let Some(cc) = hint_value(v) {
                f(cc);
            }
            for (_, val) in map {
                each_hint(val, f);
            }
        }
        Value::Array(items) => {
            for item in items {
                each_hint(item, f);
            }
        }
        _ => {}
    }
}

/// Count the cache hints in a request document: content-block hints
/// everywhere outside the `tools` subtree, plus each tool definition's
/// own hint (tool objects carry no `type`, but their hints count against
/// the provider limit; payload keys nested deeper are not hints).
pub fn count(doc: &Value) -> usize {
    fn walk(v: &Value) -> usize {
        match v {
            Value::Object(map) => {
                usize::from(is_hint(v)) + map.iter().map(|(_, val)| walk(val)).sum::<usize>()
            }
            Value::Array(items) => items.iter().map(walk).sum(),
            _ => 0,
        }
    }
    let mut n = 0;
    if let Value::Object(map) = doc {
        for (k, val) in map {
            if k != "tools" {
                n += walk(val);
            }
        }
    }
    if let Some(Value::Array(tools)) = doc.get("tools") {
        n += tools
            .iter()
            .filter(|t| t.get("cache_control").is_some_and(Value::is_object))
            .count();
    }
    n
}

/// The outcome of one management pass over a request document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Managed {
    /// The request carries client-authored breakpoints: left untouched,
    /// per `--manage-breakpoints` without `--force-breakpoints`.
    pub declined: bool,
    /// Breakpoints in the request as it now stands, everywhere they occur.
    pub total: usize,
}

/// Manage breakpoints on an Anthropic-dialect request document.
/// `ours_last_turn` is what the proxy placed on the previous as-sent
/// request of this session+model (the ledger's count).
pub fn manage(doc: &mut Value, force: bool, survey: &Survey, ours_last_turn: usize) -> Managed {
    if !force && (survey.foreign || survey.managing > ours_last_turn) {
        tracing::info!(
            target: "cachemax_breakpoints",
            hints = survey.managing,
            "client manages its own breakpoints; leaving them untouched"
        );
        return Managed {
            declined: true,
            total: count(doc),
        };
    }
    strip(doc);
    // Tool-definition hints the proxy never places still count against
    // the provider's limit.
    let budget = LIMIT.saturating_sub(count(doc));
    place(doc, budget);
    Managed {
        declined: false,
        total: count(doc),
    }
}

/// Strip cache hints from `system` and `messages` (never `tools`), in
/// place.
fn strip(doc: &mut Value) {
    for section in ["system", "messages"] {
        if let Some(section) = doc.get_mut(section) {
            strip_hints_in_place(section);
        }
    }
}

/// Whether any content block in `v` carries a cache hint. The repair
/// rewriter consults this before replacing a drifted element: rewriting
/// strips hint placement, and a client-managed placement must not be
/// silently erased.
pub(crate) fn has_cache_control(v: &Value) -> bool {
    fn carries_hint(v: &Value) -> bool {
        match v {
            Value::Object(map) => {
                map.get("cache_control").is_some_and(Value::is_object)
                    || map.values().any(carries_hint)
            }
            Value::Array(items) => items.iter().any(carries_hint),
            _ => false,
        }
    }
    carries_hint(v)
}

/// Remove every content-block cache hint in `v`, recursively, in place.
/// Shared with the repair rewriter, which must never let the canonical
/// side's hints ride into a rewritten element.
pub(crate) fn strip_hints_in_place(v: &mut Value) {
    match v {
        Value::Object(map) => {
            if map.contains_key("type") {
                if let Some(cc) = map.get("cache_control") {
                    if cc.is_object() {
                        map.remove("cache_control");
                    }
                }
            }
            for (_, val) in map.iter_mut() {
                strip_hints_in_place(val);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                strip_hints_in_place(item);
            }
        }
        _ => {}
    }
}

/// Place breakpoints per the incremental guidance, within `budget`.
fn place(doc: &mut Value, budget: usize) {
    let mut placed = 0;
    if budget == 0 {
        return;
    }
    // The last system block; a plain-string system becomes block form so it
    // can carry the hint at all. A non-object block (or a null) is not a
    // placeable block — skipped, never coerced.
    if let Some(system) = doc.get_mut("system") {
        let as_string = system.as_str().map(str::to_string);
        if let Some(s) = as_string {
            if !s.is_empty() {
                *system = serde_json::json!([
                    {"type": "text", "text": s, "cache_control": ephemeral()},
                ]);
                placed += 1;
            }
        } else if let Some(last) = system
            .as_array_mut()
            .and_then(|b| b.last_mut())
            .filter(|b| b.is_object())
        {
            // Key presence, not object-ness: a block whose `cache_control`
            // holds something else carries payload, never a hint.
            if last.get("cache_control").is_none() {
                last["cache_control"] = ephemeral();
                placed += 1;
            }
        }
    }
    // The last suitable block of each of the last three user messages.
    let mut user_placed = 0;
    if let Some(messages) = doc.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for msg in messages.iter_mut().rev() {
            if user_placed >= USER_BLOCKS || placed >= budget {
                break;
            }
            if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
                continue;
            }
            match msg.get_mut("content") {
                Some(Value::Array(blocks)) => {
                    for block in blocks.iter_mut().rev() {
                        let kind = block.get("type").and_then(|t| t.as_str());
                        // Key presence, not object-ness: a block whose
                        // `cache_control` holds something else carries
                        // payload, never a hint.
                        let carries = block.get("cache_control").is_some();
                        if matches!(kind, Some("text") | Some("tool_result")) && !carries {
                            block["cache_control"] = ephemeral();
                            placed += 1;
                            user_placed += 1;
                            break;
                        }
                    }
                }
                Some(content) => {
                    // A plain-string user message becomes block form, the
                    // same mechanical conversion the API supports.
                    if let Some(s) = content.as_str().map(str::to_string) {
                        if !s.is_empty() {
                            *content = serde_json::json!([
                                {"type": "text", "text": s, "cache_control": ephemeral()},
                            ]);
                            placed += 1;
                            user_placed += 1;
                        }
                    }
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": [{"type": "text", "text": text}]})
    }

    fn assistant(text: &str) -> Value {
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]})
    }

    /// The message indices whose content carries a cache hint, in order.
    fn hinted_messages(doc: &Value) -> Vec<usize> {
        doc["messages"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m["content"]
                    .as_array()
                    .is_some_and(|blocks| blocks.iter().any(is_hint))
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Manage a fresh document (nothing ours last turn).
    fn manage_fresh(doc: &mut Value, force: bool) -> Managed {
        let s = survey(doc);
        manage(doc, force, &s, 0)
    }

    #[test]
    fn places_system_plus_last_three_user_blocks() {
        let mut doc = json!({
            "model": "claude-3",
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                user("q1"), assistant("a1"),
                user("q2"), assistant("a2"),
                user("q3"), assistant("a3"),
                user("q4"), assistant("a4"),
                user("q5"),
            ],
        });
        let m = manage_fresh(&mut doc, false);
        assert!(!m.declined);
        assert_eq!(m.total, 4, "the provider limit");
        assert!(is_hint(&doc["system"][0]));
        // User messages sit at indices 0, 2, 4, 6, 8; the last three are
        // 4, 6, 8 — the older ones (0, 2) carry no hint.
        assert_eq!(hinted_messages(&doc), vec![4, 6, 8]);
    }

    #[test]
    fn string_system_and_string_user_become_block_form() {
        let mut doc = json!({
            "system": "be brief",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let m = manage_fresh(&mut doc, false);
        assert_eq!(m.total, 2);
        assert_eq!(
            doc["system"],
            json!([{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}])
        );
        assert_eq!(
            doc["messages"][0]["content"],
            json!([{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}])
        );
    }

    #[test]
    fn a_non_object_system_block_is_skipped_not_panicked() {
        // A nonstandard system array element must neither panic (IndexMut
        // on a non-object) nor be coerced into a typeless block; the
        // message side still gets its placements.
        let mut doc = json!({
            "system": ["be brief"],
            "messages": [user("q1")],
        });
        let m = manage_fresh(&mut doc, false);
        assert_eq!(doc["system"], json!(["be brief"]), "left as it came");
        assert_eq!(m.total, 1, "only the user block");
    }

    #[test]
    fn tool_result_blocks_carry_hints_too() {
        let mut doc = json!({
            "messages": [
                user("q1"),
                json!({"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                ]}),
                json!({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "20C"},
                ]}),
            ],
        });
        let m = manage_fresh(&mut doc, false);
        assert_eq!(m.total, 2, "q1 and the tool result");
        assert!(is_hint(&doc["messages"][2]["content"][0]));
        assert!(
            !is_hint(&doc["messages"][1]["content"][0]),
            "assistant tool_use blocks are never candidates"
        );
    }

    #[test]
    fn tool_definition_hints_are_budget_not_decline() {
        // Tool-definition hints carry no `type` of their own but count
        // against the provider limit. They are not "management" — a
        // client hinting its tools from turn 0 still gets its message
        // breakpoints managed within the remaining budget.
        let tools = json!([
            {"name": "f", "description": "d", "input_schema": {},
             "cache_control": {"type": "ephemeral"}},
            {"name": "g", "description": "d", "input_schema": {},
             "cache_control": {"type": "ephemeral"}},
        ]);
        let mut doc = json!({
            "tools": tools.clone(),
            "messages": [user("q1"), assistant("a1"), user("q2")],
        });
        let m = manage_fresh(&mut doc, false);
        assert!(!m.declined, "tool hints alone never decline");
        // 2 tool hints leave budget 2: both user blocks, exactly at the
        // provider limit.
        assert_eq!(m.total, 4, "tools kept + budget-respecting placements");
        assert_eq!(hinted_messages(&doc), vec![0, 2]);
        assert_eq!(doc["tools"], tools, "tool hints are never touched");
    }

    #[test]
    fn nested_tool_payload_keys_are_not_hints() {
        // A `cache_control` key buried in a tool's schema is payload: it
        // counts neither toward the limit nor the survey.
        let mut doc = json!({
            "tools": [{
                "name": "f", "description": "d",
                "input_schema": {"cache_control": {"doc": "payload"}},
            }],
            "messages": [user("q1"), assistant("a1"), user("q2"), assistant("a2"), user("q3")],
        });
        assert_eq!(count(&doc), 0, "no real hints anywhere yet");
        let s = survey(&doc);
        assert_eq!((s.managing, s.foreign), (0, false));
        let m = manage_fresh(&mut doc, false);
        assert!(!m.declined);
        assert_eq!(m.total, 3, "full placement: the payload key cost nothing");
    }

    #[test]
    fn management_is_idempotent() {
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [user("q1"), assistant("a1"), user("q2")],
        });
        let first = manage_fresh(&mut doc, false);
        let snapshot = doc.clone();
        // The client echoes exactly what went out: same hints, same shapes.
        let second = {
            let s = survey(&doc);
            manage(&mut doc, false, &s, first.total)
        };
        assert!(!second.declined, "echoed placements are ours, re-derived");
        assert_eq!(second.total, first.total);
        assert_eq!(doc, snapshot, "strip + re-place lands on the same bytes");
    }

    #[test]
    fn stepped_back_placements_are_re_derived() {
        // The conversation grew: the echoed breakpoint sits one user block
        // too early. Re-derivation keeps only the last three user blocks
        // hinted, so the stale one loses its hint.
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
                ]},
                assistant("a1"),
                user("q2"),
                assistant("a2"),
                user("q3"),
                assistant("a3"),
                user("q4"),
            ],
        });
        let m = {
            let s = survey(&doc);
            manage(&mut doc, false, &s, 1)
        };
        assert!(!m.declined, "a stale echo within the count is still ours");
        assert_eq!(
            hinted_messages(&doc),
            vec![2, 4, 6],
            "q1's stale hint stepped to the last three user blocks"
        );
    }

    #[test]
    fn assistant_block_hints_are_client_managed() {
        // The proxy never hints assistant content; a client that does is
        // managing, and passes through untouched.
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                user("q1"),
                {"role": "assistant", "content": [
                    {"type": "text", "text": "a1", "cache_control": {"type": "ephemeral"}},
                ]},
            ],
        });
        let snapshot = doc.clone();
        let m = manage_fresh(&mut doc, false);
        assert!(m.declined);
        assert_eq!(doc, snapshot, "not one byte touched");
    }

    #[test]
    fn ttl_hints_are_client_managed() {
        // A hint richer than the bare marker (a ttl) is the client's own
        // retention policy, never something to strip or replace.
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ]},
            ],
        });
        let snapshot = doc.clone();
        let m = manage_fresh(&mut doc, false);
        assert!(m.declined);
        assert_eq!(doc, snapshot, "the client's ttl hint survives verbatim");
    }

    #[test]
    fn non_last_system_block_hints_are_client_managed() {
        let mut doc = json!({
            "system": [
                {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "last"},
            ],
            "messages": [user("q1")],
        });
        let snapshot = doc.clone();
        let m = manage_fresh(&mut doc, false);
        assert!(m.declined, "only the LAST system block is ever ours");
        assert_eq!(doc, snapshot);
    }

    #[test]
    fn more_hints_than_we_ever_placed_is_client_managed() {
        // Shapes are ours, count is beyond anything the proxy placed:
        // the client is stepping its own breakpoints.
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
                ]},
                assistant("a1"),
                {"role": "user", "content": [
                    {"type": "text", "text": "q2", "cache_control": {"type": "ephemeral"}},
                ]},
                assistant("a2"),
                {"role": "user", "content": [
                    {"type": "text", "text": "q3", "cache_control": {"type": "ephemeral"}},
                ]},
                assistant("a3"),
                {"role": "user", "content": [
                    {"type": "text", "text": "q4", "cache_control": {"type": "ephemeral"}},
                ]},
            ],
        });
        let snapshot = doc.clone();
        let m = {
            let s = survey(&doc);
            manage(&mut doc, false, &s, 4)
        };
        assert!(m.declined, "five bare hints where four is our maximum");
        assert_eq!(doc, snapshot);
    }

    #[test]
    fn force_re_derives_over_client_placements() {
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ]},
                user("q2"),
            ],
        });
        let m = manage_fresh(&mut doc, true);
        assert!(!m.declined);
        assert_eq!(m.total, 3, "system + both user blocks, re-derived");
        assert_eq!(hinted_messages(&doc), vec![0, 1]);
        assert_eq!(
            doc["messages"][0]["content"][0]["cache_control"],
            ephemeral(),
            "the client's ttl hint is replaced by the bare marker"
        );
    }

    #[test]
    fn a_same_named_key_without_block_shape_is_payload() {
        // `cache_control` on an object with no `type` sibling — or with a
        // non-object value — is user payload: never counted, never
        // stripped, never placed upon.
        let mut doc = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1", "cache_control": "not-a-hint"},
                ]},
                user("q2"),
            ],
        });
        assert_eq!(count(&doc), 0);
        let m = manage_fresh(&mut doc, false);
        assert!(!m.declined);
        assert_eq!(
            doc["messages"][0]["content"][0]["cache_control"], "not-a-hint",
            "payload keys are never rewritten"
        );
        assert_eq!(
            m.total, 1,
            "only the placed hint on q2 counts; the payload key does not"
        );
    }

    #[test]
    fn empty_string_content_stays_empty() {
        let mut doc = json!({
            "system": "",
            "messages": [{"role": "user", "content": ""}],
        });
        let m = manage_fresh(&mut doc, false);
        assert_eq!(m.total, 0);
        assert_eq!(doc["system"], "", "nothing to place a hint on");
    }
}
