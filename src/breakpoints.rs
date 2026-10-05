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
//! Two rules keep it honest:
//!
//! - **Client management is respected.** A request that carries MORE
//!   breakpoints than the proxy placed last turn is client-managed: it
//!   passes through untouched, unless `--force-breakpoints`. The ledger
//!   remembers the count the proxy as-sent, so a compliant client echoing
//!   those bytes back never exceeds it — echoed placements are recognized
//!   as ours and simply re-derived, which is also what makes management
//!   idempotent.
//! - **Never duplicate.** Placement starts from a strip of existing hints
//!   in `system`/`messages`, so a block never carries two markers.
//!
//! Breakpoints are cache hints, not content: the repair match model
//! ignores `cache_control` entirely (see `crate::repair`), so a client
//! echoing a stale placement is drift-free, never "repaired".

use serde_json::Value;

/// The provider's per-request breakpoint limit.
const LIMIT: usize = 4;

/// How many user-message blocks carry breakpoints (the system block takes
/// the fourth slot).
const USER_BLOCKS: usize = 3;

/// The hint the provider understands; ephemeral is the documented default.
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

/// Count the cache hints in a request document, everywhere they occur:
/// on content blocks (an object naming a `type`), and on tool definitions
/// (inside `tools`, where the objects carry no `type` of their own).
pub fn count(doc: &Value) -> usize {
    fn walk(v: &Value, in_tools: bool) -> usize {
        match v {
            Value::Object(map) => {
                let hint = if in_tools {
                    map.get("cache_control").is_some_and(Value::is_object)
                } else {
                    is_hint(v)
                };
                usize::from(hint)
                    + map
                        .iter()
                        .map(|(k, val)| walk(val, in_tools || k == "tools"))
                        .sum::<usize>()
            }
            Value::Array(items) => items.iter().map(|i| walk(i, in_tools)).sum(),
            _ => 0,
        }
    }
    walk(doc, false)
}

/// The outcome of one management pass over a request document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Managed {
    /// The request carried client-authored breakpoints (more than the
    /// proxy ever placed on this conversation): left untouched. Only
    /// `--force-breakpoints` overrides.
    pub declined: bool,
    /// Breakpoints in the request as it now stands, everywhere they occur.
    pub total: usize,
}

/// Manage breakpoints on an Anthropic-dialect request document.
///
/// `client_placed` counts the hints the client's request carried;
/// `ours_last_turn` is what the proxy placed on the previous as-sent
/// request of this session+model (the ledger's count). Echoed-back proxy
/// placements never exceed that count, so anything above it is the
/// client's own management.
pub fn manage(
    doc: &mut Value,
    force: bool,
    client_placed: usize,
    ours_last_turn: usize,
) -> Managed {
    if client_placed > ours_last_turn && !force {
        tracing::info!(
            target: "cachemax_breakpoints",
            hints = client_placed,
            "client manages its own breakpoints; leaving them untouched"
        );
        return Managed {
            declined: true,
            total: client_placed,
        };
    }
    strip(doc);
    // Hints the proxy never places (tool definitions) still count against
    // the provider's limit.
    let budget = LIMIT.saturating_sub(count(doc));
    place(doc, budget);
    Managed {
        declined: false,
        total: count(doc),
    }
}

/// Strip cache hints from `system` and `messages` (never `tools`), in
/// place. Returns how many were removed.
fn strip(doc: &mut Value) -> usize {
    fn walk(v: &mut Value) -> usize {
        match v {
            Value::Object(map) => {
                let mut removed = 0;
                // `is_hint`, read through the map (the object shape it checks).
                if map.contains_key("type")
                    && map.get("cache_control").is_some_and(Value::is_object)
                {
                    map.remove("cache_control");
                    removed += 1;
                }
                for (_, val) in map.iter_mut() {
                    removed += walk(val);
                }
                removed
            }
            Value::Array(items) => items.iter_mut().map(walk).sum(),
            _ => 0,
        }
    }
    let mut removed = 0;
    for section in ["system", "messages"] {
        if let Some(section) = doc.get_mut(section) {
            removed += walk(section);
        }
    }
    removed
}

/// Place breakpoints per the incremental guidance, within `budget`.
/// Returns how many were placed.
fn place(doc: &mut Value, budget: usize) -> usize {
    let mut placed = 0;
    if budget == 0 {
        return 0;
    }
    // The last system block; a plain-string system becomes block form so it
    // can carry the hint at all.
    if let Some(system) = doc.get_mut("system") {
        let as_string = system.as_str().map(str::to_string);
        if let Some(s) = as_string {
            if !s.is_empty() {
                *system = serde_json::json!([
                    {"type": "text", "text": s, "cache_control": ephemeral()},
                ]);
                placed += 1;
            }
        } else if let Some(last) = system.as_array_mut().and_then(|b| b.last_mut()) {
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
    placed
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

    /// The message indices whose content carries a hint, in order.
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
        let m = manage(&mut doc, false, 0, 0);
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
        let m = manage(&mut doc, false, 0, 0);
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
        let m = manage(&mut doc, false, 0, 0);
        assert_eq!(m.total, 2, "q1 and the tool result");
        assert!(is_hint(&doc["messages"][2]["content"][0]));
        assert!(
            !is_hint(&doc["messages"][1]["content"][0]),
            "assistant tool_use blocks are never candidates"
        );
    }

    #[test]
    fn never_exceeds_the_limit_with_tool_hints_present() {
        // Tool-definition hints carry no `type` of their own but count
        // against the provider limit all the same. The ledger's count
        // includes them (the request went out carrying them), so an echo
        // is recognized; a fresh conversation is not.
        let mut doc = json!({
            "tools": [
                {"name": "f", "description": "d", "input_schema": {},
                 "cache_control": {"type": "ephemeral"}},
                {"name": "g", "description": "d", "input_schema": {},
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [user("q1"), assistant("a1"), user("q2")],
        });
        // Unknown hints on a fresh conversation: client-managed, declined.
        let m = manage(&mut doc, false, 2, 0);
        assert!(m.declined);
        // Echoed (ours last turn carried them): re-derived within budget.
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
        let m = manage(&mut doc, false, 2, 2);
        assert!(!m.declined);
        // 2 tool hints leave budget 2: both user blocks, exactly at the
        // provider limit.
        assert_eq!(m.total, 4, "tools kept + budget-respecting placements");
        assert_eq!(hinted_messages(&doc), vec![0, 2]);
        assert_eq!(doc["tools"], tools, "tool hints are never touched");
    }

    #[test]
    fn management_is_idempotent() {
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [user("q1"), assistant("a1"), user("q2")],
        });
        let first = manage(&mut doc, false, 0, 0);
        let snapshot = doc.clone();
        // The client echoes exactly what went out: same count, no extras.
        let second = manage(&mut doc, false, first.total, first.total);
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
        let m = manage(&mut doc, false, 1, 1);
        assert!(!m.declined);
        assert_eq!(
            hinted_messages(&doc),
            vec![2, 4, 6],
            "q1's stale hint stepped to the last three user blocks"
        );
    }

    #[test]
    fn client_managed_breakpoints_are_respected() {
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "extra", "cache_control": {"type": "ephemeral"}},
                ]},
                user("q2"),
            ],
        });
        let snapshot = doc.clone();
        // Two client-authored hints, none of them ours last turn.
        let m = manage(&mut doc, false, 2, 0);
        assert!(m.declined);
        assert_eq!(m.total, 2);
        assert_eq!(doc, snapshot, "not one byte touched");
    }

    #[test]
    fn force_re_derives_over_client_placements() {
        let mut doc = json!({
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "extra", "cache_control": {"type": "ephemeral"}},
                ]},
                user("q2"),
            ],
        });
        let m = manage(&mut doc, true, 2, 0);
        assert!(!m.declined);
        assert_eq!(m.total, 3, "system + both user blocks, re-derived");
        assert_eq!(hinted_messages(&doc), vec![0, 1]);
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
        let m = manage(&mut doc, false, 0, 0);
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
        let m = manage(&mut doc, false, 0, 0);
        assert_eq!(m.total, 0);
        assert_eq!(doc["system"], "", "nothing to place a hint on");
    }
}
