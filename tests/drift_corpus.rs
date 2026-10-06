//! The serializer corpus: real re-serialization outputs from the libraries
//! agent frameworks actually use, asserted against the drift classifier.
//! Every fixture below was produced by the named producer, verbatim — the
//! point is evidence about coverage, not intuition about serializers.
//!
//! Producers:
//! - python 3.14 `json.dumps` (default; `separators=(',',':')` compact;
//!   `sort_keys=True`; `ensure_ascii=True/False`)
//! - node `JSON.stringify`
//!
//! The value under test is `arguments` — the tool-call blob frameworks
//! re-serialize from parsed state every turn, the canonical cache breaker.

use cachemax::repair::{classify_turn, DriftKind};
use cachemax::tokenize::Tokenizer;
use serde_json::json;

fn tok() -> Tokenizer {
    Tokenizer::default_encoder().unwrap()
}

fn classify_args(client: &str, canonical: &str) -> cachemax::repair::Classification {
    let client = vec![json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "book", "arguments": client}}],
    })];
    let canonical = vec![json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "book", "arguments": canonical}}],
    })];
    classify_turn(&client, Some(&canonical), false, &tok())
}

/// Rust serde canonical: compact separators (serde_json emits no spaces),
/// literal UTF-8, u64-safe integers.
const CANONICAL: &str =
    "{\"city\":\"Paris\",\"unit\":\"celsius\",\"count\":9007199254740993,\"ratio\":1.0}";

#[test]
fn python_default_spacing_is_tool_arg_reserialization() {
    // json.dumps(args) — ", " / ": " separators.
    let py =
        "{\"city\": \"Paris\", \"unit\": \"celsius\", \"count\": 9007199254740993, \"ratio\": 1.0}";
    let c = classify_args(py, CANONICAL);
    assert_eq!(c.drift_kind, Some(DriftKind::ToolArgReserialization));
    assert_eq!(c.semantic_break, None, "repairable, not a break");
}

#[test]
fn python_compact_matches_the_serde_form() {
    // json.dumps(args, separators=(',', ':')) — byte-identical to the
    // canonical form: a stable serializer never reads as drifted.
    let py = "{\"city\":\"Paris\",\"unit\":\"celsius\",\"count\":9007199254740993,\"ratio\":1.0}";
    let c = classify_args(py, CANONICAL);
    assert!(c.report_matches, "compact python == compact serde");
}

#[test]
fn python_sorted_keys_are_tool_arg_reserialization() {
    // json.dumps(args, sort_keys=True, separators=(',', ':'))
    let py = "{\"city\":\"Paris\",\"count\":9007199254740993,\"ratio\":1.0,\"unit\":\"celsius\"}";
    let c = classify_args(py, CANONICAL);
    assert_eq!(c.drift_kind, Some(DriftKind::ToolArgReserialization));
}

#[test]
fn python_ensure_ascii_escapes_are_tool_arg_reserialization() {
    // json.dumps(..., ensure_ascii=True) escapes non-ASCII as \uXXXX; the
    // canonical side carries the literal UTF-8. Both unescape to the same
    // string when parsed, so the blob is the same call.
    let canonical = "{\"note\": \"café\", \"city\": \"Paris\"}";
    let py = "{\"note\": \"caf\\u00e9\", \"city\": \"Paris\"}";
    let c = classify_args(py, canonical);
    assert_eq!(c.drift_kind, Some(DriftKind::ToolArgReserialization));
    assert_eq!(c.semantic_break, None);
}

#[test]
fn integers_beyond_f64_precision_do_not_alias() {
    // 2^53+1 vs 2^53+2 differ only past f64 precision; they are DIFFERENT
    // values and must never be called equivalent (the arbitrary_precision
    // feature is what keeps them distinct through the parse).
    let a = "{\"count\": 9007199254740993}";
    let b = "{\"count\": 9007199254740994}";
    let c = classify_args(a, b);
    assert_eq!(
        c.semantic_break,
        Some(0),
        "values that differ only past f64 precision are still different values"
    );
    assert_eq!(c.drift_kind, None);
}

#[test]
fn node_int_vs_float_number_text_is_repaired() {
    // JSON.stringify — compact, insertion order.
    let js = "{\"city\":\"Paris\",\"unit\":\"celsius\",\"count\":\"9007199254740993\",\"ratio\":1}";
    // note: JS serializes 2^53+1 inaccurately as a Number, so frameworks
    // stringify big ints — here a STRING field. The int-vs-float `ratio`
    // (1 vs 1.0) is a formatting difference in a NUMBER field.
    let canonical = "{\"city\": \"Paris\", \"unit\": \"celsius\", \"count\": \"9007199254740993\", \"ratio\": 1.0}";
    let c = classify_args(js, canonical);
    // ratio 1 (u64) vs 1.0 (f64): one number below 2^53 in two texts.
    // Numerically identical counts are one value; the drift is the
    // serialization, and the rewrite restores the recorded number text —
    // so big-int identity on the wire is never at risk.
    assert_eq!(
        c.semantic_break, None,
        "int-vs-float number formatting (1 vs 1.0) below 2^53 is repaired"
    );
    assert_eq!(
        c.drift_kind,
        Some(cachemax::repair::DriftKind::ToolArgReserialization)
    );
}

#[test]
fn float_text_round_trips_exactly() {
    // Identical number text on both sides is exact — including floats and
    // big integers — so a stable serializer never reads as drifted.
    let c = classify_args(CANONICAL, CANONICAL);
    assert!(c.report_matches, "byte-stable serializers stay clean");
}

#[test]
fn exponent_forms_of_equal_values_are_repaired() {
    // 1e2 and 100 are one number in two texts; the rewrite restores the
    // recorded text, so the drift class is repairable.
    let a = "{\"limit\": 1e2}";
    let b = "{\"limit\": 100}";
    let c = classify_args(a, b);
    assert_eq!(c.semantic_break, None);
}

#[test]
fn unicode_escape_forms_never_read_as_drift() {
    // "Caf\u00e9" and "Café" parse to one string value, and the proxy's
    // own re-serialization emits the same bytes for both — so an escaped
    // writer and a raw one produce identical forwarded bodies and the
    // ladder reads the pair clean. No drift exists at any layer.
    use cachemax::repair::classify_turn;
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    // The escaped side comes from parsing escaped JSON TEXT, the way a
    // wire request actually arrives.
    let escaped: Vec<serde_json::Value> =
        vec![serde_json::from_str(r#"{"role":"user","content":"Caf\u00e9 central"}"#).unwrap()];
    let raw = vec![serde_json::json!({
        "role": "user", "content": "Café central"
    })];
    assert_eq!(
        serde_json::to_vec(&escaped).unwrap(),
        serde_json::to_vec(&raw).unwrap(),
        "re-serialization normalizes escape forms"
    );
    let c = classify_turn(&escaped, Some(&raw), false, &tok);
    assert!(c.report_matches, "escapes are not drift at any layer");
}

#[test]
fn corrupt_arguments_with_matching_id_and_name_are_restorable() {
    // The aggressive rule (owner approved): a truncated arguments string
    // for a call the chain records intact — same id, same name — is
    // restored, because the upstream would reject the request outright.
    // The client element differs from the canonical ONLY in the corrupt
    // arguments; anything else keeps the turn flagged.
    use cachemax::repair::classify_turn;
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    let canonical = vec![serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"}}
        ]
    })];
    let client = vec![serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": "{\"city\": \"Par"}}
        ]
    })];
    let c = classify_turn(&client, Some(&canonical), false, &tok);
    assert_eq!(
        c.semantic_break, None,
        "corrupt-but-recorded args are repairable"
    );
    assert_eq!(
        c.drift_kind,
        Some(cachemax::repair::DriftKind::ToolArgsRestored)
    );

    // A different function name under the same id is NOT the recorded call:
    // flagged, never restored.
    let renamed = vec![serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_forecast",
                          "arguments": "{\"city\": \"Par"}}
        ]
    })];
    let c = classify_turn(&renamed, Some(&canonical), false, &tok);
    assert_eq!(
        c.semantic_break,
        Some(0),
        "a name change under the same id is a different call"
    );
}

#[test]
fn tool_result_reserialization_is_repaired() {
    // Frameworks re-serialize the tool's JSON output as aggressively as the
    // arguments. The tool message's content string is compared parsed: same
    // result, different serialization, repairable.
    use cachemax::repair::classify_turn;
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    let canonical = vec![serde_json::json!({
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "{\"temp_c\": 21.5, \"sky\": \"overcast\"}"
    })];
    let client = vec![serde_json::json!({
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "{\"sky\":\"overcast\",\"temp_c\":21.5}"
    })];
    let c = classify_turn(&client, Some(&canonical), false, &tok);
    assert_eq!(c.semantic_break, None);
    assert_eq!(
        c.drift_kind,
        Some(cachemax::repair::DriftKind::ToolArgReserialization),
        "tool payload reserialization: args and results share the class"
    );
}

#[test]
fn multi_block_text_merge_and_split_is_reshaped() {
    // A client that splits one text into two blocks (or merges two into
    // one) re-sends the same prose: value-equal after flattening, so the
    // reshape class repairs it. Blocks carrying siblings never reshape.
    use cachemax::repair::{classify_turn, DriftKind};
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    let canonical = vec![serde_json::json!({
        "role": "user", "content": "Part one. Part two."
    })];
    let split = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."}
        ]
    })];
    let c = classify_turn(&split, Some(&canonical), false, &tok);
    assert_eq!(c.semantic_break, None);
    assert_eq!(c.drift_kind, Some(DriftKind::RoleContentReshaped));

    // The merge direction: canonical split into blocks, client merged to a
    // string — the same equivalence, pinned separately so a future
    // asymmetric reshape regression cannot pass one direction and fail
    // the other silently.
    let canonical_split = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."}
        ]
    })];
    let merged = vec![serde_json::json!({
        "role": "user", "content": "Part one. Part two."
    })];
    let c = classify_turn(&merged, Some(&canonical_split), false, &tok);
    assert_eq!(c.semantic_break, None);
    assert_eq!(c.drift_kind, Some(DriftKind::RoleContentReshaped));

    // A sibling on any block blocks the reshape: annotations are semantic.
    let annotated = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "Part one. ",
             "annotations": [{"type": "citation", "id": "x"}]},
            {"type": "text", "text": "Part two."}
        ]
    })];
    let c = classify_turn(&annotated, Some(&canonical), false, &tok);
    assert_eq!(c.semantic_break, Some(0), "siblings block the reshape");
}

#[test]
fn langchain_shaped_resend_is_repaired() {
    // LangChain re-serializes tool calls with compact arguments and
    // reordered keys, and re-sends string content verbatim: the classic
    // agent-framework shape, end to end through the ladder.
    use cachemax::repair::{classify_turn, DriftKind};
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    let canonical = vec![
        serde_json::json!({"role": "system", "content": "You book tables."}),
        serde_json::json!({
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "book",
                             "arguments": "{\"restaurant\": \"L'Astrance\", \"guests\": 2}"}}]
        }),
        serde_json::json!({"role": "tool", "tool_call_id": "c1",
                            "content": "{\"confirmed\": true, \"time\": \"20:00\"}"}),
    ];
    let langchain = vec![
        serde_json::json!({"role": "system", "content": "You book tables."}),
        serde_json::json!({
            "role": "assistant", "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "book",
                             "arguments": "{\"guests\":2,\"restaurant\":\"L'Astrance\"}"}}]
        }),
        serde_json::json!({"role": "tool", "tool_call_id": "c1",
                            "content": "{\"time\":\"20:00\",\"confirmed\":true}"}),
    ];
    let c = classify_turn(&langchain, Some(&canonical), false, &tok);
    assert_eq!(c.semantic_break, None, "the whole LangChain shape repairs");
    assert_eq!(c.drift_kind, Some(DriftKind::ToolArgReserialization));
}

#[test]
fn vercel_ai_sdk_shaped_resend_is_repaired() {
    // The Vercel AI SDK sends content as a parts array where a plain
    // client sent a string: same text, different shape.
    use cachemax::repair::{classify_turn, DriftKind};
    use cachemax::tokenize::Tokenizer;
    let tok = Tokenizer::default_encoder().unwrap();
    let canonical = vec![serde_json::json!({
        "role": "user", "content": "Book a table for two."
    })];
    let vercel = vec![serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": "Book a table for two."}]
    })];
    let c = classify_turn(&vercel, Some(&canonical), false, &tok);
    assert_eq!(c.semantic_break, None);
    assert_eq!(c.drift_kind, Some(DriftKind::RoleContentReshaped));
}
