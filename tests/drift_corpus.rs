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
fn node_int_vs_float_number_text_is_flagged() {
    // JSON.stringify — compact, insertion order.
    let js = "{\"city\":\"Paris\",\"unit\":\"celsius\",\"count\":\"9007199254740993\",\"ratio\":1}";
    // note: JS serializes 2^53+1 inaccurately as a Number, so frameworks
    // stringify big ints — here a STRING field. The int-vs-float `ratio`
    // (1 vs 1.0) is a formatting difference in a NUMBER field.
    let canonical = "{\"city\": \"Paris\", \"unit\": \"celsius\", \"count\": \"9007199254740993\", \"ratio\": 1.0}";
    let c = classify_args(js, canonical);
    // ratio 1 (u64) vs 1.0 (f64): different number texts, and the ladder
    // treats number leaves exactly — flagged, never rewritten. The string
    // fields are fine; the break comes from the numeric formatting.
    assert_eq!(
        c.semantic_break,
        Some(0),
        "int-vs-float number formatting (1 vs 1.0) is flagged, not repaired"
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
fn exponent_forms_of_equal_values_are_flagged() {
    // 1e2 and 100 parse to numerically equal but textually different
    // numbers; the ladder compares conservatively (flag, never rewrite).
    let a = "{\"limit\": 1e2}";
    let b = "{\"limit\": 100}";
    let c = classify_args(a, b);
    assert_eq!(c.semantic_break, Some(0));
}
