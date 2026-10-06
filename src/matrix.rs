//! `cachemax drift-matrix`: measure, per drift class, what
//! semantically-identical-but-byte-different request bodies actually cost
//! against a live endpoint.
//!
//! Each class is a (canonical, drifted) body pair — the same conversation,
//! different serialization. The runner samples both forms interleaved (the
//! same discipline as `replay --execute`: a, b, a, b on fresh connections)
//! and reports cached-token medians and the delta. The verdict is plain:
//! a delta of zero means the endpoint absorbed the class (nothing for
//! repair to recover there); a positive delta is the cache that repair
//! would recover on that endpoint.
//!
//! Classes are data, not code paths: the bodies are built once here, and
//! adding a class is adding a fixture. The matrix never rewrites anything —
//! it measures the raw wire cost of each serialization.

use crate::replay::{execute_bodies, ExecuteConfig, FormSample};

/// The measured cost of one drift class against one endpoint.
#[derive(Debug, Clone)]
pub struct ClassResult {
    pub name: &'static str,
    pub a_drifted: FormSample,
    pub b_canonical: FormSample,
    /// Plain verdict: "absorbed" when the medians are equal, "costs N tk"
    /// when the canonical form recovers cache, "unmeasured" when a form had
    /// no readable reading.
    pub verdict: &'static str,
    pub delta: Option<u64>,
}

/// One drift class: the same conversation twice. The canonical body is the
/// serialization the provider already cached; the drifted body is what a
/// framework re-serializing the conversation would send instead.
#[derive(Debug, Clone)]
pub struct MatrixClass {
    pub name: &'static str,
    /// Which backend dialect the pair speaks: "openai" or "anthropic".
    pub dialect: &'static str,
    pub canonical: serde_json::Value,
    pub drifted: serde_json::Value,
}

/// A deterministic, chunky shared prefix — cache effects need enough tokens
/// to sit above router noise, and identical text everywhere except the
/// mutated span keeps the comparison clean.
const PROSE: &str = "You are the records assistant for the Kyoto facility audit. The lab protocol requires quoting the reference batch identifier in every reply, which is BATCH-7741-ALPHA-9, registered during the March audit window under supervision of the quality office. The protocol file lives in the east annex, revision fourteen, and any deviation from the quoted procedure must be reported within one business day to the same office, referencing the batch identifier and the audit window. Weather queries are answered from the rooftop station feed, which reports temperature, sky condition, and observation hours, and every answer names the station that produced the reading.";

fn base_openai_body(system: &str, model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Weather report for Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\", \"extended\": 1}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "{\"temp_c\": 21.5, \"sky\": \"overcast\", \"hours\": [{\"from\": 8, \"to\": 18}]}"},
        ],
    })
}

fn base_anthropic_body(system_blocks: serde_json::Value, model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "max_tokens": 1024,
        "tools": [{
            "name": "get_weather",
            "description": "Reads the rooftop station feed for a city and unit.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "extended": {"type": "integer"}
                }
            }
        }],
        "system": system_blocks,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather report for Paris?"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "get_weather",
                 "input": {"city": "Paris", "unit": "celsius", "extended": 1}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "{\"temp_c\": 21.5, \"sky\": \"overcast\", \"hours\": [{\"from\": 8, \"to\": 18}]}"}
            ]},
        ],
    })
}

/// The class fixtures for one backend dialect. Adding a class is adding a
/// fixture here; the runner and table pick it up automatically.
pub fn classes(backend: crate::replay::Backend, model: &str) -> Vec<MatrixClass> {
    match backend {
        crate::replay::Backend::OpenAi => {
            let canonical = base_openai_body(PROSE, model);
            vec![
                // Tool-argument keys reordered and compacted by a framework.
                MatrixClass {
                    name: "tool-arg-reorder",
                    dialect: "openai",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["messages"][2]["tool_calls"][0]["function"]["arguments"] = serde_json::json!(
                            "{\"unit\":\"celsius\",\"extended\":1,\"city\":\"Paris\"}"
                        );
                        d
                    },
                },
                // Interior whitespace collapsed in the system prompt.
                MatrixClass {
                    name: "whitespace",
                    dialect: "openai",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["messages"][0]["content"] =
                            serde_json::Value::String(PROSE.replace(". ", ".  "));
                        d
                    },
                },
                // Object key order inside every message.
                MatrixClass {
                    name: "key-order",
                    dialect: "openai",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        let msgs = d["messages"].as_array_mut().unwrap();
                        for m in msgs.iter_mut() {
                            let obj = m.as_object_mut().unwrap();
                            // Reverse the key insertion order: same value,
                            // different wire bytes under preserve_order.
                            let mut rotated: Vec<(String, serde_json::Value)> = Vec::new();
                            for k in obj.keys().cloned().collect::<Vec<_>>() {
                                let v = obj.shift_remove(&k).unwrap();
                                rotated.push((k, v));
                            }
                            for (k, v) in rotated.into_iter().rev() {
                                obj.insert(k, v);
                            }
                        }
                        d
                    },
                },
                // Number text: 1 re-serialized as 1.0 in tool arguments.
                MatrixClass {
                    name: "number-text",
                    dialect: "openai",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["messages"][2]["tool_calls"][0]["function"]["arguments"] = serde_json::json!(
                            "{\"city\": \"Paris\", \"unit\": \"celsius\", \"extended\": 1.0}"
                        );
                        d
                    },
                },
                // Tool definitions re-serialized (key order) — the
                // cache-root class on providers that cache the prefix.
                MatrixClass {
                    name: "tools-reserialization",
                    dialect: "openai",
                    canonical: {
                        let mut d = canonical.clone();
                        d["tools"] = serde_json::json!([
                            {"type": "function",
                             "function": {"name": "get_weather",
                                          "description": "Reads the rooftop station feed.",
                                          "parameters": {"type": "object",
                                                         "properties": {"city": {"type": "string"}}}}}
                        ]);
                        d
                    },
                    drifted: {
                        let mut d = canonical.clone();
                        d["tools"] = serde_json::json!([
                            {"type": "function",
                             "function": {"name": "get_weather",
                                          "description": "Reads the rooftop station feed.",
                                          "parameters": {"type": "object",
                                                         "properties": {"city": {"type": "string"}}}}}
                        ]);
                        let tools = d["tools"].as_array_mut().unwrap();
                        for t in tools.iter_mut() {
                            let obj = t.as_object_mut().unwrap();
                            let keys: Vec<String> = obj.keys().cloned().collect();
                            let mut taken: Vec<(String, serde_json::Value)> = Vec::new();
                            for k in keys {
                                let v = obj.shift_remove(&k).unwrap();
                                taken.push((k, v));
                            }
                            for (k, v) in taken.into_iter().rev() {
                                obj.insert(k, v);
                            }
                        }
                        d
                    },
                },
                // Content shape: a plain string re-sent as a one-block array.
                MatrixClass {
                    name: "content-string-vs-array",
                    dialect: "openai",
                    canonical,
                    drifted: {
                        let mut d = base_openai_body(PROSE, model);
                        let text = d["messages"][1]["content"].take();
                        d["messages"][1]["content"] =
                            serde_json::json!([{"type": "text", "text": text}]);
                        d
                    },
                },
            ]
        }
        crate::replay::Backend::Anthropic => {
            let canonical = base_anthropic_body(
                serde_json::json!([
                    {"type": "text", "text": PROSE, "cache_control": {"type": "ephemeral"}}
                ]),
                model,
            );
            vec![
                MatrixClass {
                    name: "tool-arg-reorder",
                    dialect: "anthropic",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["messages"][1]["content"][0]["input"] =
                            serde_json::json!({"unit": "celsius", "extended": 1, "city": "Paris"});
                        d
                    },
                },
                MatrixClass {
                    name: "whitespace",
                    dialect: "anthropic",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["system"][0]["text"] =
                            serde_json::Value::String(PROSE.replace(". ", ".  "));
                        d
                    },
                },
                // Hint placement moved from the system block to the tool
                // result block: placement is policy, content is identical.
                MatrixClass {
                    name: "hint-placement",
                    dialect: "anthropic",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        d["system"][0]
                            .as_object_mut()
                            .unwrap()
                            .remove("cache_control");
                        d["messages"][2]["content"][0]["cache_control"] =
                            serde_json::json!({"type": "ephemeral"});
                        d
                    },
                },
                // Tool definitions re-serialized (key order).
                MatrixClass {
                    name: "tools-reserialization",
                    dialect: "anthropic",
                    canonical: canonical.clone(),
                    drifted: {
                        let mut d = canonical.clone();
                        let tools = d["tools"].as_array_mut().unwrap();
                        for t in tools.iter_mut() {
                            let obj = t.as_object_mut().unwrap();
                            let keys: Vec<String> = obj.keys().cloned().collect();
                            let mut taken: Vec<(String, serde_json::Value)> = Vec::new();
                            for k in keys {
                                let v = obj.shift_remove(&k).unwrap();
                                taken.push((k, v));
                            }
                            for (k, v) in taken.into_iter().rev() {
                                obj.insert(k, v);
                            }
                        }
                        d
                    },
                },
                // System shape: block array vs bare string, both sides
                // hint-free — pure serialization drift, no placement policy
                // mixed in.
                MatrixClass {
                    name: "system-shape",
                    dialect: "anthropic",
                    canonical: base_anthropic_body(
                        serde_json::json!([{"type": "text", "text": PROSE}]),
                        model,
                    ),
                    drifted: base_anthropic_body(
                        serde_json::Value::String(PROSE.to_string()),
                        model,
                    ),
                },
                // Hint placement vs the managed baseline: this pair
                // measures placement policy AND shape together (the moved
                // hint also changes which span is cacheable) — the docs
                // say so, so a verdict here is not attributed to shape.
                MatrixClass {
                    name: "hint-placement",
                    dialect: "anthropic",
                    canonical: base_anthropic_body(
                        serde_json::json!([
                            {"type": "text", "text": PROSE, "cache_control": {"type": "ephemeral"}}
                        ]),
                        model,
                    ),
                    drifted: {
                        let mut d = base_anthropic_body(
                            serde_json::json!([
                                {"type": "text", "text": PROSE, "cache_control": {"type": "ephemeral"}}
                            ]),
                            model,
                        );
                        d["system"][0]
                            .as_object_mut()
                            .unwrap()
                            .remove("cache_control");
                        d["messages"][2]["content"][0]["cache_control"] =
                            serde_json::json!({"type": "ephemeral"});
                        d
                    },
                },
            ]
        }
    }
}

fn verdict_for(a: &FormSample, b: &FormSample) -> (&'static str, Option<u64>) {
    match (a.median_cached, b.median_cached) {
        (Some(ma), Some(mb)) => {
            // A delta is only comparable when one instance answered both
            // forms — a cross-namespace difference is routing, not drift.
            let shared =
                !a.instances.is_empty() && a.instances.iter().any(|i| b.instances.contains(i));
            if !shared {
                ("cross-instance", None)
            } else if mb > ma {
                ("costs", Some(mb - ma))
            } else if ma > mb {
                // The drifted form cached MORE: an inversion (router
                // artifact, warm cache, tokenizer quirk). Report it as
                // what it is — never fold it into "absorbed".
                ("inverted", Some(ma - mb))
            } else if ma == 0 && a.max_cached == Some(0) && b.max_cached == Some(0) {
                // Every reading on both forms, median AND max, was a
                // reported zero: the endpoint did not serve from cache at
                // all during the run. That is not "absorbed drift" — the
                // classes cost nothing because caching costs nothing — and
                // saying so would misstate the mechanism.
                ("no-caching", Some(0))
            } else {
                ("absorbed", Some(0))
            }
        }
        _ => ("unmeasured", None),
    }
}

/// Run every class for the backend against one endpoint, sequentially.
/// Within a class the delta is like-for-like (both forms share everything
/// outside the mutated span, so warming moves both medians together);
/// absolute medians are order-dependent across classes, which is why only
/// the delta is judged.
pub async fn run(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    classes: &[MatrixClass],
) -> Vec<ClassResult> {
    let mut out = Vec::new();
    for class in classes {
        let r = execute_bodies(
            client,
            cfg,
            &serde_json::Value::Null,
            &serde_json::Value::Null,
            &class.drifted,
            &class.canonical,
        )
        .await;
        let (verdict, delta) = verdict_for(&r.a_drifted, &r.b_canonical);
        out.push(ClassResult {
            name: class.name,
            a_drifted: r.a_drifted,
            b_canonical: r.b_canonical,
            verdict,
            delta,
        });
    }
    out
}

/// Render the cost table. Every class shows its own send count and instance
/// count; a class with a failed send is a gap on its own row, never a zero
/// blended into someone else's verdict.
pub fn render(cfg_endpoint: &str, samples: usize, results: &[ClassResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "drift-cost matrix · {cfg_endpoint} · n={samples}\n"
    ));
    out.push_str(&format!(
        "  {:<24} {:>11} {:>11} {:>7} {:>4}  {}\n",
        "class", "drifted m/m", "canonical m/m", "delta", "send", "verdict"
    ));
    for r in results {
        fn pair(f: &FormSample) -> String {
            match (f.median_cached, f.max_cached) {
                (Some(m), Some(x)) => format!("{m}/{x}"),
                _ => "—".to_string(),
            }
        }
        let delta = match (r.verdict, r.delta) {
            ("costs", Some(d)) => d.to_string(),
            ("inverted", Some(d)) => format!("+{d}"),
            (_, Some(_)) => "0".to_string(),
            (_, None) => "—".to_string(),
        };
        let sends = format!("{}/{}", r.a_drifted.sends, r.b_canonical.sends);
        out.push_str(&format!(
            "  {:<24} {:>11} {:>11} {:>7} {:>4}  {}\n",
            r.name,
            pair(&r.a_drifted),
            pair(&r.b_canonical),
            delta,
            sends,
            match r.verdict {
                "costs" => format!("costs {} tk (median)", r.delta.unwrap_or(0)),
                "inverted" => "inverted: drifted cached more".to_string(),
                "cross-instance" => "cross-instance: no like-for-like delta".to_string(),
                "no-caching" => "no caching on this endpoint".to_string(),
                v => v.to_string(),
            }
        ));
    }
    for r in results {
        if r.a_drifted.instances.len() > 1 || r.b_canonical.instances.len() > 1 {
            out.push_str(&format!(
                "  ! {}: more than one upstream instance answered — read max-of-N\n",
                r.name
            ));
        }
        if r.a_drifted.sends == 0 && r.b_canonical.sends == 0 {
            out.push_str(&format!(
                "  · {}: not a measurement — every send failed or carried no cache figure\n",
                r.name
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fixture_pair_is_semantically_equal_but_byte_different() {
        // Byte difference is generic; semantic equality is per class, in
        // the terms each class claims: JSON string fields are equal inside
        // their parsed form, key order is equal under sorted serialization,
        // number text is equal numerically, shape classes are equal field
        // by field.
        fn sorted_json(v: &serde_json::Value) -> String {
            fn sort(v: &serde_json::Value) -> serde_json::Value {
                match v {
                    serde_json::Value::Object(m) => {
                        let mut keys: Vec<&String> = m.keys().collect();
                        keys.sort();
                        let mut out = serde_json::Map::new();
                        for k in keys {
                            let inner = sort(&m[k]);
                            out.insert(k.clone(), inner);
                        }
                        serde_json::Value::Object(out)
                    }
                    serde_json::Value::Array(a) => {
                        serde_json::Value::Array(a.iter().map(sort).collect())
                    }
                    other => other.clone(),
                }
            }
            serde_json::to_string(&sort(v)).unwrap()
        }
        for backend in [
            crate::replay::Backend::OpenAi,
            crate::replay::Backend::Anthropic,
        ] {
            for class in classes(backend, "auto/fast") {
                let a = serde_json::to_string(&class.canonical).unwrap();
                let b = serde_json::to_string(&class.drifted).unwrap();
                assert_ne!(
                    a, b,
                    "{}: a drifted fixture that is byte-identical measures nothing",
                    class.name
                );
                match class.name {
                    "key-order" => {
                        // Object key order is not meaning: sorted form equal.
                        assert_eq!(
                            sorted_json(&class.canonical),
                            sorted_json(&class.drifted),
                            "{}: drifted and canonical are the same request",
                            class.name
                        );
                    }
                    "number-text" => {
                        // Same mathematical count in different number text.
                        let ca = class.canonical["messages"][2]["tool_calls"][0]["function"]
                            ["arguments"]
                            .as_str()
                            .unwrap();
                        let cd = class.drifted["messages"][2]["tool_calls"][0]["function"]
                            ["arguments"]
                            .as_str()
                            .unwrap();
                        let pa: serde_json::Value = serde_json::from_str(ca).unwrap();
                        let pd: serde_json::Value = serde_json::from_str(cd).unwrap();
                        assert_eq!(
                            pa["extended"].as_f64(),
                            pd["extended"].as_f64(),
                            "{}: the count is numerically identical",
                            class.name
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn key_order_fixture_actually_reorders_keys() {
        // preserve_order makes key order significant in equality; the
        // key-order fixture must therefore reorder, not just restate. Its
        // wire bytes differ (checked above) AND its message objects carry
        // rotated keys.
        let openai = classes(crate::replay::Backend::OpenAi, "auto/fast");
        let ko = openai
            .iter()
            .find(|c| c.name == "key-order")
            .expect("key-order fixture");
        let first_canonical = ko.canonical["messages"][0].as_object().unwrap();
        let first_drifted = ko.drifted["messages"][0].as_object().unwrap();
        let mut keys_canonical: Vec<_> = first_canonical.keys().collect();
        let mut keys_drifted: Vec<_> = first_drifted.keys().collect();
        keys_canonical.sort();
        keys_drifted.sort();
        assert_eq!(keys_canonical, keys_drifted, "same key set");
        assert_ne!(
            first_canonical.keys().collect::<Vec<_>>(),
            first_drifted.keys().collect::<Vec<_>>(),
            "keys rotated: insertion order differs"
        );
    }

    #[test]
    fn verdicts_are_plain_and_total() {
        let mk = |a: Option<u64>, b: Option<u64>, instance: &str| FormSample {
            form: "x",
            sends: 2,
            cached_readings: vec![],
            prompt_readings: vec![],
            median_cached: a,
            max_cached: b,
            instances: vec![instance.to_string()],
            failures: vec![],
        };
        let one = mk(Some(148), Some(148), "i");
        // Costs, absorbed, inverted: one instance answering both forms.
        let (v, d) = verdict_for(&one, &mk(Some(149), Some(149), "i"));
        assert_eq!((v, d), ("costs", Some(1)));
        let (v, d) = verdict_for(&one, &mk(Some(148), Some(148), "i"));
        assert_eq!((v, d), ("absorbed", Some(0)));
        let (v, d) = verdict_for(&mk(Some(150), Some(150), "i"), &one);
        assert_eq!(
            (v, d),
            ("inverted", Some(2)),
            "drifted cached more: said, never folded into absorbed"
        );
        // No shared instance: no comparable delta, whatever the numbers.
        let (v, d) = verdict_for(&one, &mk(Some(149), Some(149), "other"));
        assert_eq!((v, d), ("cross-instance", None));
        // All-zero readings on both forms is a non-caching endpoint, not
        // absorbed drift.
        let zeros = mk(Some(0), Some(0), "i");
        let (v, _) = verdict_for(&zeros, &zeros);
        assert_eq!(v, "no-caching");
        // Both medians zero but one max nonzero (a single send landed on a
        // caching instance amid zeros): a routing artifact, not evidence of
        // normalization — reads absorbed only because the medians held.
        let artifact = mk(Some(0), Some(384), "i");
        let (v, _) = verdict_for(&artifact, &zeros);
        assert_eq!(v, "absorbed", "median-level equality with a max artifact");
        let (v, d) = verdict_for(&mk(None, None, "i"), &mk(Some(5), Some(5), "i"));
        assert_eq!(v, "unmeasured");
        assert_eq!(d, None);
    }
}
