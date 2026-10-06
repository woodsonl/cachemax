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
    /// Plain verdict: "costs N tk" when the drifted form's first send after
    /// the canonical warm missed cache the canonical form had established,
    /// "absorbed" when it hit, "no caching" when nothing ever cached,
    /// "unmeasured" when a form had no readable reading.
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
const PROSE_PARA: &str = "You are the records assistant for the Kyoto facility audit. The lab protocol requires quoting the reference batch identifier in every reply, which is BATCH-7741-ALPHA-9, registered during the March audit window under supervision of the quality office. The protocol file lives in the east annex, revision fourteen, and any deviation from the quoted procedure must be reported within one business day to the same office, referencing the batch identifier and the audit window. Weather queries are answered from the rooftop station feed, which reports temperature, sky condition, and observation hours, and every answer names the station that produced the reading.";

/// The fixture system text, repeated to clear the strictest provider's
/// minimum cacheable prefix. Anthropic documents 2048 tokens; measured
/// live against api.anthropic.com, the observed floor for
/// claude-haiku-4-5 sits between 3884 and 5161 input tokens — consistent
/// with 4096 — so the fixtures clear the observed floor with margin.
/// Below it the provider refuses to cache at all, and the matrix would
/// read a false "no caching on this endpoint" for fixture-size reasons,
/// not endpoint reasons. Both forms share the same text, so class
/// semantics are unchanged — only the prefix is long enough to cache.
fn floored_prose() -> String {
    PROSE_PARA.repeat(40)
}

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

/// Stamp the class's own tag into the system prompt of BOTH forms, so
/// sequential classes cannot share a warmed prefix: without this, class N's
/// first send hits class N-1's leftover cache and reads a warm number on a
/// cold cache.
fn isolate(mut c: MatrixClass) -> MatrixClass {
    let tag = format!(" Class tag: {}.", c.name);
    for body in [&mut c.canonical, &mut c.drifted] {
        let mut tagged = false;
        // OpenAI dialect: the system rides as messages[0].content.
        if let Some(text) = body
            .pointer("/messages/0/content")
            .and_then(|v| v.as_str().map(str::to_string))
        {
            if let Some(v) = body.pointer_mut("/messages/0/content") {
                *v = serde_json::Value::String(format!("{text}{tag}"));
                tagged = true;
            }
        }
        // Anthropic dialect: the system rides as system[0].text, or as a
        // bare string (the system-shape class drifts between the two).
        if let Some(text) = body
            .pointer("/system/0/text")
            .and_then(|v| v.as_str().map(str::to_string))
        {
            if let Some(v) = body.pointer_mut("/system/0/text") {
                *v = serde_json::Value::String(format!("{text}{tag}"));
                tagged = true;
            }
        } else if let Some(text) = body
            .get("system")
            .and_then(|v| v.as_str().map(str::to_string))
        {
            if let Some(v) = body.get_mut("system") {
                *v = serde_json::Value::String(format!("{text}{tag}"));
                tagged = true;
            }
        }
        // A form the tagger cannot reach would silently break the pair's
        // equivalence — fail the run instead.
        assert!(tagged, "isolate: no system position found for {}", c.name);
    }
    c
}

/// The class fixtures for one backend dialect. Adding a class is adding a
/// fixture here; the runner and table pick it up automatically.
pub fn classes(backend: crate::replay::Backend, model: &str) -> Vec<MatrixClass> {
    match backend {
        crate::replay::Backend::OpenAi => {
            let canonical = base_openai_body(&floored_prose(), model);
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
                            serde_json::Value::String(floored_prose().replace(". ", ".  "));
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
                        let mut d = base_openai_body(&floored_prose(), model);
                        let text = d["messages"][1]["content"].take();
                        d["messages"][1]["content"] =
                            serde_json::json!([{"type": "text", "text": text}]);
                        d
                    },
                },
            ]
                .into_iter()
                .map(isolate)
                .collect()
        }
        crate::replay::Backend::Anthropic => {
            let canonical = base_anthropic_body(
                serde_json::json!([
                    {"type": "text", "text": floored_prose(), "cache_control": {"type": "ephemeral"}}
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
                            serde_json::Value::String(floored_prose().replace(". ", ".  "));
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
                        serde_json::json!([{"type": "text", "text": floored_prose()}]),
                        model,
                    ),
                    drifted: base_anthropic_body(serde_json::Value::String(floored_prose()), model),
                },
            ]
            .into_iter()
            .map(isolate)
            .collect()
        }
    }
}

/// The drift cost is the FIRST drifted send's reading against the cache
/// the canonical warm established: that is the repair scenario — the
/// provider holds the canonical form, the client arrives with drifted
/// bytes. Later drifted sends re-warm the drifted form's own entry and
/// hide the cost, which is why the first send is the judged number and the
/// steady-state readings stay visible in the columns.
fn verdict_for(
    first_drifted: Option<u64>,
    drift_max: Option<u64>,
    warm_max: Option<u64>,
) -> (&'static str, Option<u64>) {
    match (first_drifted, warm_max) {
        (Some(d), Some(c)) => {
            if c > d {
                ("costs", Some(c - d))
            } else if c == 0 && d == 0 && drift_max == Some(0) {
                // Nothing ever read above zero on either side: a different
                // mechanism than absorption, stated differently. A nonzero
                // drift max amid zero medians is a routing artifact —
                // absorbed-with-artifact, never "no caching".
                ("no-caching", Some(0))
            } else {
                // The drifted bytes hit the cache the canonical form
                // established: this endpoint keys on something the drift
                // does not touch.
                ("absorbed", Some(0))
            }
        }
        _ => ("unmeasured", None),
    }
}

/// Run every class for the backend against one endpoint, sequentially —
/// per-class system tags keep classes from sharing a warmed prefix. The
/// verdict compares the drifted form's FIRST send against the warm's
/// maximum (see [`verdict_for`]).
pub async fn run(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    classes: &[MatrixClass],
) -> Vec<ClassResult> {
    let mut out = Vec::new();
    for class in classes {
        // Warm the canonical form: two sends establish its cache entry
        // (one to write, one to confirm the read).
        let warm_cfg = ExecuteConfig {
            samples: 2,
            ..cfg.clone()
        };
        let warm = execute_bodies(
            client,
            &warm_cfg,
            &serde_json::Value::Null,
            &serde_json::Value::Null,
            &class.canonical,
            &class.canonical,
        )
        .await;
        // The drifted sends, from a cache that holds only the canonical
        // form. The FIRST reading is the cost; the rest show the
        // steady-state after the drifted form re-warms itself.
        let drift = execute_bodies(
            client,
            cfg,
            &serde_json::Value::Null,
            &serde_json::Value::Null,
            &class.drifted,
            &class.drifted,
        )
        .await;
        let first_drifted = drift.a_drifted.cached_readings.first().copied();
        // The baseline is the maximum across ALL warm sends — both forms
        // send the same canonical body, so every warm reading is evidence
        // of the level the canonical form demonstrably established.
        // Individual warm sends can read zero seconds after establishing
        // it (observed on real providers: eviction or shard routing);
        // judging against fewer readings would let flaky zeros declare
        // the cache absent. Max is monotone: more evidence can only raise
        // the baseline.
        let warm_max = warm
            .a_drifted
            .max_cached
            .into_iter()
            .chain(warm.b_canonical.max_cached)
            .max();
        let (verdict, delta) = verdict_for(first_drifted, drift.a_drifted.max_cached, warm_max);
        // The displayed canonical column carries the full warm evidence:
        // both forms' readings, not just b's half.
        let mut b_all = warm.b_canonical.clone();
        b_all
            .cached_readings
            .extend(warm.a_drifted.cached_readings.iter().copied());
        b_all.sends = b_all.cached_readings.len();
        b_all.median_cached = crate::replay::median(&b_all.cached_readings);
        b_all.max_cached = b_all.cached_readings.iter().copied().max();
        out.push(ClassResult {
            name: class.name,
            a_drifted: drift.a_drifted,
            b_canonical: b_all,
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
        "  {:<24} {:>13} {:>13} {:>7} {:>4}  {}\n",
        "class", "drifted 1st/m", "canonical m/m", "delta", "send", "verdict"
    ));
    for r in results {
        fn pair(f: &FormSample) -> String {
            match (f.cached_readings.first().copied(), f.max_cached) {
                (Some(first), Some(x)) => format!("{first}/{x}"),
                _ => "—".to_string(),
            }
        }
        fn canon(f: &FormSample) -> String {
            match (f.median_cached, f.max_cached) {
                (Some(m), Some(x)) => format!("{m}/{x}"),
                _ => "—".to_string(),
            }
        }
        let delta = match (r.verdict, r.delta) {
            ("costs", Some(d)) => d.to_string(),
            (_, Some(_)) => "0".to_string(),
            (_, None) => "—".to_string(),
        };
        let sends = format!("{}/{}", r.a_drifted.sends, r.b_canonical.sends);
        out.push_str(&format!(
            "  {:<24} {:>11} {:>11} {:>7} {:>4}  {}\n",
            r.name,
            pair(&r.a_drifted),
            canon(&r.b_canonical),
            delta,
            sends,
            match r.verdict {
                "costs" => {
                    format!("costs {} tk (first send)", r.delta.unwrap_or(0))
                }
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
                let tag = format!("Class tag: {}.", class.name);
                assert!(
                    a.contains(&tag) && b.contains(&tag),
                    "{}: both forms carry the class tag (isolation)",
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
        // Costs: the first drifted send missed what the canonical warm had.
        assert_eq!(
            verdict_for(Some(30), Some(199), Some(1400)),
            ("costs", Some(1370))
        );
        // Absorbed: the drifted bytes hit the canonical-established cache;
        // reading slightly MORE than the warm (tokenizer drift) is still
        // absorbed.
        assert_eq!(
            verdict_for(Some(1400), Some(1400), Some(1400)),
            ("absorbed", Some(0))
        );
        assert_eq!(
            verdict_for(Some(1410), Some(1410), Some(1400)),
            ("absorbed", Some(0))
        );
        // Flaky-zero robustness: the warm ENDED in zero but its max proves
        // the cache was established — the verdict still judges against the
        // established level.
        assert_eq!(
            verdict_for(Some(30), Some(199), Some(1400)),
            ("costs", Some(1370))
        );
        // A nonzero drift max amid zero first/warm is a routing artifact:
        // absorbed, never "no caching" — the row's own max would contradict
        // the mechanism claim.
        assert_eq!(
            verdict_for(Some(0), Some(384), Some(0)),
            ("absorbed", Some(0))
        );
        // Nothing ever read above zero on either side: no caching.
        assert_eq!(
            verdict_for(Some(0), Some(0), Some(0)),
            ("no-caching", Some(0))
        );
        // Either side unreadable: unmeasured.
        assert_eq!(verdict_for(None, Some(5), Some(5)), ("unmeasured", None));
        assert_eq!(verdict_for(Some(5), Some(5), None), ("unmeasured", None));
    }
}
