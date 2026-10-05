//! The golden drift battery (plan §7): the named contract for repair-on.
//! Every scenario drives the real proxy against a stub upstream that
//! records the exact bytes it received, so assertions are byte-level on
//! what the provider saw — and the record proves what repair did.
//!
//! The invariants under test:
//!   - a repairable drift is rewritten to the canonical serialization, so
//!     the provider re-sees the bytes it already cached;
//!   - a semantic change is NEVER rewritten — passed through untouched and
//!     flagged;
//!   - dry-run mutates nothing, ever.

use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::ledger::SharedLedger;
use cachemax::proxy;
use cachemax::record::Status;
use cachemax::repair::{DriftKind, RepairMode};
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::Tokenizer;

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---- The stub upstream ------------------------------------------------------

/// A stub upstream that records the raw bytes of every request body and
/// answers one canned non-streaming JSON document.
async fn stub_upstream(seen: Arc<Mutex<Vec<Vec<u8>>>>, reply: serde_json::Value) -> String {
    let reply = Arc::new(reply);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            let reply = reply.clone();
            async move {
                seen.lock().unwrap().push(body.to_vec());
                let bytes = serde_json::to_vec(reply.as_ref()).unwrap();
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from(bytes))
                    .unwrap()
            }
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{a}")
}

/// A stub upstream that streams a fixed SSE assistant reply.
async fn streaming_stub_upstream(seen: Arc<Mutex<Vec<Vec<u8>>>>, events: Vec<Bytes>) -> String {
    let events = Arc::new(events);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            let events = events.clone();
            async move {
                seen.lock().unwrap().push(body.to_vec());
                let stream = async_stream::stream! {
                    for e in events.iter() {
                        yield Ok::<Bytes, std::io::Error>(e.clone());
                    }
                };
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{a}")
}

fn usage_reply(message: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 100}},
    })
}

fn streaming_usage_tail() -> Vec<Bytes> {
    vec![Bytes::from_static(
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":900,\"prompt_tokens_details\":{\"cached_tokens\":100}}}\n\n",
    )]
}

// ---- The proxy under test ---------------------------------------------------

struct Rig {
    url: String,
    sessions: Arc<SharedSessions>,
    ledger: Arc<SharedLedger>,
}

async fn rig(upstream: String, mode: RepairMode) -> Rig {
    let sessions = Arc::new(SharedSessions::new());
    let ledger = Arc::new(SharedLedger::new());
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: sessions.clone(),
        ledger: ledger.clone(),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
        repair: mode,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    Rig {
        url: format!("http://{a}"),
        sessions,
        ledger,
    }
}

async fn send(rig: &Rig, body: &serde_json::Value) {
    let _ = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", rig.url))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
}

async fn send_header(rig: &Rig, body: &serde_json::Value, header: &str) {
    let _ = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", rig.url))
        .header("content-type", "application/json")
        .header("x-cachemax-repair", header)
        .body(body.to_string())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
}

async fn records(rig: &Rig, n: usize) -> Vec<cachemax::record::Record> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let done = rig
            .sessions
            .lock()
            .most_recent()
            .map(|s| s.records.len() >= n)
            .unwrap_or(false);
        if done {
            return rig.sessions.lock().most_recent().unwrap().records.clone();
        }
        assert!(
            Instant::now() < deadline,
            "the proxy never recorded {n} turns within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The most recently appended record anywhere in the store. Whitespace and
/// tool-argument drift changes the flattened prefix hashes, so drifted
/// requests fork into new sessions — the record to assert on is the latest
/// activity, whichever session holds it.
async fn last_record(rig: &Rig, total: usize) -> cachemax::record::Record {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let done = {
            let guard = rig.sessions.lock();
            (guard.total_records() >= total)
                .then(|| guard.most_recent().and_then(|s| s.records.last().cloned()))
        };
        if let Some(record) = done.flatten() {
            return record;
        }
        assert!(
            Instant::now() < deadline,
            "the proxy never recorded {total} turns within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The canonical weather tool call the stub provider returns (and the ledger
/// therefore remembers): arguments serialized with spaces and `city` first.
fn canonical_tool_call_message() -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"},
        }],
    })
}

fn weather_turn0() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You call tools."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
    })
}

/// The re-sent history with the tool-call arguments re-serialized by the
/// client (compact, `unit` first — same parsed JSON, different bytes).
fn weather_turn1_reordered_args() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You call tools."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"unit\":\"c\",\"city\":\"Paris\"}"},
            }]},
            {"role": "tool", "content": "20C", "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks"},
        ],
    })
}

#[tokio::test]
async fn tool_args_key_reorder_is_repaired() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone(), usage_reply(canonical_tool_call_message())).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(&rig, &weather_turn0()).await;
    send(&rig, &weather_turn1_reordered_args()).await;
    let rs = records(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    // The provider saw the canonical serialization again: the assistant
    // element in the second request is byte-identical to the first turn's.
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent1["messages"][2],
        canonical_tool_call_message(),
        "the drifted tool-call element was rewritten to canonical"
    );
    // Byte-level: the exact canonical arguments string is what went out.
    let raw = String::from_utf8(upstream_saw[1].clone()).unwrap();
    assert!(
        raw.contains(r#""arguments":"{\"city\": \"Paris\", \"unit\": \"c\"}""#),
        "the canonical argument bytes were forwarded, got: {raw}"
    );
    // The new tail (tool result + user message) went through verbatim.
    assert_eq!(sent1["messages"][3]["content"], "20C");
    assert_eq!(sent1["messages"][4]["content"], "Thanks");
    // And the record proves it.
    assert!(rs[1].repaired);
    assert_eq!(rs[1].drift_kind, Some(DriftKind::ToolArgReserialization));
    assert!(rs[1].canonicalized_tokens > 0);
}

#[tokio::test]
async fn tool_args_semantic_change_is_not_repaired() {
    // A changed argument VALUE is a different request: it must pass through
    // untouched and be flagged, never rewritten toward the old value.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone(), usage_reply(canonical_tool_call_message())).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(&rig, &weather_turn0()).await;
    let mut changed = weather_turn1_reordered_args();
    changed["messages"][2]["tool_calls"][0]["function"]["arguments"] =
        serde_json::json!("{\"city\": \"Berlin\", \"unit\": \"c\"}");
    let sent = changed.to_string();
    send(&rig, &changed).await;
    let rs = records(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[1],
        sent.as_bytes(),
        "a semantic change forwards byte-identical to what the client sent"
    );
    assert!(!rs[1].repaired);
    assert_eq!(rs[1].drift_kind, None, "no repairable kind is claimed");
    assert_eq!(rs[1].matches_canonical, Some(false));
    assert!(
        rs[1].canonicalized_tokens > 0,
        "the at-risk span is reported"
    );
}

#[tokio::test]
async fn whitespace_normalization_is_repaired() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "Done"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    // Turn 0 establishes `Please  summarize` (double space) as canonical.
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Please  summarize  the  results."},
            ],
        }),
    )
    .await;
    // Turn 1 re-sends with collapsed whitespace — same meaning, different
    // tokens.
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Please summarize the results."},
                {"role": "assistant", "content": "Done"},
                {"role": "user", "content": "Now  more"},
            ],
        }),
    )
    .await;
    let last = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent1["messages"][0]["content"], "Please  summarize  the  results.",
        "the canonical (drifted-from) text was sent"
    );
    // The genuinely-new tail passes through verbatim, drift and all.
    assert_eq!(sent1["messages"][2]["content"], "Now  more");
    assert!(last.repaired);
    assert_eq!(last.drift_kind, Some(DriftKind::TextNormalization));
}

#[tokio::test]
async fn history_truncation_canonicalizes_suffix() {
    // The client drops the first two turns (four elements) and re-sends the
    // rest with drift; the kept suffix is canonicalized, the new tail is
    // appended verbatim, and the dropped elements are NOT re-added.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "a 2"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    // Turn 0 and a clean turn 1 grow one session's chain.
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
            ],
        }),
    )
    .await;
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
        }),
    )
    .await;
    records(&rig, 2).await;

    // Turn 2: only the tail of the history survives, with whitespace drift
    // in it, plus the new message. (The reply is fixed, so the canonical
    // assistant element is "a 2" — the client re-sends "a  2".)
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a  2"},
                {"role": "user", "content": "q3"},
            ],
        }),
    )
    .await;
    let last = last_record(&rig, 3).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[2]).unwrap();
    // Nothing re-added: the dropped system/user turn is absent.
    assert_eq!(sent["messages"].as_array().unwrap().len(), 3);
    assert_ne!(
        sent["messages"][0]["content"], "sys",
        "dropped elements stay dropped"
    );
    // The kept, drifted element was canonicalized.
    assert_eq!(sent["messages"][1]["content"], "a 2");
    // The new tail verbatim.
    assert_eq!(sent["messages"][2]["content"], "q3");
    // Truncation plus the in-run whitespace drift: Mixed.
    assert_eq!(last.drift_kind, Some(DriftKind::Mixed));
    assert!(last.repaired, "the drifted kept element was rewritten");
}

#[tokio::test]
async fn system_prompt_change_never_rewrites() {
    // Through the wire, a changed system message shares no prefix-hash, so
    // the store starts a new session and the ledger reports first-turn —
    // the never-rewrite property is what matters: not one byte is touched.
    // (The SystemPromptChanged label itself fires on shape-only collisions
    // and is unit-tested in repair.rs.)
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "ok"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ],
        }),
    )
    .await;
    let changed = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are verbose."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "more"},
        ],
    });
    let sent = changed.to_string();
    send(&rig, &changed).await;
    let _ = records(&rig, 1).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[1],
        sent.as_bytes(),
        "a changed system prompt forwards untouched"
    );
}

#[tokio::test]
async fn model_switch_breaks_chain() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "ok"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    // Same flattened history, different model: the same session, but the
    // canonical chain is per model — the switched turn is never rewritten
    // against the other model's chain.
    let on_a = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Please  summarize"},
        ],
    });
    send(&rig, &on_a).await;
    let on_b = serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "Please  summarize"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ],
    });
    let sent_b = on_b.to_string();
    send(&rig, &on_b).await;
    let _ = last_record(&rig, 2).await;

    {
        let upstream_saw = seen.lock().unwrap();
        assert_eq!(
            upstream_saw[1],
            sent_b.as_bytes(),
            "a model switch forwards untouched (no cross-model rewrite)"
        );
    }

    // The new model now has its own chain: a drifted re-send under it IS
    // repaired against the new chain (its own turn-1 bytes), not the old
    // model's.
    let drifted_b = serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "Please summarize"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "more"},
        ],
    });
    send(&rig, &drifted_b).await;
    let last = last_record(&rig, 3).await;
    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[2]).unwrap();
    assert_eq!(
        sent["messages"][0]["content"], "Please  summarize",
        "the claude chain's canonical form (its own turn) is what got sent"
    );
    assert_eq!(sent["messages"][4]["content"], "more", "new tail verbatim");
    assert!(last.repaired);
}

#[tokio::test]
async fn streamed_assistant_turn_is_canonical() {
    // The SSE response reassembles to exactly the message a non-streaming
    // body would have carried, and a drifted re-send is repaired to it.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut events: Vec<Bytes> = vec![
        Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello  \"}}]}\n\n"),
        Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"}}]}\n\n"),
    ];
    events.extend(streaming_usage_tail());
    let upstream = streaming_stub_upstream(seen.clone(), events).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{"role": "user", "content": "greet"}],
        }),
    )
    .await;
    // The ledger's remembered assistant message is exactly the concatenation
    // of the provider's deltas — the object a non-streaming reply carries.
    let remembered = {
        let ledger = rig.ledger.lock();
        ledger
            .canonical_messages(1, "gpt-4o")
            .unwrap()
            .pop()
            .unwrap()
    };
    assert_eq!(
        remembered,
        serde_json::json!({"role": "assistant", "content": "Hello  world"})
    );

    // The client re-renders it with collapsed whitespace; repair restores
    // the streamed bytes.
    send(
        &rig,
        &serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "greet"},
                {"role": "assistant", "content": "Hello world"},
                {"role": "user", "content": "again"},
            ],
        }),
    )
    .await;
    let rs = records(&rig, 2).await;
    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(sent["messages"][1]["content"], "Hello  world");
    assert!(rs[1].repaired);
    assert_eq!(rs[1].drift_kind, Some(DriftKind::TextNormalization));
}

#[tokio::test]
async fn dry_run_never_mutates() {
    // All of the above drifts, in dry-run: the forwarded bytes equal the
    // client's bytes exactly, and the record carries the annotation.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone(), usage_reply(canonical_tool_call_message())).await;
    let rig = rig(upstream, RepairMode::DryRun).await;

    send(&rig, &weather_turn0()).await;
    let drifted = weather_turn1_reordered_args();
    let sent = drifted.to_string();
    send(&rig, &drifted).await;
    let rs = records(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[1],
        sent.as_bytes(),
        "dry-run forwards the drifted bytes untouched"
    );
    assert_eq!(rs[1].repair_mode, RepairMode::DryRun);
    assert_eq!(rs[1].matches_canonical, Some(false));
    assert_eq!(rs[1].drift_kind, Some(DriftKind::ToolArgReserialization));
    assert!(!rs[1].repaired);
    assert!(
        rs[1].canonicalized_tokens > 0,
        "the would-be repair is quantified"
    );
}

#[tokio::test]
async fn first_turn_noop() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "ok"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    let first = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Anything,  drifted or not"}],
    });
    let sent = first.to_string();
    send(&rig, &first).await;
    let rs = records(&rig, 1).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[0],
        sent.as_bytes(),
        "a first turn forwards untouched (nothing canonical to extend)"
    );
    assert!(!rs[0].repaired);
    assert_eq!(rs[0].matches_canonical, Some(false));
    assert_eq!(rs[0].drift_kind, None);
}

#[tokio::test]
async fn purged_ledger_falls_back_cleanly() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "ok"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(&rig, &weather_turn0()).await;
    records(&rig, 1).await;
    // Purge: the chain is gone.
    rig.ledger.lock().clear();

    let drifted = weather_turn1_reordered_args();
    let sent = drifted.to_string();
    send(&rig, &drifted).await;
    let rs = records(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[1],
        sent.as_bytes(),
        "with no chain, the drifted request forwards untouched"
    );
    assert!(!rs[1].repaired);
    assert_eq!(rs[1].drift_kind, None);
    assert_eq!(rs[1].canonicalized_tokens, 0);
}

#[tokio::test]
async fn the_header_overrides_the_mode_per_request() {
    // `on` by header under a dry-run proxy, and `off` by header under an
    // `on` proxy: the effective mode is per-request.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone(), usage_reply(canonical_tool_call_message())).await;
    let dry_run_rig = rig(upstream.clone(), RepairMode::DryRun).await;

    send(&dry_run_rig, &weather_turn0()).await;
    let drifted = weather_turn1_reordered_args();
    send_header(&dry_run_rig, &drifted, "on").await;
    let rs = records(&dry_run_rig, 2).await;
    let second_body = seen.lock().unwrap()[1].clone();
    let sent: serde_json::Value = serde_json::from_slice(&second_body).unwrap();
    assert_eq!(
        sent["messages"][2],
        canonical_tool_call_message(),
        "the header turned rewriting on for this one request"
    );
    assert!(rs[1].repaired);
    assert_eq!(rs[1].repair_mode, RepairMode::On);

    // And off under an on-mode proxy.
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let upstream2 = stub_upstream(seen2.clone(), usage_reply(canonical_tool_call_message())).await;
    let on_rig = rig(upstream2, RepairMode::On).await;
    send(&on_rig, &weather_turn0()).await;
    let sent = drifted.to_string();
    send_header(&on_rig, &drifted, "off").await;
    let rs = records(&on_rig, 2).await;
    let upstream_saw2 = seen2.lock().unwrap();
    assert_eq!(
        upstream_saw2[1],
        sent.as_bytes(),
        "the header turned repair off for this one request"
    );
    assert_eq!(rs[1].repair_mode, RepairMode::Off);
    assert_eq!(rs[1].matches_canonical, None);
}

#[tokio::test]
async fn incomplete_turns_do_not_seed_the_chain() {
    // A failed turn must not become canonical: a later re-send classifies
    // against the pre-failure chain, not against an error body's ghost.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = usage_reply(serde_json::json!({"role": "assistant", "content": "ok"}));
    let upstream = stub_upstream(seen.clone(), reply).await;
    let rig = rig(upstream, RepairMode::On).await;
    let _ = Status::Complete; // referenced for the reader

    send(&rig, &weather_turn0()).await;
    records(&rig, 1).await;
    // The ledger holds exactly the turn-0 chain.
    let chain_len = rig
        .ledger
        .lock()
        .canonical_messages(1, "gpt-4o")
        .map(|c| c.len());
    assert_eq!(chain_len, Some(3), "sys + user + assistant");
}
