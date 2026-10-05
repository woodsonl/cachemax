//! Anthropic top-level `system` repair, end-to-end: the system prompt is
//! cache-prime material, and drift there is total cache loss, so it rides
//! the same equivalence ladder as messages — canonicalized when
//! equivalent-but-not-exact, and never touched when semantically different.
//!
//! Driven against the real proxy and a stub upstream that records the exact
//! bytes it received.

use cachemax::adapters::anthropic::AnthropicAdapter;
use cachemax::ledger::SharedLedger;
use cachemax::proxy;
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

async fn stub_upstream(seen: Arc<Mutex<Vec<Vec<u8>>>>) -> String {
    let reply = serde_json::json!({
        "id": "msg_1",
        "content": [{"type": "text", "text": "Done."}],
        "usage": {
            "input_tokens": 100,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
            "output_tokens": 5,
        },
    });
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push(body.to_vec());
                let bytes = serde_json::to_vec(&reply).unwrap();
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

struct Rig {
    url: String,
    sessions: Arc<SharedSessions>,
    ledger: Arc<SharedLedger>,
}

async fn rig(upstream: String, mode: RepairMode) -> Rig {
    let sessions = Arc::new(SharedSessions::new());
    let ledger = Arc::new(SharedLedger::new());
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(AnthropicAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: sessions.clone(),
        ledger: ledger.clone(),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
        repair: mode,
        manage_breakpoints: false,
        force_breakpoints: false,
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

fn user(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": [{"type": "text", "text": text}]})
}

/// Turn 1's messages: turn 0's canonical chain plus a new user turn.
fn extended(rig: &Rig, model: &str, text: &str) -> serde_json::Value {
    let mut messages = rig
        .ledger
        .lock()
        .canonical_messages(1, model)
        .expect("turn 0 is canonical");
    messages.push(user(text));
    serde_json::Value::Array(messages)
}

#[tokio::test]
async fn system_drift_is_repaired_to_the_canonical_serialization() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    // Turn 0: a system string carrying a source artifact (double space).
    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": "Be  terse.",
        "messages": [user("q1")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Turn 1: the client re-emits the same system text with the artifact
    // normalized away (an SDK pass through a text scorer), plus a new turn.
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": "Be terse.",
        "messages": extended(&rig, "claude-3", "q2"),
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent1["system"], "Be  terse.",
        "the system is rewritten to the canonical bytes"
    );
    assert_eq!(record1.drift_kind, Some(DriftKind::TextNormalization));
    assert!(record1.repaired, "the rewrite is receipted");
    assert_eq!(record1.matches_canonical, Some(false));
    assert!(
        record1.canonicalized_tokens > 0,
        "the repaired span is token-quantified"
    );
}

#[tokio::test]
async fn a_semantically_different_system_is_never_rewritten() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": "Be  terse.",
        "messages": [user("q1")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Turn 1 changes the system to a different instruction AND carries
    // message drift. A different system re-bases everything downstream:
    // nothing this turn may be rewritten, including the messages.
    let mut messages = rig.ledger.lock().canonical_messages(1, "claude-3").unwrap();
    // Introduce message drift too (collapsed spaces in the prior user turn).
    messages[0] = user("q  1");
    messages.push(user("q2"));
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": "Be verbose instead.",
        "messages": messages,
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent1["system"], "Be verbose instead.",
        "a different system passes through untouched"
    );
    assert_eq!(
        sent1["messages"][0],
        user("q  1"),
        "message drift is also left untouched behind a rebase"
    );
    assert!(!record1.repaired, "nothing was rewritten on a rebased turn");
    assert_eq!(record1.canonicalized_tokens, 0);
}

#[tokio::test]
async fn system_and_message_drift_together_read_as_mixed() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": "Be  terse.",
        "messages": [user("q  1")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Both the system (double→single space, TextNormalization) and a prior
    // message (block content → plain string, RoleContentReshaped) were
    // touched: two different drift flavors in one turn read as mixed, and
    // both are rewritten.
    let mut messages = rig.ledger.lock().canonical_messages(1, "claude-3").unwrap();
    messages[0] = serde_json::json!({"role": "user", "content": "q  1"});
    messages.push(user("q2"));
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": "Be terse.",
        "messages": messages,
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(sent1["system"], "Be  terse.");
    assert_eq!(sent1["messages"][0], user("q  1"));
    assert_eq!(record1.drift_kind, Some(DriftKind::Mixed));
    assert!(record1.repaired);
    assert_eq!(record1.matches_canonical, Some(false));
}

#[tokio::test]
async fn a_stable_system_reads_clean() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": "Be terse.",
        "messages": [user("q1")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": "Be terse.",
        "messages": extended(&rig, "claude-3", "q2"),
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    assert!(
        !record1.repaired,
        "a byte-stable system never reads as drifted"
    );
    assert_eq!(record1.canonicalized_tokens, 0);
}

#[tokio::test]
async fn a_system_block_with_siblings_is_never_reshaped_away() {
    // A text block carrying a sibling key (`annotations`, `signature`, …) is
    // not proven equal to a bare string or a two-key block: the siblings are
    // semantic and repair must never drop them. The turn is flagged, not
    // rewritten.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": "Be terse.",
        "messages": [user("q1")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Turn 1 re-emits the same text as a block that also carries a citation
    // annotation. The annotation is content the client sent; it must survive.
    let annotated = serde_json::json!([
        {"type": "text", "text": "Be terse.", "annotations": [{"type": "url_citation"}]}
    ]);
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": annotated,
        "messages": extended(&rig, "claude-3", "q2"),
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent1["system"], annotated,
        "the annotated block passes through untouched"
    );
    assert!(
        !record1.repaired,
        "a system block with semantic siblings is not rewritten"
    );
}
