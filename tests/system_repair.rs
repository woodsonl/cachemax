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
    let reply2 = reply.clone();
    let seen2 = seen.clone();
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(move |body: Bytes| {
                let seen = seen2.clone();
                let reply = reply2.clone();
                async move {
                    seen.lock().unwrap().push(body.to_vec());
                    let bytes = serde_json::to_vec(&reply).unwrap();
                    Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(bytes))
                        .unwrap()
                }
            }),
        )
        .route(
            "/v1/messages",
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

    // Turn 1 changes the system to a different instruction while the
    // messages stay an exact continuation of the chain (byte-identical
    // prefix, new tail): the SystemPromptChanged path must fire for the
    // system alone, with the at-risk span quantified from the canonical
    // system — and nothing rewritten.
    let mut messages = rig.ledger.lock().canonical_messages(1, "claude-3").unwrap();
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
        sent1["messages"][2],
        user("q2"),
        "the new tail passes through untouched"
    );
    assert!(
        !record1.repaired,
        "nothing was rewritten behind a changed system"
    );
    assert!(
        record1.canonicalized_tokens > 0,
        "the endangered span is quantified from the canonical system, got {}",
        record1.canonicalized_tokens
    );
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

#[tokio::test]
async fn a_model_switch_is_never_mislabeled_as_a_system_change() {
    // Switching models starts a new chain: there is no canonical system to
    // differ from. Reading that absence as "changed" fabricated a hard stop
    // with an at-risk figure counted from the literal "null" (1 tk) — the
    // switch must read as its own honest event with nothing at risk.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be terse.",
            "messages": [user("q1")],
        }),
    )
    .await;
    last_record(&rig, 1).await;
    send(
        &rig,
        &serde_json::json!({
            "model": "claude-4",
            "system": "Be terse.",
            "messages": [user("q1")],
        }),
    )
    .await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent["system"], "Be terse.",
        "the switched turn forwards untouched"
    );
    assert!(!record.repaired);
    assert_eq!(
        record.canonicalized_tokens, 0,
        "a switch has no at-risk span; the 'null' figure was a fabrication"
    );
}

#[tokio::test]
async fn dry_run_never_rewrites_the_system() {
    // The system rewrite sits behind the same mode gate as the message
    // rewrite. If a regression hoisted it out of the gate, dry-run would
    // silently mutate requests — the one thing it may never do.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::DryRun).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be  terse.",
            "messages": [user("q1")],
        }),
    )
    .await;
    last_record(&rig, 1).await;
    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be terse.",
            "messages": [user("q1"), user("q2")],
        }),
    )
    .await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent["system"], "Be terse.",
        "dry run forwards the client's system verbatim"
    );
    assert!(!record.repaired);
}

#[tokio::test]
async fn client_managed_hints_are_never_stripped_by_a_rewrite() {
    // repair and breakpoint management are independent flags. With manage
    // off, the client owns hint placement: a system rewrite would strip
    // their cache_control and silently erase their breakpoint — the proxy
    // inducing the total cache loss it exists to prevent. The drifted
    // system passes through untouched instead.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be  terse.",
            "messages": [user("q1")],
        }),
    )
    .await;
    last_record(&rig, 1).await;
    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": [
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [user("q1"), user("q2")],
        }),
    )
    .await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent["system"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral"}),
        "the client's breakpoint survives the drifted system"
    );
    assert!(!record.repaired);
}

#[tokio::test]
async fn recorded_ttft_covers_the_upstream_wait() {
    // The TTFT clock is seeded from request-send, so the recorded figure
    // includes the upstream's full first-token wait. An upstream that waits
    // 150ms before its first byte must produce a record of at least 100ms;
    // an unseeded clock (headers→first-chunk on a buffered body) reads
    // sub-millisecond, so the assertion fails in the right direction.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = serde_json::json!({
        "id": "msg_1",
        "content": [{"type": "text", "text": "Done."}],
        "usage": {"input_tokens": 100, "output_tokens": 5},
    });
    let reply2 = reply.clone();
    let seen2 = seen.clone();
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(move |body: Bytes| {
                let seen = seen2.clone();
                let reply = reply2.clone();
                async move {
                    seen.lock().unwrap().push(body.to_vec());
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    let bytes = serde_json::to_vec(&reply).unwrap();
                    Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(bytes))
                        .unwrap()
                }
            }),
        )
        .route(
            "/v1/messages",
            post(move |body: Bytes| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(body.to_vec());
                    tokio::time::sleep(Duration::from_millis(150)).await;
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
    let rig = rig(format!("http://{a}"), RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be terse.",
            "messages": [user("q1")],
        }),
    )
    .await;
    let record = last_record(&rig, 1).await;
    assert!(
        record.ttft_ms.unwrap_or(0.0) >= 100.0,
        "ttft must cover send→first-byte, got {:?}",
        record.ttft_ms
    );
}

#[tokio::test]
async fn a_system_without_a_recorded_baseline_is_unexamined_not_matching() {
    // A chain recorded before systems were captured carries no baseline:
    // the next turn's system goes unexamined. The record must say so —
    // `None`, not a whole-span match that covers a span never compared.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "messages": [user("q1")],
        }),
    )
    .await;
    last_record(&rig, 1).await;
    let mut messages = rig.ledger.lock().canonical_messages(1, "claude-3").unwrap();
    messages.push(user("q2"));
    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be terse.",
            "messages": messages,
        }),
    )
    .await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    assert_eq!(
        sent["system"], "Be terse.",
        "the unexamined system forwards untouched"
    );
    assert!(!record.repaired);
    assert_eq!(
        record.matches_canonical, None,
        "an unexamined span is not asserted as matching"
    );
}

#[tokio::test]
async fn a_known_message_mismatch_survives_an_unexamined_system() {
    // Unexamined system plus PROVEN message drift: the mismatch is fact and
    // must reach the dashboard (which renders drift on Some(false)) —
    // downgrading it to None would hide a mismatch the classifier found.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On).await;

    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "messages": [user("q1")],
        }),
    )
    .await;
    last_record(&rig, 1).await;
    let mut messages = rig.ledger.lock().canonical_messages(1, "claude-3").unwrap();
    messages[0] = user("q  1"); // whitespace drift in the recorded turn itself
    messages.push(user("q2"));
    send(
        &rig,
        &serde_json::json!({
            "model": "claude-3",
            "system": "Be terse.",
            "messages": messages,
        }),
    )
    .await;
    let record = last_record(&rig, 2).await;

    assert!(!record.repaired);
    assert_eq!(
        record.matches_canonical,
        Some(false),
        "proven message drift is reported even with the system unexamined"
    );
}
