//! Breakpoint management battery (plan §4): the contract for
//! `--manage-breakpoints`, driven end-to-end against the real proxy and a
//! stub upstream that records the exact bytes it received.
//!
//! The invariants under test:
//!   - hints land on the last system block plus the last user/tool-result
//!     blocks, never more than 4, never duplicated;
//!   - placements step forward as the conversation grows;
//!   - a client that manages its own breakpoints is passed through
//!     untouched (unless forced);
//!   - management composes with repair: a canonical rewrite carries hints,
//!     and echoed hints never read as drift.

use cachemax::adapters::anthropic::AnthropicAdapter;
use cachemax::ledger::SharedLedger;
use cachemax::proxy;
use cachemax::repair::RepairMode;
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::Tokenizer;

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A stub Anthropic upstream: records raw request bytes, answers one canned
/// non-streaming message with the write/read cache split.
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
    // The proxy posts the Anthropic dialect to the native /v1/messages;
    // the compat route stays for any client that still speaks it.
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

async fn rig(upstream: String, mode: RepairMode, manage: bool, force: bool) -> Rig {
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
        manage_breakpoints: manage,
        force_breakpoints: force,
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

/// The most recently appended record anywhere in the store (bounded wait).
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

/// The message indices whose content carries a cache hint.
fn hinted_messages(doc: &serde_json::Value) -> Vec<usize> {
    doc["messages"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m["content"].as_array().is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|b| b.get("cache_control").is_some_and(|c| c.is_object()))
            })
        })
        .map(|(i, _)| i)
        .collect()
}

fn system_hinted(doc: &serde_json::Value) -> bool {
    doc["system"].as_array().is_some_and(|blocks| {
        blocks
            .last()
            .is_some_and(|b| b.get("cache_control").is_some_and(|c| c.is_object()))
    })
}

fn user(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": [{"type": "text", "text": text}]})
}

#[tokio::test]
async fn places_and_steps_breakpoints_across_turns() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::DryRun, true, false).await;

    // Turn 0: block-form system and one user message.
    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [user("q1")],
    });
    send(&rig, &turn0).await;
    let record0 = last_record(&rig, 1).await;

    let sent0: serde_json::Value = {
        let guard = seen.lock().unwrap();
        serde_json::from_slice(&guard[0]).unwrap()
    };
    assert!(
        system_hinted(&sent0),
        "the last system block carries a hint"
    );
    assert_eq!(hinted_messages(&sent0), vec![0]);
    assert_eq!(record0.breakpoint_count, Some(2));

    // Turn 1: a compliant client echoes the conversation as it appeared on
    // the wire (the ledger's chain: as-sent user + as-received assistant)
    // and adds a new user turn.
    let mut messages = rig
        .ledger
        .lock()
        .canonical_messages(1, "claude-3")
        .expect("turn 0 is canonical");
    messages.push(user("q2"));
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": sent0["system"].clone(),
        "messages": messages,
    });
    send(&rig, &turn1).await;
    let record1 = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    // The hints stepped forward: system + both user blocks, never more
    // than 4, the echoed placements re-derived rather than duplicated.
    assert!(system_hinted(&sent1));
    assert_eq!(hinted_messages(&sent1), vec![0, 2]);
    assert_eq!(record1.breakpoint_count, Some(3));
    assert_eq!(
        rig.ledger
            .lock()
            .last_turn(1, "claude-3")
            .map(|t| t.breakpoints),
        Some(3),
        "the ledger remembers what the proxy placed"
    );
    // And the echo was never drift: the match model ignores hints.
    assert_eq!(
        record1.matches_canonical,
        Some(true),
        "echoed hints are not drift"
    );
    assert_eq!(record1.drift_kind, None);
}

#[tokio::test]
async fn client_managed_breakpoints_pass_through_untouched() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::DryRun, true, false).await;

    // The client places its own hints where the proxy's rule would not.
    let client = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
            ]},
            user("q2"),
        ],
    });
    let sent = client.to_string();
    send(&rig, &client).await;
    let record = last_record(&rig, 1).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(
        upstream_saw[0],
        sent.as_bytes(),
        "a client-managed request forwards byte-identical"
    );
    assert_eq!(record.breakpoint_count, Some(1), "its own hint is counted");
    assert_eq!(
        rig.ledger
            .lock()
            .last_turn(1, "claude-3")
            .map(|t| t.breakpoints),
        Some(0),
        "and never claimed as the proxy's"
    );
}

#[tokio::test]
async fn force_breakpoints_re_derives_over_client_placements() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::DryRun, true, true).await;

    let client = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "q1", "cache_control": {"type": "ephemeral"}},
            ]},
            user("q2"),
        ],
    });
    send(&rig, &client).await;
    let record = last_record(&rig, 1).await;

    let upstream_saw = seen.lock().unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&upstream_saw[0]).unwrap();
    assert!(system_hinted(&sent));
    // Re-derived per the rule: both user messages are within the last
    // three, so both carry hints — wherever the client's own sat.
    assert_eq!(hinted_messages(&sent), vec![0, 1]);
    assert_eq!(record.breakpoint_count, Some(3));
}

#[tokio::test]
async fn management_composes_with_repair() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On, true, false).await;

    // Turn 0 establishes the canonical serialization (double space).
    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be  brief"}],
        "messages": [user("Please  summarize")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Turn 1: the client re-sends whitespace-drifted history in its own
    // serialization (no hints) plus the new tail.
    let mut messages = rig
        .ledger
        .lock()
        .canonical_messages(1, "claude-3")
        .expect("turn 0 is canonical");
    {
        let block = &mut messages[0]["content"][0];
        block["text"] = serde_json::json!("Please summarize");
        block.as_object_mut().unwrap().remove("cache_control");
    }
    messages.push(user("Now  more"));
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": messages,
    });
    send(&rig, &turn1).await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    // The drifted message element was rewritten to the canonical
    // serialization (hints included — they are part of what went out)...
    assert_eq!(
        sent1["messages"][0]["content"][0]["text"], "Please  summarize",
        "the canonical text went out"
    );
    // ...the top-level system was ALSO rewritten to its canonical
    // serialization (the Anthropic system prompt is prime cache material
    // and rides the same ladder as messages, even though it lives outside
    // `messages`)...
    assert_eq!(sent1["system"][0]["text"], "be  brief");
    // ...the new tail verbatim, drift and all...
    assert_eq!(sent1["messages"][2]["content"][0]["text"], "Now  more");
    // ...and the hints are freshly placed on the result.
    assert!(system_hinted(&sent1));
    assert_eq!(hinted_messages(&sent1), vec![0, 2]);
    // The record proves both features ran.
    assert!(record.repaired);
    assert_eq!(
        record.drift_kind,
        Some(cachemax::repair::DriftKind::TextNormalization)
    );
    assert_eq!(record.breakpoint_count, Some(3));
}

#[tokio::test]
async fn a_drifted_rewrite_never_leaks_our_hints_into_a_client_managed_request() {
    // The composition the scalar rules got wrong: the client drifts its
    // history AND manages its own hints (here: an assistant-block hint
    // with a ttl — shapes the proxy never writes). Repair rewrites the
    // drifted element to canonical content; those replacements must not
    // carry the chain's hints, or the declined pass-through would forward
    // more hints than the provider's 4-block limit accepts.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let rig = rig(upstream, RepairMode::On, true, false).await;

    // Turn 0: the proxy places its hints (system + the user block).
    let turn0 = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [user("Please  summarize")],
    });
    send(&rig, &turn0).await;
    last_record(&rig, 1).await;

    // Turn 1: whitespace-drifted history in the client's own serialization
    // (no echoed hints) plus its own ttl hint on the assistant block.
    let chain = rig
        .ledger
        .lock()
        .canonical_messages(1, "claude-3")
        .expect("turn 0 is canonical");
    let mut messages = vec![json!({
        "role": "user",
        "content": [{"type": "text", "text": "Please summarize"}],
    })];
    let mut hinted_assistant = chain[1].clone();
    hinted_assistant["content"][0]["cache_control"] =
        serde_json::json!({"type": "ephemeral", "ttl": "1h"});
    messages.push(hinted_assistant);
    messages.push(user("Now  more"));
    let turn1 = serde_json::json!({
        "model": "claude-3",
        "system": [{"type": "text", "text": "be brief"}],
        "messages": messages,
    });
    send(&rig, &turn1).await;
    let record = last_record(&rig, 2).await;

    let upstream_saw = seen.lock().unwrap();
    let sent1: serde_json::Value = serde_json::from_slice(&upstream_saw[1]).unwrap();
    // Repair canonicalized the drifted element — content only.
    assert_eq!(
        sent1["messages"][0]["content"][0]["text"],
        "Please  summarize"
    );
    assert!(record.repaired);
    // The client's own hint survived verbatim (ttl included)…
    assert_eq!(
        sent1["messages"][1]["content"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "1h"})
    );
    // …the client's own hint survived verbatim (ttl included), nothing was
    // placed alongside it (declined), and nothing leaked in: one hint in
    // the whole document, the client's.
    assert_eq!(cachemax::breakpoints::count(&sent1), 1);
    assert_eq!(hinted_messages(&sent1), vec![1]);
    assert!(!system_hinted(&sent1));
    assert_eq!(record.breakpoint_count, Some(1));
    // The drifted re-send forked the session store; the declined turn is
    // the fork's turn 0, and it claims none of the client's hints as ours.
    assert_eq!(
        rig.ledger
            .lock()
            .last_turn(2, "claude-3")
            .map(|t| t.breakpoints),
        Some(0),
        "a declined turn claims none of the client's hints as ours"
    );
    assert_eq!(
        rig.ledger
            .lock()
            .last_turn(1, "claude-3")
            .map(|t| t.breakpoints),
        Some(2),
        "turn 0's placements stay recorded"
    );
}

#[tokio::test]
async fn a_non_anthropic_backend_is_untouched_even_when_the_flag_is_set() {
    // Direct construction (the rig is Anthropic): the stage gates on the
    // adapter, so an OpenAI-dialect request under a mis-set flag forwards
    // untouched and records no count. Startup refuses the combination;
    // this is the belt to that braces.
    use cachemax::adapters::openai::OpenAiAdapter;

    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
    let sessions = Arc::new(SharedSessions::new());
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: sessions.clone(),
        ledger: Arc::new(SharedLedger::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
        repair: RepairMode::DryRun,
        manage_breakpoints: true,
        force_breakpoints: false,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    let rig = Rig {
        url: format!("http://{a}"),
        sessions,
        ledger: Arc::new(SharedLedger::new()),
    };

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    });
    let sent = body.to_string();
    send(&rig, &body).await;
    let record = last_record(&rig, 1).await;

    let upstream_saw = seen.lock().unwrap();
    assert_eq!(upstream_saw[0], sent.as_bytes());
    assert_eq!(record.breakpoint_count, None);
}
