//! C6 release verification: the replay A/B. `cachemax replay` must
//! reconstruct request pairs from a recorded ledger where the two
//! variants are semantically identical (the classifier reads them as a
//! clean match) yet byte-different exactly where a provider's cache
//! keys — the delta the repair claim prices.

use cachemax::adapters::openai::OpenAiAdapter;
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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A stub upstream that answers the canonical tool-call reply.
async fn stub_upstream(seen: Arc<Mutex<Vec<Vec<u8>>>>) -> String {
    let reply = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}",
                    },
                }],
            },
        }],
        "usage": {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 100}},
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

#[tokio::test]
async fn replay_pair_is_semantically_identical_and_byte_different() {
    // Record a canonical chain by driving the real proxy.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = stub_upstream(seen.clone()).await;
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
        repair: RepairMode::DryRun,
        manage_breakpoints: false,
        force_breakpoints: false,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    let url = format!("http://{a}");

    let _ = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You call tools."},
                    {"role": "user", "content": "Weather in Paris?"},
                ],
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();

    // The ledger is in-memory here; the replay path reads a directory. The
    // record below proves what the chain holds; the pair assertions work
    // on a reconstructed ReplayRequest with the same shape.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if ledger.lock().canonical_messages(1, "gpt-4o").is_some() {
            break;
        }
        assert!(Instant::now() < deadline, "the chain never formed");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let messages = ledger.lock().canonical_messages(1, "gpt-4o").unwrap();
    let request = cachemax::ledger::ReplayRequest {
        session_id: 1,
        turn: 1,
        model: "gpt-4o".into(),
        messages: serde_json::Value::Array(messages),
    };

    // The pair: B is the chain exactly; A is the drifted client form.
    let pair = cachemax::repair::replay_pair(&request);
    let a_drifted = &pair["a_drifted"];
    let b_canonical = &pair["b_canonical"];

    // Byte-different where it matters: the tool-call arguments are
    // key-sorted and interior whitespace runs collapsed.
    assert_ne!(a_drifted, b_canonical, "the drift must change bytes");
    // The chain is [system, user, assistant+tool_calls]; its tool-call
    // element is at index 2.
    let a_tool = &a_drifted["messages"][2]["tool_calls"];
    assert!(
        a_tool.is_array(),
        "chain shape: {:#}",
        b_canonical["messages"]
    );
    let a_args = a_tool[0]["function"]["arguments"].as_str().unwrap();
    assert_eq!(
        a_args, "{\"city\":\"Paris\",\"unit\":\"c\"}",
        "the drifted arguments are compact, keys sorted"
    );
    let b_args = b_canonical["messages"][2]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert_eq!(b_args, "{\"city\": \"Paris\", \"unit\": \"c\"}");

    // Semantically identical: the classifier reads A as an equivalent
    // re-serialization of B's whole chain — repairable drift, not a break.
    // (report_matches stays false: A is not byte-exact, that is the point.)
    let tok = Tokenizer::default_encoder().unwrap();
    let client: Vec<serde_json::Value> = a_drifted["messages"].as_array().unwrap().clone();
    let canonical: Vec<serde_json::Value> = b_canonical["messages"].as_array().unwrap().clone();
    let c = cachemax::repair::classify_turn(&client, Some(&canonical), false, &tok);
    assert!(!c.report_matches, "the drifted form is not byte-exact");
    assert_eq!(
        c.drift_kind,
        Some(cachemax::repair::DriftKind::ToolArgReserialization)
    );
    assert_eq!(c.canonical_offset, 0);
    assert_eq!(
        c.equivalent_run,
        canonical.len(),
        "the whole re-sent history aligns"
    );
    assert_eq!(c.semantic_break, None);
    // And repair-on turns A back into exactly B's bytes.
    let mut repaired = client.clone();
    let rw = cachemax::repair::apply_canonical(&mut repaired, &c, &canonical, &tok).unwrap();
    assert!(rw.elements_replaced > 0);
    assert_eq!(
        serde_json::Value::Array(repaired),
        serde_json::Value::Array(canonical),
        "the rewrite closes the byte gap"
    );
}

// debug helper (temp)
