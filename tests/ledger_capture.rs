//! C1 ledger-capture verification: drive the real axum proxy against an
//! in-process fake upstream and prove the canonical ledger remembers the
//! request exactly as the upstream received it and the response exactly as
//! the provider returned it.
//!
//! These are the invariants repair will stand on:
//!   - as-sent fidelity: the ledger's `request_messages` equal the messages
//!     parsed out of the body that reached the upstream (post-injection);
//!   - as-received fidelity: a streamed response reassembles to the same
//!     assistant message object a non-streaming body would have carried;
//!   - completeness gate: incomplete turns never enter the chain.

use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::ledger::{CanonicalTurn, SharedLedger};
use cachemax::proxy;
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::Tokenizer;

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A fake upstream that records the exact request body it received and
/// streams a fixed assistant reply.
async fn recording_stream_upstream(
    seen: Arc<Mutex<Vec<serde_json::Value>>>,
    events: Vec<Bytes>,
) -> String {
    let events = Arc::new(events);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            let events = events.clone();
            async move {
                seen.lock()
                    .unwrap()
                    .push(serde_json::from_slice(&body).unwrap());
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

/// A fake upstream that answers one non-streaming JSON document.
async fn json_upstream(reply: serde_json::Value) -> String {
    let reply = Arc::new(reply);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let reply = reply.clone();
            async move {
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

/// Boot the proxy against `upstream` with an on-disk ledger in `dir`, and
/// return (base url, ledger handle).
async fn boot(upstream: String, ledger: Arc<SharedLedger>) -> String {
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        ledger,
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    format!("http://{a}")
}

fn client_body(stream: bool) -> String {
    serde_json::json!({
        "model": "gpt-4o",
        "stream": stream,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ],
    })
    .to_string()
}

/// Poll the ledger until `cond` holds, bounded by a deadline so a genuinely
/// missing turn fails loudly rather than hanging (no fixed sleeps to lose
/// races on a loaded runner).
async fn await_ledger<F: Fn(&cachemax::ledger::Ledger) -> bool>(ledger: &SharedLedger, cond: F) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !cond(&ledger.lock()) {
        assert!(
            std::time::Instant::now() < deadline,
            "the ledger never satisfied the condition within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn a_streamed_turn_is_remembered_exactly_as_sent_and_received() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let events: Vec<Bytes> = vec![
        Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n"),
        Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n"),
        Bytes::from_static(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n"),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let upstream = recording_stream_upstream(seen.clone(), events).await;
    let ledger = Arc::new(SharedLedger::new());
    let proxy_url = boot(upstream, ledger.clone()).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(client_body(true))
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();

    // As-sent: the ledger's messages equal what the upstream parsed out of
    // the body it received (which includes the injected stream_options).
    await_ledger(&ledger, |l| l.last_turn(1, "gpt-4o").is_some()).await;
    let upstream_saw = seen.lock().unwrap()[0].clone();
    assert_eq!(
        upstream_saw["stream_options"]["include_usage"], true,
        "injection applied before forwarding"
    );
    let turn: CanonicalTurn = {
        let guard = ledger.lock();
        guard.last_turn(1, "gpt-4o").unwrap().clone()
    };
    assert_eq!(turn.request_messages, upstream_saw["messages"]);
    assert_eq!(turn.turn, 0);

    // As-received: the streamed deltas reassembled to the message object a
    // non-streaming response would have carried.
    assert_eq!(
        serde_json::json!(turn.response_messages),
        serde_json::json!([{"role": "assistant", "content": "Hello"}])
    );

    // The canonical chain is the request extended by the response.
    let chain = ledger.lock().canonical_messages(1, "gpt-4o").unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(
        chain[2],
        serde_json::json!({"role": "assistant", "content": "Hello"})
    );
}

#[tokio::test]
async fn a_non_streaming_turn_remembered_verbatim() {
    let upstream = json_upstream(serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
        "usage": {"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}},
    }))
    .await;
    let ledger = Arc::new(SharedLedger::new());
    let proxy_url = boot(upstream, ledger.clone()).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(client_body(false))
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();
    await_ledger(&ledger, |l| l.last_turn(1, "gpt-4o").is_some()).await;

    let turn: CanonicalTurn = {
        let guard = ledger.lock();
        guard.last_turn(1, "gpt-4o").unwrap().clone()
    };
    assert_eq!(
        serde_json::json!(turn.response_messages),
        serde_json::json!([{"role": "assistant", "content": "Hi there"}])
    );
}

#[tokio::test]
async fn an_incomplete_turn_never_enters_the_chain() {
    // A 500 upstream: the turn is recorded Incomplete in the metrics store
    // but the canonical chain must stay empty — the client never received a
    // complete assistant message, so there is nothing canonical to remember.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Response::builder()
                .status(500)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"error":"boom"}"#))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });

    let ledger = Arc::new(SharedLedger::new());
    let proxy_url = boot(format!("http://{a}"), ledger.clone()).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(client_body(false))
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();

    // Bounded wait for the finalize to run, then assert absence.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        ledger.lock().last_turn(1, "gpt-4o").is_none(),
        "a failed turn must not become canonical"
    );
}

#[tokio::test]
async fn the_disk_ledger_persists_turns_across_a_restart() {
    let dir = std::env::temp_dir().join(format!(
        "cachemax-c1-disk-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    // Boot with a disk-backed ledger, run one streamed turn.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let events: Vec<Bytes> = vec![
        Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}\n\n"),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let upstream = recording_stream_upstream(seen.clone(), events).await;
    let ledger = Arc::new(SharedLedger::on_disk(dir.clone()).unwrap());
    let proxy_url = boot(upstream, ledger.clone()).await;
    let _ = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(client_body(true))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    await_ledger(&ledger, |l| l.last_turn(1, "gpt-4o").is_some()).await;

    // One line on disk for the one complete turn.
    let text = std::fs::read_to_string(dir.join("1.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 1);

    // "Restart": a fresh ledger over the same directory recovers the chain.
    let reloaded = SharedLedger::on_disk(dir.clone()).unwrap();
    let chain = reloaded.lock().canonical_messages(1, "gpt-4o").unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(
        chain[2],
        serde_json::json!({"role": "assistant", "content": "ok"})
    );
    std::fs::remove_dir_all(&dir).ok();
}
