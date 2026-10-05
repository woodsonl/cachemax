//! C2 verification: drift detection and dry-run annotation through the real
//! proxy. The batch's binding contract: dry-run NEVER mutates the request —
//! the upstream receives the client's messages verbatim — while every record
//! carries the drift claim (matches/kind/tokens-at-risk) for the dashboard
//! and the export.

use cachemax::adapters::openai::OpenAiAdapter;
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

/// A fake upstream that records every request body it receives and streams
/// a fixed assistant reply per call.
async fn recording_upstream(seen: Arc<Mutex<Vec<serde_json::Value>>>) -> String {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            async move {
                seen.lock()
                    .unwrap()
                    .push(serde_json::from_slice(&body).unwrap());
                let stream = async_stream::stream! {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                        b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
                    ));
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                        b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
                    ));
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n",
                    ));
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
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

async fn boot(upstream: String) -> (String, Arc<SharedSessions>) {
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
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    (format!("http://{a}"), sessions)
}

async fn send(proxy_url: &str, body: &serde_json::Value) {
    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();
}

/// Poll until the session holds `n` records, then return them.
async fn records(sessions: &Arc<SharedSessions>, n: usize) -> Vec<cachemax::record::Record> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let done = sessions
            .lock()
            .most_recent()
            .map(|s| s.records.len() >= n)
            .unwrap_or(false);
        if done {
            return sessions.lock().most_recent().unwrap().records.clone();
        }
        assert!(
            Instant::now() < deadline,
            "the proxy never recorded {n} turns within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn first_turn_body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ],
    })
}

/// The second turn, re-sent with whitespace drift in the first user message
/// (the re-serialized-history failure mode the dry-run exists to catch).
fn drifted_second_turn_body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi "},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "More"},
        ],
    })
}

#[tokio::test]
async fn dry_run_annotates_drift_and_never_mutates_the_request() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = recording_upstream(seen.clone()).await;
    let (proxy_url, sessions) = boot(upstream).await;

    let first = first_turn_body();
    let drifted = drifted_second_turn_body();
    send(&proxy_url, &first).await;
    send(&proxy_url, &drifted).await;

    let rs = records(&sessions, 2).await;

    // Turn 0: nothing canonical to extend yet — examined, first-turn.
    assert_eq!(rs[0].repair_mode, RepairMode::DryRun);
    assert_eq!(rs[0].matches_canonical, Some(false));
    assert_eq!(rs[0].drift_kind, None);
    assert_eq!(rs[0].canonicalized_tokens, 0);
    assert!(!rs[0].repaired);

    // Turn 1: the re-sent history drifted (whitespace) and the record says
    // so, quantified — but the request was never touched.
    assert_eq!(rs[1].repair_mode, RepairMode::DryRun);
    assert_eq!(rs[1].matches_canonical, Some(false));
    assert_eq!(rs[1].drift_kind, Some(DriftKind::TextNormalization));
    assert!(rs[1].canonicalized_tokens > 0, "the drift is quantified");
    assert!(!rs[1].repaired, "dry-run never rewrites");

    // The binding contract: the upstream saw the client's messages verbatim.
    // (Usage injection may add stream_options; it never touches messages.)
    let upstream_saw = seen.lock().unwrap();
    assert_eq!(upstream_saw.len(), 2);
    assert_eq!(
        upstream_saw[1]["messages"], drifted["messages"],
        "dry-run must forward the client's drifted messages untouched"
    );
    assert_ne!(
        upstream_saw[1]["messages"], first["messages"],
        "the drift really was present in what the upstream received"
    );
}

#[tokio::test]
async fn a_clean_continuation_reports_a_match() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = recording_upstream(seen.clone()).await;
    let (proxy_url, sessions) = boot(upstream).await;

    let first = first_turn_body();
    let clean = serde_json::json!({
        "model": "gpt-4o",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "More"},
        ],
    });
    send(&proxy_url, &first).await;
    send(&proxy_url, &clean).await;

    let rs = records(&sessions, 2).await;
    assert_eq!(rs[1].matches_canonical, Some(true));
    assert_eq!(rs[1].drift_kind, None);
    assert_eq!(rs[1].canonicalized_tokens, 0);
    assert!(!rs[1].repaired);
}

#[tokio::test]
async fn a_changed_system_prompt_is_surfaced_not_rewritten() {
    // A changed system message is a changed conversation prefix: the session
    // store (prefix-hash keyed from element 0) correctly starts a new
    // session, whose empty ledger reports first-turn. The safety property
    // the plan demands — never rewrite under a changed system prompt — holds
    // because there is no chain to rewrite against; the request itself
    // reaches the upstream untouched. (The SystemPromptChanged label fires
    // when flattened hashes collide but Values differ — shape-only system
    // drift; that path is unit-tested in repair.rs.)
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = recording_upstream(seen.clone()).await;
    let (proxy_url, sessions) = boot(upstream).await;

    let first = first_turn_body();
    let switched = serde_json::json!({
        "model": "gpt-4o",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are verbose."},
            {"role": "user", "content": "Hi"},
        ],
    });
    send(&proxy_url, &first).await;
    send(&proxy_url, &switched).await;

    // Two distinct sessions now: the original, and the re-based one.
    let deadline = Instant::now() + Duration::from_secs(5);
    let snapshot = loop {
        let done = {
            let guard = sessions.lock();
            (guard.len() == 2).then(|| guard.most_recent().unwrap().records.clone())
        };
        if let Some(records) = done {
            break records;
        }
        assert!(
            Instant::now() < deadline,
            "the system change never started its own session"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    let last = snapshot.last().unwrap();
    assert_eq!(last.matches_canonical, Some(false));
    assert_eq!(last.drift_kind, None);
    assert_eq!(
        last.canonicalized_tokens, 0,
        "nothing canonical to extend — no at-risk claim"
    );
    assert!(!last.repaired);

    // And the request still reached the upstream untouched.
    let upstream_saw = seen.lock().unwrap();
    assert_eq!(upstream_saw[1]["messages"], switched["messages"]);
}

#[tokio::test]
async fn mode_off_makes_no_claim_at_all() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = recording_upstream(seen.clone()).await;
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
        repair: RepairMode::Off,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    let proxy_url = format!("http://{a}");

    send(&proxy_url, &first_turn_body()).await;
    send(&proxy_url, &drifted_second_turn_body()).await;
    let rs = records(&sessions, 2).await;

    assert_eq!(rs[0].repair_mode, RepairMode::Off);
    assert_eq!(rs[0].matches_canonical, None, "off = unexamined, not false");
    assert_eq!(rs[1].matches_canonical, None);
    assert_eq!(rs[1].drift_kind, None);
}

#[tokio::test]
async fn the_export_carries_the_drift_fields() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let upstream = recording_upstream(seen.clone()).await;
    let (proxy_url, _sessions) = boot(upstream).await;

    send(&proxy_url, &first_turn_body()).await;
    send(&proxy_url, &drifted_second_turn_body()).await;

    let jsonl = reqwest::Client::new()
        .get(format!("{proxy_url}/api/export"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let lines: Vec<&str> = jsonl.lines().collect();
    assert_eq!(lines.len(), 2);
    let last: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(last["repair_mode"], "dry_run");
    assert_eq!(last["matches_canonical"], false);
    assert_eq!(last["drift_kind"], "text_normalization");
    assert!(last["canonicalized_tokens"].as_u64().unwrap() > 0);
    assert_eq!(last["repaired"], false);
    // Metrics only: the export never carries message content.
    assert!(!jsonl.contains("terse"), "no system prompt content");
    assert!(!jsonl.contains("Hello"), "no assistant content");
}
