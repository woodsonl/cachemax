//! `cachemax canary`: the alignment health check, driven end to end
//! against a stub upstream. Green when the repair claim, the cache hit,
//! and the invariant checks all hold; RED naming every failure otherwise.

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A stub that reports cached tokens above zero once the canary's warm
/// turn arrives (the second request), so the cache-collapse check has a
/// real signal to verify.
async fn warm_stub() -> String {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let n = hits.fetch_add(1, Ordering::SeqCst);
            async move {
                let _ = body;
                let cached = if n >= 1 { 120 } else { 0 };
                let reply = serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": "CANARY 7741, recorded."}}],
                    "usage": {"prompt_tokens": 300,
                              "prompt_tokens_details": {"cached_tokens": cached}},
                });
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&reply).unwrap()))
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
async fn a_green_canary_verifies_repair_cache_and_invariants() {
    let upstream = warm_stub().await;
    let report = cachemax::canary::run(upstream, None).await;
    assert!(
        report.failures.is_empty(),
        "green canary, got: {:?}",
        report.failures
    );
    assert!(report.repaired_turn, "the drifted turn was rewritten");
    assert_eq!(report.warm_cached, Some(120));
    assert_eq!(report.violations, 0);
    let line = cachemax::canary::render(&report);
    assert!(line.contains("canary green"), "{line}");
    assert!(line.contains("120"), "{line}");
}

#[tokio::test]
async fn a_collapsing_cache_turns_the_canary_red() {
    // The stub never reports cached tokens: the warm turn reads zero and
    // the canary says so instead of claiming health.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let reply = serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "CANARY 7741, recorded."}}],
                "usage": {"prompt_tokens": 300,
                          "prompt_tokens_details": {"cached_tokens": 0}},
            });
            Response::builder()
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&reply).unwrap()))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let report = cachemax::canary::run(format!("http://{a}"), None).await;
    assert!(!report.failures.is_empty());
    let line = cachemax::canary::render(&report);
    assert!(line.contains("canary RED"), "{line}");
    assert!(line.contains("cache collapse"), "{line}");
}

#[tokio::test]
async fn a_space_free_reply_still_drifts_deterministically() {
    // The drift walk goes backward through the recorded chain: a reply
    // with no spaces falls back to the canary's own spaced request
    // elements, so the run stays deterministic — green on a healthy
    // endpoint — instead of silently sending a clean continuation (the
    // model obeying "one line, no commentary" must never page anyone).
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let reply = serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 300,
                          "prompt_tokens_details": {"cached_tokens": 100}},
            });
            Response::builder()
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&reply).unwrap()))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let report = cachemax::canary::run(format!("http://{a}"), None).await;
    assert!(
        report.failures.is_empty(),
        "space-free reply: deterministic drift via the request elements, got {:?}",
        report.failures
    );
    assert!(report.repaired_turn);
}
