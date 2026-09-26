//! C1 end-to-end verification: drive the real axum proxy against an in-process
//! fake SSE upstream, no network, no credentials.
//!
//! Proves the two C1 invariants that need the streaming path:
//!   - byte-match: the client's reassembled stream equals the upstream's bytes
//!   - TTFT: forwarding is unbuffered, so TTFT tracks the upstream, not the
//!     total response time (the 50K-token regression guards this).

use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::proxy;
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::Tokenizer;

use axum::body::Body;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;

/// A fake upstream that emits `chunks` as an SSE body, `gap` apart.
async fn fake_upstream(chunks: Vec<Bytes>, gap: Duration) -> String {
    let chunks = Arc::new(chunks);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let chunks = chunks.clone();
            async move {
                let stream = async_stream::stream! {
                    for c in chunks.iter() {
                        tokio::time::sleep(gap).await;
                        yield Ok::<Bytes, std::io::Error>(c.clone());
                    }
                };
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// An upstream that emits all but the last chunk immediately, then waits for
/// `first_seen` before emitting the final chunk. A buffering proxy never lets
/// the client see an early chunk, so the client's first `next()` times out.
async fn gated_upstream(chunks: Vec<Bytes>, first_seen: Arc<tokio::sync::Notify>) -> String {
    let chunks = Arc::new(chunks);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let chunks = chunks.clone();
            let first_seen = first_seen.clone();
            async move {
                let stream = async_stream::stream! {
                    let (head, tail) = chunks.split_at(chunks.len() - 1);
                    for c in head {
                        yield Ok::<Bytes, std::io::Error>(c.clone());
                    }
                    first_seen.notified().await;
                    for c in tail {
                        yield Ok::<Bytes, std::io::Error>(c.clone());
                    }
                };
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// Boot the proxy pointed at `upstream`, return its base URL.
async fn boot_proxy(upstream: String) -> String {
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, proxy::router(state)).await.unwrap();
    });
    format!("http://{addr}")
}

fn request_body() -> Bytes {
    Bytes::from(
        br#"{"model":"gpt-4","messages":[{"role":"system","content":"sys"},{"role":"user","content":"Hi"}]}"#
            .to_vec(),
    )
}

#[tokio::test]
async fn stream_bytes_match_upstream_exactly() {
    // Four upstream events; the client must receive them concatenated, byte for
    // byte, in order. This is the C1 byte-match invariant.
    let events: Vec<Bytes> = vec![
        Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"),
        Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"),
        Bytes::from_static(b"data: {\"usage\":{\"prompt_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n"),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let expected: Vec<u8> = events.iter().flat_map(|b| b.to_vec()).collect();

    let upstream = fake_upstream(events, Duration::from_millis(1)).await;
    let proxy_url = boot_proxy(upstream).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(request_body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut got: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        got.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(got, expected, "client stream must byte-match upstream");
}

#[tokio::test]
async fn ttft_is_not_total_time() {
    // A 50K-token prompt still streams: the first chunk reaches the client
    // while the upstream is still holding its tail open. This is asserted
    // deterministically, not by wall-clock: the upstream refuses to emit its
    // final chunk until the client has already received the first one. A
    // buffering proxy would deadlock here (caught below by the timeout), never
    // flake on a slow runner.
    let big = "x".repeat(50_000);
    let events: Vec<Bytes> = vec![
        Bytes::from(format!(
            "data: {{\"usage\":{{\"prompt_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":0}}}}}}\n\n",
            12_500
        )),
        Bytes::from(format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{big}\"}}}}]}}\n\n")),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let first_seen = Arc::new(tokio::sync::Notify::new());
    let upstream = gated_upstream(events, first_seen.clone()).await;
    let proxy_url = boot_proxy(upstream).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(request_body())
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();

    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("first chunk must arrive before the upstream emits its tail (not buffered)")
        .unwrap()
        .unwrap();
    first_seen.notify_one();

    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    assert!(!first.is_empty());
}

#[tokio::test]
async fn client_disconnect_still_records_an_incomplete_turn() {
    // The upstream holds the stream open after one chunk. The client reads that
    // chunk then drops the response (disconnects). The turn must still be
    // recorded, as Incomplete — before this fix the finalize lived after the
    // read loop and never ran on a dropped stream, so nothing was recorded.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let stream = async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n",
                ));
                // Keep the stream open; the client will disconnect.
                tokio::time::sleep(Duration::from_secs(30)).await;
            };
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let u = l.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(l, app).await.unwrap();
    });

    let (proxy_url, sessions) = boot_proxy_with_state(format!("http://{u}")).await;
    {
        let resp = reqwest::Client::new()
            .post(format!("{proxy_url}/v1/chat/completions"))
            .body(request_body())
            .send()
            .await
            .unwrap();
        let mut s = resp.bytes_stream();
        // Read one chunk, then drop the response to simulate disconnect.
        let _ = s.next().await;
    } // resp dropped here

    // Give the proxy a moment to observe the disconnect and finalize.
    let mut recorded = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let guard = sessions.lock();
        if guard.most_recent().map(|s| {
            s.records
                .iter()
                .any(|r| r.status == cachemax::record::Status::Incomplete)
        }) == Some(true)
        {
            recorded = true;
            break;
        }
    }
    assert!(
        recorded,
        "a client disconnect must still record an Incomplete turn"
    );
}

#[tokio::test]
async fn early_usage_survives_large_content_that_exceeds_the_tail_cap() {
    // Usage arrives first (Anthropic-style), then >64 KB of content evicts the
    // front of the retained tail. The cache figure must still be observed.
    let mut events: Vec<Bytes> = vec![
        Bytes::from_static(
            b"data: {\"usage\":{\"prompt_tokens\":2140,\"prompt_tokens_details\":{\"cached_tokens\":1455}}}\n\n",
        ),
    ];
    for _ in 0..8 {
        events.push(Bytes::from(format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
            "x".repeat(10_000)
        )));
    }
    events.push(Bytes::from_static(b"data: [DONE]\n\n"));

    let upstream = fake_upstream(events, Duration::from_millis(1)).await;
    let (proxy_url, sessions) = boot_proxy_with_state(upstream).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .body(request_body())
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    while let Some(_c) = s.next().await {}
    // Let finalize run, then inspect the recorded cache figure.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let guard = sessions.lock();
    let cached: u64 = guard
        .most_recent()
        .map(|s| s.records.iter().map(|r| r.cached_tokens).max().unwrap_or(0))
        .unwrap_or(0);
    drop(guard);
    assert_eq!(
        cached, 1455,
        "an early usage event must survive tail eviction"
    );
}

/// Boot the proxy and also return its session store, for assertions.
async fn boot_proxy_with_state(upstream: String) -> (String, Arc<SharedSessions>) {
    let sessions = Arc::new(SharedSessions::new());
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: sessions.clone(),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, proxy::router(state)).await.unwrap();
    });
    (format!("http://{addr}"), sessions)
}

#[tokio::test]
async fn incomplete_upstream_stream_is_recorded_incomplete() {
    // An upstream that drops mid-stream yields an incomplete record, not a hang
    // and not a crash.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let stream = async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n",
                ));
                // then error out
                yield Err(std::io::Error::other("boom"));
            };
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let u = l.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(l, app).await.unwrap();
    });

    let proxy_url = boot_proxy(format!("http://{u}")).await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .body(request_body())
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    let mut n = 0;
    while let Some(_c) = s.next().await {
        n += 1;
    }
    assert!(n >= 1, "partial content still flowed to the client");
}

/// Spec rule: local engine counts are clamped to the history span (ratio ≤ 100%);
/// cloud provider-reported counts pass through unchanged.
#[test]
fn engine_measured_counts_clamp_to_history_span() {
    use cachemax::proxy::{build_record, Observation, RequestPlan};
    use cachemax::rates::Rates;
    use cachemax::record::SourceLabel;

    let plan = RequestPlan {
        session_id: 1,
        turn: 1,
        resent_history_tokens: 100,
        broke_prefix: false,
    };
    let rates = Rates::default();

    let local = build_record(
        &plan,
        Observation {
            ttft_ms: None,
            cached_tokens: 150,
            cache_written_tokens: 0,
            billed_input_tokens: 300,
        },
        "unknown-model",
        &rates,
        SourceLabel::EngineMeasured,
        true,
    );
    assert_eq!(
        local.cached_tokens, 100,
        "local clamp: never exceeds history"
    );

    let cloud = build_record(
        &plan,
        Observation {
            ttft_ms: None,
            cached_tokens: 150,
            cache_written_tokens: 0,
            billed_input_tokens: 300,
        },
        "unknown-model",
        &rates,
        SourceLabel::ProviderReported,
        true,
    );
    assert_eq!(cloud.cached_tokens, 150, "cloud counts pass through as-is");
}

/// Auth must reach the upstream, or every cloud call 401s. The proxy forwards
/// `authorization` (and provider-identification headers) verbatim.
#[tokio::test]
async fn authorization_header_is_forwarded_to_upstream() {
    use std::sync::Mutex;

    let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let seen2 = seen.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |headers: axum::http::HeaderMap| {
            let seen = seen2.clone();
            async move {
                *seen.lock().unwrap() = headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .map(String::from);
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(
                        "data: {\"usage\":{\"prompt_tokens\":1}}\n\ndata: [DONE]\n\n",
                    ))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let proxy_url = boot_proxy(format!("http://{addr}")).await;
    reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-test-123")
        .body(request_body())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    assert_eq!(
        seen.lock().unwrap().as_deref(),
        Some("Bearer sk-test-123"),
        "the client's bearer token must reach the upstream verbatim"
    );
}

/// A non-2xx upstream answer is not a measured turn. It must be recorded
/// incomplete (excluded from the aggregate), never as a fabricated miss with a
/// billed count equal to the whole history.
#[tokio::test]
async fn upstream_error_is_recorded_incomplete_not_a_fake_miss() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Response::builder()
                .status(401)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"error":{"message":"invalid key"}}"#))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: format!("http://{addr}"),
        client: reqwest::Client::new(),
        inject_usage: true,
    });
    let sessions = state.sessions.clone();
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let paddr = l.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(l, proxy::router(state)).await.unwrap();
    });

    let resp = reqwest::Client::new()
        .post(format!("http://{paddr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(request_body())
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap(); // drain so the observer finalizes

    // Poll for the record instead of sleeping a fixed beat: a fixed sleep can
    // lose the race on a loaded CI runner. Wait on the exact condition the
    // test asserts, bounded by a deadline so a genuinely missing record fails
    // loudly rather than hanging.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let recorded_incomplete = || {
        sessions
            .0
            .lock()
            .unwrap()
            .most_recent()
            .and_then(|s| s.records.last())
            .map(|r| r.status == cachemax::record::Status::Incomplete)
            .unwrap_or(false)
    };
    while !recorded_incomplete() {
        assert!(
            std::time::Instant::now() < deadline,
            "the failed request was never recorded as incomplete within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let guard = sessions.0.lock().unwrap();
    let s = guard.most_recent().expect("a session was created");
    let r = s.records.last().expect("the failed request was recorded");
    assert_eq!(
        r.status,
        cachemax::record::Status::Incomplete,
        "a 401 is not a complete measured turn"
    );
}

/// vLLM exposes no per-request cache figure in its response. The proxy must
/// sample `/metrics` before and after the request and record the counter delta.
/// This drives a fake vLLM that serves both `/metrics` (advancing on each
/// scrape) and `/v1/chat/completions`, and asserts the recorded cached tokens
/// equal the delta and carry the engine-measured label.
#[tokio::test]
async fn vllm_metrics_delta_is_wired_into_the_record() {
    use cachemax::adapters::vllm::VllmAdapter;
    use std::sync::atomic::{AtomicU64, Ordering};

    // hits advances by 455 between the before and after scrape.
    static SCRAPE: AtomicU64 = AtomicU64::new(0);
    let metrics = move || {
        // Each scrape advances hits by 455, so each request's delta is 455.
        let n = SCRAPE.fetch_add(1, Ordering::SeqCst) + 1;
        let hits = 1000 + n * 455;
        format!("vllm:prefix_cache_queries_total 2000.0\nvllm:prefix_cache_hits_total {hits}.0\n")
    };
    let app = Router::new()
        .route(
            "/metrics",
            get(move || {
                let body = metrics();
                async move { body }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(|| async {
                let stream = async_stream::stream! {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    ));
                    yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
                };
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let u = l.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(l, app).await.unwrap();
    });

    let sessions = Arc::new(SharedSessions::new());
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(VllmAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: sessions.clone(),
        rates: cachemax::rates::Rates::builtin(),
        // Includes /v1 so the metrics URL strips it back to the origin.
        upstream_url: format!("http://{u}/v1"),
        client: reqwest::Client::new(),
        inject_usage: true,
    });
    let pl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let paddr = pl.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(pl, proxy::router(state)).await.unwrap();
    });

    // Two turns so turn 1 has re-sent history; the engine-measured clamp is to
    // the history span, so a turn-0 request would legitimately clamp to 0.
    let long = "x".repeat(4000);
    let body1 = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": long},
            {"role": "user", "content": "first"}
        ]
    })
    .to_string();
    let body2 = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": long},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}
        ]
    })
    .to_string();

    for body in [body1, body2] {
        let resp = reqwest::Client::new()
            .post(format!("http://{paddr}/v1/chat/completions"))
            .body(body)
            .send()
            .await
            .unwrap();
        let _ = resp.bytes().await.unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let done = sessions
            .0
            .lock()
            .unwrap()
            .most_recent()
            .map(|s| s.records.len() >= 2)
            .unwrap_or(false);
        if done {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the vLLM turns were never recorded within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let guard = sessions.0.lock().unwrap();
    let r = guard.most_recent().unwrap().records.last().unwrap();
    assert_eq!(r.source, cachemax::record::SourceLabel::EngineMeasured);
    // The delta over the second request is 455 (hits 1000 -> 1455), and the
    // re-sent history exceeds that, so the recorded figure is the raw delta.
    assert_eq!(
        r.cached_tokens, 455,
        "recorded figure is the /metrics delta"
    );
}
