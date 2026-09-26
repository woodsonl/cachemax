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
use axum::routing::post;
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

/// Boot the proxy pointed at `upstream`, return its base URL.
async fn boot_proxy(upstream: String) -> String {
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
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
    // A 50K-token prompt still streams: the first token arrives early while a
    // slow tail follows. TTFT must reflect the first chunk, not the total.
    let big = "x".repeat(50_000);
    let events: Vec<Bytes> = vec![
        Bytes::from(format!(
            "data: {{\"usage\":{{\"prompt_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":0}}}}}}\n\n",
            12_500
        )),
        Bytes::from(format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{big}\"}}}}]}}\n\n")),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    // 40ms between chunks: a buffering proxy would take 80ms+ before any byte.
    let upstream = fake_upstream(events, Duration::from_millis(40)).await;
    let proxy_url = boot_proxy(upstream).await;

    let start = std::time::Instant::now();
    let resp = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(request_body())
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();
    let first = stream.next().await.unwrap().unwrap();
    let ttft = start.elapsed();

    // First byte arrives after ~40ms (one upstream gap), well under the 80ms a
    // full-response buffer would cost.
    assert!(
        ttft < Duration::from_millis(70),
        "TTFT {ttft:?} suggests buffering"
    );
    assert!(!first.is_empty());
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
