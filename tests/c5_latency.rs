//! C5 — latency budget: proxy-vs-direct TTFT overhead, with a p95 gate.
//!
//! The proxy must add ≤5 ms p95 of TTFT (target) and hard-fail CI above 6 ms
//! p95. Measurement is median-of-N over warm-pinned calls: we time the same
//! request against the upstream directly and through the proxy, and take the
//! per-call *delta*. The proxy's job is to forward first and observe on a side
//! copy, so the delta should be dominated by one extra loopback hop, not by
//! tokenization or JSON work.

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
use std::time::{Duration, Instant};

/// p95 target (ms) and the CI hard-fail gate (ms).
pub const P95_TARGET_MS: f64 = 5.0;
pub const P95_GATE_MS: f64 = 6.0;

/// p95 of a set of durations, in milliseconds.
pub fn p95_ms(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() as f64) * 0.95).ceil() as usize;
    samples[idx.saturating_sub(1).min(samples.len() - 1)]
}

/// Median of a set of durations, in milliseconds.
pub fn median_ms(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

const EVENT: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";

/// Upstream that emits one small event immediately, optionally with an injected
/// delay before the first byte.
async fn upstream(delay: Duration) -> String {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            let stream = async_stream::stream! {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(EVENT));
            };
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{a}")
}

async fn proxy_for(upstream: String) -> String {
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        ledger: Arc::new(cachemax::ledger::SharedLedger::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
        inject_usage: true,
        // The product default: the budget gates what ships, and drift
        // classification sits on the pre-forward path in dry-run.
        repair: cachemax::repair::RepairMode::DryRun,
        manage_breakpoints: false,
        force_breakpoints: false,
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    format!("http://{a}")
}

fn body() -> Bytes {
    Bytes::from_static(
        br#"{"model":"gpt-4o","messages":[{"role":"system","content":"sys"},{"role":"user","content":"Hi"}]}"#,
    )
}

/// Time to first byte of one request to `url`.
async fn ttft_ms(client: &reqwest::Client, url: &str) -> f64 {
    let start = Instant::now();
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .body(body())
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    let _first = s.next().await.unwrap().unwrap();
    start.elapsed().as_secs_f64() * 1000.0
}

/// Run N warm-pinned delta samples and return them in ms.
async fn deltas(direct: &str, via_proxy: &str, n: usize) -> Vec<f64> {
    let client = reqwest::Client::new();
    // Warm both paths so connection setup isn't counted.
    for _ in 0..5 {
        let _ = ttft_ms(&client, direct).await;
        let _ = ttft_ms(&client, via_proxy).await;
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let d = ttft_ms(&client, direct).await;
        let p = ttft_ms(&client, via_proxy).await;
        out.push(p - d);
    }
    out
}

#[tokio::test]
async fn clean_tree_is_within_the_p95_target() {
    if cfg!(debug_assertions) {
        // The budget is only meaningful optimized; debug builds inflate the
        // proxy's own work. CI certifies it in the dedicated release step.
        eprintln!("skipping latency budget in debug; run --release to gate");
        return;
    }
    let up = upstream(Duration::ZERO).await;
    let proxy_url = proxy_for(up.clone()).await;
    let samples = deltas(&up, &proxy_url, 60).await;
    let p95 = p95_ms(samples.clone());
    let med = median_ms(samples);
    eprintln!("clean: median {med:.3} ms, p95 {p95:.3} ms");
    assert!(
        p95 <= P95_GATE_MS,
        "p95 {p95:.3} ms exceeds the {P95_GATE_MS} ms gate (target {P95_TARGET_MS})"
    );
}

#[tokio::test]
async fn injected_delay_trips_the_gate() {
    // 20 ms is far above the 6 ms gate, so this holds in debug too: it proves
    // the gate detects a buffering hop regardless of build profile.
    // An added delay on the *proxy* path only (a second hop that sleeps) must
    // push the delta over the gate. This simulates a proxy that buffers.
    let up = upstream(Duration::ZERO).await;
    let proxy_url = proxy_for(up.clone()).await;
    // Build a deliberately-slow stand-in for a buffering proxy: an endpoint
    // that waits 20 ms before replying, then streams.
    let slow = upstream(Duration::from_millis(20)).await;
    let samples = deltas(&up, &slow, 20).await;
    let p95 = p95_ms(samples);
    assert!(
        p95 > P95_GATE_MS,
        "a 20 ms buffering hop must trip the {P95_GATE_MS} ms gate (got {p95:.3} ms)"
    );
    let _ = proxy_url; // the healthy proxy is the contrast case
}

#[test]
fn p95_and_median_helpers_are_sane() {
    assert_eq!(median_ms(vec![]), 0.0);
    assert_eq!(p95_ms(vec![]), 0.0);
    assert_eq!(median_ms(vec![1.0, 2.0, 3.0]), 2.0);
    assert_eq!(median_ms(vec![1.0, 2.0, 3.0, 4.0]), 2.5);
    // p95 of 100 samples 1..=100 is the 95th value.
    let s: Vec<f64> = (1..=100).map(|v| v as f64).collect();
    assert_eq!(p95_ms(s), 95.0);
}
