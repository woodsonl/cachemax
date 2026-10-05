//! `cachemax replay --execute`: the executor that drives recorded A/B pairs
//! against a real endpoint and reports what each form measurably costs.
//!
//! The printed JSONL pairs are the input; this drives them. The invariants
//! under test:
//!   - each form is sent `n` times, and the readings are the endpoint's;
//!   - the drifted form can read lower cached tokens than the canonical one
//!     (the cache delta the repair claim prices), and the executor recovers
//!     it from the endpoint's own numbers;
//!   - a failed send is an honest gap (`sends` short), never a fabricated 0;
//!   - distinct upstream instances are counted, so a routing-lottery spread
//!     is visible rather than silently averaged away.

use cachemax::replay::{execute_pair, Backend, ChainReport, ExecuteConfig, FormSample};

use axum::body::Body;
use axum::extract::State;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A stub endpoint whose cached-token reply depends on the body's form: the
/// canonical body (tokens with spaces) reads high, the drifted body (compact
/// tokens) reads low — the cache penalty drift causes. It also answers a
/// different instance header every `rotate_every` requests, so a router's
/// multi-instance spread is reproducible.
async fn stub_endpoint(hits: Arc<AtomicUsize>, rotate_every: usize) -> String {
    let app = Router::new()
        .route("/v1/chat/completions", post(stub_handler))
        .with_state((hits, rotate_every));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    format!("http://{a}")
}

async fn stub_handler(
    State((hits, rotate_every)): State<(Arc<AtomicUsize>, usize)>,
    body: Bytes,
) -> Response {
    let n = hits.fetch_add(1, Ordering::SeqCst);
    let text = String::from_utf8_lossy(&body);
    // The drifted body carries the compact (no-space) tool-call arguments;
    // the canonical body carries the spaced form. Same parsed JSON —
    // different bytes, different cache.
    let cached = if text.contains(r#""{\"city\":\"Paris\""#) {
        20
    } else {
        1455
    };
    let reply = json!({
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 2140, "prompt_tokens_details": {"cached_tokens": cached}},
    });
    let instance = match rotate_every {
        0 => "one".to_string(),
        every => format!("inst-{}", (n / every) % 2),
    };
    Response::builder()
        .header("content-type", "application/json")
        .header("server", instance)
        .body(Body::from(serde_json::to_vec(&reply).unwrap()))
        .unwrap()
}

fn config(endpoint: String, samples: usize) -> ExecuteConfig {
    ExecuteConfig {
        endpoint,
        backend: Backend::OpenAi,
        api_key: None,
        samples,
        timeout: Duration::from_secs(30),
    }
}

fn report() -> ChainReport {
    ChainReport {
        session_id: 1,
        turn: 1,
        model: "gpt-4o".into(),
        a_drifted: FormSample::empty("a_drifted"),
        b_canonical: FormSample::empty("b_canonical"),
    }
}

#[tokio::test]
async fn the_executor_measures_both_forms_and_recovers_the_cache_delta() {
    let hits = Arc::new(AtomicUsize::new(0));
    let endpoint = stub_endpoint(hits.clone(), 0).await;
    let cfg = config(endpoint, 3);
    let client = reqwest::Client::builder()
        .timeout(cfg.timeout)
        .build()
        .unwrap();

    // The canonical body has spaced args; the drifted body is compact.
    let canonical = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"}}
            ]}
        ],
    });
    let drifted = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\":\"Paris\",\"unit\":\"c\"}"}}
            ]}
        ],
    });

    let out = execute_pair(&client, &cfg, &report(), &drifted, &canonical).await;

    // Each form was sent n times and read the endpoint's own numbers.
    assert_eq!(out.a_drifted.sends, 3);
    assert_eq!(out.b_canonical.sends, 3);
    assert_eq!(out.a_drifted.cached_readings, vec![20, 20, 20]);
    assert_eq!(out.b_canonical.cached_readings, vec![1455, 1455, 1455]);
    assert_eq!(out.a_drifted.median_cached, 20);
    assert_eq!(out.b_canonical.max_cached, 1455);
    // Prompt tokens come from the endpoint too, not fabricated.
    assert_eq!(out.b_canonical.prompt_readings, vec![2140, 2140, 2140]);
    // One instance answered both forms.
    assert_eq!(out.a_drifted.instances.len(), 1);
    assert_eq!(out.b_canonical.instances.len(), 1);

    // The rendered table names the recovery, derived from the readings.
    let table = cachemax::replay::render_report(&out);
    assert!(
        table.contains("canonical form recovers 1435 cached tokens"),
        "the delta is the endpoint's, reported: {table}"
    );
    // Six sends total: 3 drifted + 3 canonical.
    assert_eq!(hits.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn a_routed_spread_is_visible_not_averaged_away() {
    // Two instances answer alternately. Over four samples of one form the
    // executor sees both fingerprints and says so — the exact condition that
    // made a single live reading a routing lottery.
    let hits = Arc::new(AtomicUsize::new(0));
    let endpoint = stub_endpoint(hits.clone(), 2).await;
    let cfg = config(endpoint, 4);
    let client = reqwest::Client::builder()
        .timeout(cfg.timeout)
        .build()
        .unwrap();
    let canonical = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});
    let drifted = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});

    let out = execute_pair(&client, &cfg, &report(), &drifted, &canonical).await;
    let mut seen = out.a_drifted.instances.clone();
    seen.extend(out.b_canonical.instances.iter().cloned());
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 2, "both instances named: {seen:?}");

    let table = cachemax::replay::render_report(&out);
    assert!(
        table.contains("more than one upstream instance"),
        "the spread is called out: {table}"
    );
}

#[tokio::test]
async fn a_failed_send_is_an_honest_gap_not_a_zero() {
    // The endpoint answers 500 for the drifted form's body (a rejected or
    // broken request) and 200 for the canonical one. The failed sends are
    // omitted from the readings — never counted as 0 cached tokens, which
    // would fabricate a cache miss.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|body: Bytes| async move {
            let text = String::from_utf8_lossy(&body);
            if text.contains(r#""{\"city\":\"Paris\""#) {
                return Response::builder()
                    .status(500)
                    .body(Body::from("upstream error"))
                    .unwrap();
            }
            let reply = json!({
                "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}},
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
    let cfg = config(format!("http://{a}"), 2);
    let client = reqwest::Client::builder()
        .timeout(cfg.timeout)
        .build()
        .unwrap();

    let canonical = json!({"messages": [{"content": "ok"}]});
    // The drifted form carries the compact args the stub rejects.
    let drifted = json!({"messages": [{"tool_calls": [{"arguments": "{\"city\":\"Paris\"}"}]}]});

    let out = execute_pair(&client, &cfg, &report(), &drifted, &canonical).await;
    assert_eq!(out.a_drifted.sends, 0, "both failed — no reading invented");
    assert!(out.a_drifted.cached_readings.is_empty());
    assert_eq!(out.a_drifted.median_cached, 0);
    assert_eq!(out.b_canonical.sends, 2, "the good form was measured");
    assert_eq!(out.b_canonical.cached_readings, vec![90, 90]);
}
