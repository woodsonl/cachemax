//! `cachemax drift-matrix`: the per-class cost table. The runner measures
//! every fixture pair the same way — interleaved, unpooled — and the verdict
//! per class is plain: costs (canonical recovers cache), absorbed (the
//! endpoint normalizes the class away), or unmeasured (a gap, never a zero).

use cachemax::matrix::{self, run as run_matrix};
use cachemax::replay::{Backend, ExecuteConfig};

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(0)
        .build()
        .unwrap()
}

fn config(endpoint: String, samples: usize) -> ExecuteConfig {
    ExecuteConfig {
        endpoint,
        backend: Backend::OpenAi,
        api_key: None,
        samples,
    }
}

/// A byte-identity cache: the compact tool-arg form and the spaced form are
/// different prefixes — whichever form sent last, the other misses. The
/// tool-arg-reorder class must therefore read as COSTS, while whitespace in
/// the system (absent here) cannot be judged by this stub.
async fn byte_cache_stub() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
    let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = hits.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            async move {
                let n = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let text = String::from_utf8_lossy(&body);
                // Compact args = drifted wire form; spaced = canonical.
                let cached = if text.contains(r#"\"city\": \"Paris\""#) {
                    1400
                } else {
                    30
                };
                let reply = serde_json::json!({
                    "usage": {"prompt_tokens": 1500,
                              "prompt_tokens_details": {"cached_tokens": cached}},
                });
                let _ = n;
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
    (format!("http://{a}"), hits)
}

fn cachemax() -> Command {
    // tokio's Command: std's blocking output() would starve the runtime
    // that the axum stub lives on.
    Command::new(env!("CARGO_BIN_EXE_cachemax"))
}

#[tokio::test]
async fn the_matrix_costs_the_class_the_stub_says_it_costs() {
    let (endpoint, hits) = byte_cache_stub().await;
    let cfg = config(endpoint.clone(), 3);
    let classes: Vec<matrix::MatrixClass> = matrix::classes(Backend::OpenAi, "auto/fast")
        .into_iter()
        .filter(|c| c.name == "tool-arg-reorder")
        .collect();
    let results = run_matrix(&client(), &cfg, &classes).await;

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(r.a_drifted.sends, 3);
    assert_eq!(r.b_canonical.sends, 3);
    assert_eq!(
        r.verdict, "costs",
        "byte-identity cache punishes the reorder"
    );
    assert!(r.delta.unwrap() > 1000, "the whole drifted tail misses");

    let table = matrix::render(&endpoint, 3, &results);
    assert!(table.contains("tool-arg-reorder"));
    assert!(table.contains("costs"), "table: {table}");
    // 1 class × 2 forms × 3 samples: six sends, no more.
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 6);
}

#[tokio::test]
async fn a_normalizing_endpoint_absorbs_every_class() {
    // A router that reports the same cached count regardless of bytes: every
    // class reads absorbed. The matrix says repair has nothing to recover
    // there — the honest per-endpoint answer.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let reply = serde_json::json!({
                "usage": {"prompt_tokens": 1500,
                          "prompt_tokens_details": {"cached_tokens": 1200}},
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
    let classes = matrix::classes(Backend::OpenAi, "auto/fast");

    let results = run_matrix(&client(), &cfg, &classes).await;
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(r.verdict, "absorbed", "{}: normalized away", r.name);
    }
    let table = matrix::render(&a.to_string(), 2, &results);
    assert!(table.contains("absorbed"));
}

#[tokio::test]
async fn unknown_class_faults_with_the_valid_list() {
    let out = cachemax()
        .args([
            "drift-matrix",
            "--upstream-url",
            "http://127.0.0.1:1",
            "--classes",
            "nope",
            "--n",
            "1",
        ])
        .output()
        .await
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown drift class"), "got: {err}");
    assert!(
        err.contains("tool-arg-reorder"),
        "names a valid class: {err}"
    );
}

#[tokio::test]
async fn a_total_failure_faults_with_the_causes() {
    // Every send 500s: the matrix faults naming the status, the same
    // honesty contract as replay --execute.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Response::builder()
                .status(500)
                .body(Body::from("no"))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let out = cachemax()
        .args([
            "drift-matrix",
            "--upstream-url",
            &format!("http://{a}"),
            "--n",
            "2",
        ])
        .output()
        .await
        .unwrap();
    assert!(!out.status.success(), "all-unmeasured must fault");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("measured nothing"), "got: {err}");
    assert!(err.contains("HTTP 500"), "names the status: {err}");
}
