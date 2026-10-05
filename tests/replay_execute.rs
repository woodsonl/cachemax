//! `cachemax replay --execute`: the executor that drives recorded A/B pairs
//! against a real endpoint and reports what each form measurably costs.
//!
//! The printed JSONL pairs are the input; this drives them. The invariants
//! under test:
//!   - each form is sent `n` times, interleaved a/b, and the readings are the
//!     endpoint's own numbers;
//!   - the drifted form can read lower cached tokens than the canonical one
//!     (the cache delta the repair claim prices), and the executor recovers
//!     it — but never across different upstream instances;
//!   - a failed or unreadable send is an honest gap (`sends` short, median
//!     `None`), never a fabricated 0 and never a recovery claim;
//!   - a body whose usage carries no cache figure is unmeasured, not zero;
//!   - distinct upstream instances are counted by stable routing identity,
//!     not by volatile per-response headers;
//!   - the CLI guards the bill: planned-send count, `--yes` above the
//!     threshold, `--limit`, `--api-key-env`, and the `--n 0` floor.

use cachemax::replay::{execute_pair, render_report, Backend, ChainReport, ExecuteConfig};

use axum::body::Body;
use axum::extract::State;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::process::Command;

/// A stub endpoint whose cached-token reply depends on the body's form: the
/// canonical body (spaced tool-call arguments) reads high, the drifted body
/// (compact arguments) reads low — the cache penalty drift causes. It also
/// answers a different instance header every `rotate_every` requests, so a
/// router's multi-instance spread is reproducible.
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
    // the canonical body the spaced form.
    let cached = if is_canonical(&text) { 1455 } else { 20 };
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
        // Volatile headers that must NOT be read as instance identity.
        .header("date", format!("Mon, 05 Oct 2026 09:00:{:02} GMT", n % 60))
        .header("x-request-id", format!("req-{n}"))
        .body(Body::from(serde_json::to_vec(&reply).unwrap()))
        .unwrap()
}

fn client() -> reqwest::Client {
    // The driver under test pools no connections; mirror that here so the
    // probes measure the same transport behavior.
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

/// A recorded chain whose tool-call arguments carry spaced keys:
/// `replay_pair` drifts it to the compact reserialization, the canonical
/// form keeps the spaced bytes — the marker the stubs tell apart.
const SPACED_ARGS: &str = "{\"city\": \"Paris\", \"unit\": \"c\"}";

/// The wire body carries the canonical (spaced) argument bytes only when the
/// form under test is B; replay_pair's drifted reserialization is compact.
fn is_canonical(text: &str) -> bool {
    text.contains(r#"\"city\": \"Paris\""#)
}

fn chain_request() -> cachemax::ledger::ReplayRequest {
    cachemax::ledger::ReplayRequest {
        session_id: 1,
        turn: 1,
        model: "gpt-4o".into(),
        messages: json!([
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": SPACED_ARGS}}
            ]},
        ]),
        request_system: serde_json::Value::Null,
    }
}

#[tokio::test]
async fn the_executor_measures_both_forms_and_recovers_the_cache_delta() {
    let hits = Arc::new(AtomicUsize::new(0));
    let endpoint = stub_endpoint(hits.clone(), 0).await;
    let cfg = config(endpoint, 3);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;

    // Each form was sent n times and read the endpoint's own numbers.
    assert_eq!(out.a_drifted.sends, 3);
    assert_eq!(out.b_canonical.sends, 3);
    assert_eq!(out.a_drifted.cached_readings, vec![20, 20, 20]);
    assert_eq!(out.b_canonical.cached_readings, vec![1455, 1455, 1455]);
    assert_eq!(out.a_drifted.median_cached, Some(20));
    assert_eq!(out.b_canonical.max_cached, Some(1455));
    // Prompt tokens come from the endpoint too, not fabricated.
    assert_eq!(out.b_canonical.prompt_readings, vec![2140, 2140, 2140]);

    // The rendered table names the recovery, derived from the readings.
    let table = render_report(&out);
    assert!(
        table.contains("canonical form recovers 1435 cached tokens"),
        "the delta is the endpoint's, reported: {table}"
    );
    // Six sends total: 3 drifted + 3 canonical.
    assert_eq!(hits.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn the_two_forms_are_sent_interleaved_not_in_blocks() {
    // A×n then B×n over a warm connection would let form B read form A's
    // warmed prefix; the driver must alternate a, b, a, b, …
    let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = order.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let seen = seen.clone();
            async move {
                let text = String::from_utf8_lossy(&body);
                let form = if is_canonical(&text) { "b" } else { "a" };
                seen.lock().unwrap().push(form);
                let reply = json!({
                    "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 1}},
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
    let cfg = config(format!("http://{a}"), 3);
    execute_pair(&client(), &cfg, &chain_request()).await;

    assert_eq!(
        *order.lock().unwrap(),
        vec!["a", "b", "a", "b", "a", "b"],
        "samples alternate across forms"
    );
}

#[tokio::test]
async fn a_form_partly_failing_keeps_its_survivor_readings() {
    // One flaky send among n is the common real shape: the failed send is
    // dropped from the readings (never a 0), the survivors still measure.
    let hits = Arc::new(AtomicUsize::new(0));
    let drifted_hits = hits.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: Bytes| {
            let drifted_hits = drifted_hits.clone();
            async move {
                let text = String::from_utf8_lossy(&body);
                if !is_canonical(&text) {
                    // The drifted form's FIRST send fails; later ones succeed.
                    if drifted_hits.fetch_add(1, Ordering::SeqCst) == 0 {
                        return Response::builder()
                            .status(500)
                            .body(Body::from("upstream error"))
                            .unwrap();
                    }
                    let reply = json!({
                        "usage": {"prompt_tokens": 2140, "prompt_tokens_details": {"cached_tokens": 20}},
                    });
                    return Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&reply).unwrap()))
                        .unwrap();
                }
                let reply = json!({
                    "usage": {"prompt_tokens": 2140, "prompt_tokens_details": {"cached_tokens": 1455}},
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
    let cfg = config(format!("http://{a}"), 3);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(out.a_drifted.sends, 2, "only the failed send is dropped");
    assert_eq!(out.a_drifted.cached_readings, vec![20, 20]);
    assert_eq!(out.a_drifted.median_cached, Some(20));
    assert_eq!(out.a_drifted.failures.len(), 1);
    assert_eq!(out.b_canonical.sends, 3);
}

#[tokio::test]
async fn volatile_headers_do_not_inflate_the_instance_count() {
    // The stub varies Date and x-request-id on every reply from ONE instance.
    // The fingerprint must ignore them, so this reads as a single instance —
    // not firing the routing-lottery warning on a stable endpoint.
    let hits = Arc::new(AtomicUsize::new(0));
    let endpoint = stub_endpoint(hits.clone(), 0).await;
    let cfg = config(endpoint, 4);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(out.a_drifted.instances.len(), 1);
    assert_eq!(out.b_canonical.instances.len(), 1);
    let table = render_report(&out);
    assert!(
        !table.contains("more than one upstream instance"),
        "a stable endpoint must not be flagged as routed: {table}"
    );
}

#[tokio::test]
async fn a_routed_spread_is_visible_not_averaged_away() {
    // Two instances answer alternately. Over four samples of one form the
    // executor sees both fingerprints and says so — the exact condition that
    // made a single live reading a routing lottery.
    let hits = Arc::new(AtomicUsize::new(0));
    let endpoint = stub_endpoint(hits.clone(), 2).await;
    let cfg = config(endpoint, 4);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    let mut seen = out.a_drifted.instances.clone();
    seen.extend(out.b_canonical.instances.iter().cloned());
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 2, "both instances named: {seen:?}");

    let table = render_report(&out);
    assert!(
        table.contains("more than one upstream instance"),
        "the spread is called out: {table}"
    );
}

#[tokio::test]
async fn a_delta_across_two_instances_is_not_a_recovery_claim() {
    // Form A answered by instance 1, form B by instance 2: the "recovered
    // tokens" number would compare two different cache namespaces — routing
    // noise, not a repair effect. The table must say so, not claim.
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|body: Bytes| async move {
            let n = hits.fetch_add(1, Ordering::SeqCst);
            let text = String::from_utf8_lossy(&body);
            let cached = if is_canonical(&text) {
                1455
            } else {
                20
            };
            let reply = json!({
                "usage": {"prompt_tokens": 2140, "prompt_tokens_details": {"cached_tokens": cached}},
            });
            Response::builder()
                .header("content-type", "application/json")
                // The instance flips between the two forms' blocks: A always
                // sees inst-1, B always sees inst-2.
                .header("server", if n.is_multiple_of(2) { "inst-1" } else { "inst-2" })
                .body(Body::from(serde_json::to_vec(&reply).unwrap()))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let cfg = config(format!("http://{a}"), 2);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(out.a_drifted.instances.len(), 1, "A saw one instance");
    assert_eq!(out.b_canonical.instances.len(), 1, "B saw one instance");
    assert_ne!(
        out.a_drifted.instances[0], out.b_canonical.instances[0],
        "the two forms were answered by different instances"
    );
    let table = render_report(&out);
    assert!(
        table.contains("different upstream instances"),
        "the cross-instance delta is called out: {table}"
    );
    assert!(
        !table.contains("recovers"),
        "no recovery claim off a routing artifact: {table}"
    );
}

#[tokio::test]
async fn a_failed_send_is_unmeasured_never_a_zero_or_a_claim() {
    // The endpoint answers 500 for the drifted form's body (a rejected or
    // broken request) and 200 for the canonical one. The failed sends are
    // unmeasured: no reading, no median, no recovery claim built on a
    // fabricated zero baseline.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|body: Bytes| async move {
            let text = String::from_utf8_lossy(&body);
            if !is_canonical(&text) {
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
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(out.a_drifted.sends, 0, "both failed — no reading invented");
    assert!(out.a_drifted.cached_readings.is_empty());
    assert_eq!(out.a_drifted.median_cached, None, "unmeasured, not 0");
    assert!(!out.a_drifted.measured());
    assert_eq!(out.b_canonical.sends, 2, "the good form was measured");
    assert_eq!(out.b_canonical.cached_readings, vec![90, 90]);

    // The table shows the gap and makes NO recovery claim.
    let table = render_report(&out);
    assert!(table.contains('—'), "the unmeasured form reads as a dash");
    assert!(
        !table.contains("recovers"),
        "no recovery claim off a fabricated zero: {table}"
    );
    assert!(
        table.contains("not a measurement"),
        "the gap is stated: {table}"
    );
}

#[test]
fn no_recovery_claim_when_canonical_is_not_better() {
    // Both forms measured with equal medians: neither a recovery line nor a
    // "not a measurement" gap line — a valid measurement with no delta. A
    // regression printing "recovers 0" or a gap here would misreport.
    fn form(form: &'static str, cached: u64) -> cachemax::replay::FormSample {
        cachemax::replay::FormSample {
            form,
            sends: 2,
            cached_readings: vec![cached, cached],
            prompt_readings: vec![100, 100],
            median_cached: Some(cached),
            max_cached: Some(cached),
            instances: vec!["i".into()],
            failures: vec![],
        }
    }
    let r = ChainReport {
        session_id: 1,
        turn: 1,
        model: "m".into(),
        a_drifted: form("a_drifted", 1455),
        b_canonical: form("b_canonical", 1455),
    };
    let t = render_report(&r);
    assert!(!t.contains("recovers"), "equal medians: no claim: {t}");
    assert!(
        !t.contains("not a measurement"),
        "both measured: no gap line either: {t}"
    );
}

#[tokio::test]
async fn a_streamed_body_is_unmeasured_not_zero() {
    // An endpoint that streams (text/event-stream) with a 200 status: the
    // usage is inside SSE frames, which this command does not parse. The
    // reading is absent — never counted as 0 cached.
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(
                    "data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":999}}}\n\ndata: [DONE]\n\n",
                ))
                .unwrap()
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    let cfg = config(format!("http://{a}"), 2);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(out.a_drifted.sends, 0, "an SSE body is not a reading");
    assert_eq!(out.a_drifted.median_cached, None);
    assert_eq!(out.b_canonical.median_cached, None);
}

#[tokio::test]
async fn the_openai_path_is_version_normalized_like_serve() {
    // The endpoint already ends in /v1 (the documented form); the executor
    // must not double it into /v1/v1/... — the path the stub answers.
    let hits = Arc::new(AtomicUsize::new(0));
    let base = stub_endpoint(hits.clone(), 0).await;
    let cfg = config(format!("{base}/v1"), 1);
    let out = execute_pair(&client(), &cfg, &chain_request()).await;
    assert_eq!(
        out.a_drifted.sends, 1,
        "a /v1 endpoint resolves to one /v1/chat/completions"
    );
    assert_eq!(out.a_drifted.median_cached, Some(20));
}

// ---- CLI-level guards (the binary, not the library) ----

fn cachemax() -> Command {
    // tokio's Command: std's blocking output() would starve this test's
    // single-thread runtime — the axum stub answering the CLI lives on it.
    Command::new(env!("CARGO_BIN_EXE_cachemax"))
}

/// One real ledger chain, written through the library so the on-disk shape
/// is the proxy's own.
fn temp_ledger(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cachemax-replayx-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    {
        let mut ledger = cachemax::ledger::Ledger::on_disk(dir.clone()).unwrap();
        ledger.append(
            1,
            cachemax::ledger::CanonicalTurn {
                turn: 0,
                model: "gpt-4o".into(),
                request_messages: serde_json::json!([{"role": "user", "content": "hi"}]),
                request_system: serde_json::Value::Null,
                response_messages: vec![],
                prefix_hashes: vec![1, 2],
                breakpoints: 0,
            },
        );
    }
    dir
}

#[tokio::test]
async fn replay_execute_faults_when_no_send_was_measurable() {
    // An always-500 endpoint: the command must exit non-zero with the
    // measured-nothing fault — not exit 0 over a table of dashes.
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

    let dir = temp_ledger("fault");
    let out = cachemax()
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            &format!("http://{a}"),
            "--n",
            "2",
        ])
        .output()
        .await
        .unwrap();
    assert!(!out.status.success(), "zero measurements must fault");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("measured nothing"), "got: {err}");
    assert!(
        err.contains("HTTP 500"),
        "the cause names the status: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn replay_execute_asks_for_yes_above_the_send_threshold() {
    // 1 chain × 2 forms × 50 samples = 100 billable sends without --yes:
    // the gate must stop the run before any send.
    let dir = temp_ledger("gate");
    let out = cachemax()
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            "http://127.0.0.1:1", // gate fires before any connection attempt
            "--n",
            "50",
        ])
        .output()
        .await
        .unwrap();
    assert!(!out.status.success(), "unconfirmed bulk sends must fault");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--yes"), "got: {err}");
    assert!(err.contains("100"), "the planned total is named: {err}");

    // --limit scopes the run under the threshold: the gate passes (and the
    // failure becomes the unreachable endpoint, not the confirmation).
    let out = cachemax()
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            "http://127.0.0.1:1",
            "--n",
            "50",
            "--limit",
            "1",
            "--yes",
        ])
        .output()
        .await
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("--yes"),
        "confirmed run must not re-ask: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn n_zero_floors_to_one_sample_per_form() {
    let hits = Arc::new(AtomicUsize::new(0));
    let base = stub_endpoint(hits.clone(), 0).await;
    let dir = temp_ledger("n0");
    let out = cachemax()
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            &base,
            "--n",
            "0",
        ])
        .output()
        .await
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        hits.load(Ordering::SeqCst),
        2,
        "--n 0 floors to one sample per form"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn api_key_env_names_the_variable_the_key_is_read_from() {
    // The header-capturing stub proves the env plumbing end to end: the key
    // named by --api-key-env rides the request; an unset var sends nothing.
    let auth_seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = auth_seen.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |headers: axum::http::HeaderMap, _body: Bytes| {
            let seen = seen.clone();
            async move {
                seen.lock().unwrap().push(
                    headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_owned),
                );
                let reply = json!({
                    "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 5}},
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

    let dir = temp_ledger("key");
    let out = cachemax()
        .env("CACHEMAX_TEST_KEY", "sk-test-123")
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            &format!("http://{a}"),
            "--api-key-env",
            "CACHEMAX_TEST_KEY",
        ])
        .output()
        .await
        .unwrap();
    assert!(out.status.success(), "set key: {:?}", out.stderr);
    // Default --n is 3: three sends per form, each carrying the key.
    let expected: Vec<Option<String>> =
        std::iter::repeat_n(Some("Bearer sk-test-123".to_string()), 6).collect();
    assert_eq!(
        auth_seen.lock().unwrap().as_slice(),
        expected.as_slice(),
        "the named env var's key rode every form's requests"
    );

    // Var absent: no auth header, and the run still completes (a keyless
    // endpoint is a valid target).
    auth_seen.lock().unwrap().clear();
    let out = cachemax()
        .env_remove("CACHEMAX_TEST_KEY")
        .args([
            "replay",
            "--execute",
            "--ledger-dir",
            dir.to_str().unwrap(),
            "--upstream-url",
            &format!("http://{a}"),
            "--api-key-env",
            "CACHEMAX_TEST_KEY",
        ])
        .output()
        .await
        .unwrap();
    assert!(out.status.success(), "keyless: {:?}", out.stderr);
    assert!(
        auth_seen.lock().unwrap().iter().all(Option::is_none),
        "no auth header sent without a key"
    );
    std::fs::remove_dir_all(&dir).ok();
}
