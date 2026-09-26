//! C6 verification: the dashboard's data path and the served page.
//!
//! Boots the real proxy against an in-process upstream, drives one request, then
//! asserts `/api/state` renders the session, `/` serves the embedded page, and
//! `/api/export` returns the JSONL. Also checks the state-map and glyph rules
//! that the page depends on.

use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::dashboard::{self, TapeState};
use cachemax::proxy;
use cachemax::record::{Record, SourceLabel, Status};
use cachemax::sessions::SharedSessions;
use cachemax::tokenize::Tokenizer;

use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use std::sync::Arc;

async fn upstream() -> String {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let stream = async_stream::stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"));
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(
                    b"data: {\"usage\":{\"prompt_tokens\":2140,\"prompt_tokens_details\":{\"cached_tokens\":1455}}}\n\n"));
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
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

async fn boot(upstream: String) -> String {
    let state = Arc::new(proxy::AppState {
        adapter: Arc::new(OpenAiAdapter),
        tokenizer: Tokenizer::default_encoder().unwrap(),
        sessions: Arc::new(SharedSessions::new()),
        rates: cachemax::rates::Rates::builtin(),
        upstream_url: upstream,
        client: reqwest::Client::new(),
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, proxy::router(state)).await.unwrap() });
    format!("http://{a}")
}

#[tokio::test]
async fn dashboard_serves_page_state_and_export() {
    let up = upstream().await;
    let url = boot(up).await;
    let client = reqwest::Client::new();

    // Empty state first: live=false, no turns, banner condition.
    let empty: serde_json::Value = client
        .get(format!("{url}/api/state"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(empty["live"], false);
    assert_eq!(empty["turns"].as_array().unwrap().len(), 0);

    // Drive one request through the proxy.
    let body = br#"{"model":"gpt-4o","messages":[{"role":"system","content":"sys"},{"role":"user","content":"Hi"}]}"#;
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .body(Bytes::from_static(body))
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();

    // The page is served.
    let page = client.get(format!("{url}/")).send().await.unwrap();
    assert_eq!(page.status(), 200);
    let html = page.text().await.unwrap();
    assert!(html.contains("cachemax"), "page names the product");
    assert!(html.contains("Prefix tape"), "tape section present");
    assert!(html.contains("provider_reported") || html.contains("hero-prov")); // provenance slot

    // State now reflects the session.
    let state: serde_json::Value = client
        .get(format!("{url}/api/state"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(state["live"], true);
    assert_eq!(state["session_count"], 1);
    assert!(!state["turns"].as_array().unwrap().is_empty());
    assert_eq!(state["provenance"], "provider_reported");

    // Export returns JSONL for the session.
    let export = client
        .get(format!("{url}/api/export"))
        .send()
        .await
        .unwrap();
    assert_eq!(export.status(), 200);
    let jsonl = export.text().await.unwrap();
    assert!(!jsonl.trim().is_empty());
    let first: serde_json::Value = serde_json::from_str(jsonl.lines().next().unwrap()).unwrap();
    assert!(first["cached_tokens"].is_number());
    assert!(first.get("messages").is_none(), "export is metrics-only");
}

#[test]
fn every_tape_state_renders_distinctly_without_color() {
    // The tape must be legible with color removed: distinct glyph per state.
    let glyphs: Vec<&str> = [
        TapeState::Hit,
        TapeState::Resent,
        TapeState::Cold,
        TapeState::Miss,
        TapeState::Break,
        TapeState::Incomplete,
    ]
    .iter()
    .map(|s| s.glyph())
    .collect();
    let mut sorted = glyphs.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), glyphs.len());
}

#[test]
fn hash_level_mode_prints_no_fake_byte_detail() {
    let r = Record {
        session_id: 1,
        turn: 1,
        status: Status::Complete,
        source: SourceLabel::ProviderReported,
        ttft_ms: Some(10.0),
        cached_tokens: 1000,
        cache_written_tokens: 0,
        resent_history_tokens: 2000,
        billed_input_tokens: 2000,
        cost_usd: Some(0.01),
        cost_saved_usd: Some(0.005),
    };
    let v = dashboard::view(std::slice::from_ref(&r), true, 1);
    assert_eq!(v.tape_mode, "hash");
    // Hash-level cells are states, never byte offsets or hex.
    for row in &v.tape {
        for cell in &row.cells {
            assert!(matches!(
                cell,
                TapeState::Hit
                    | TapeState::Resent
                    | TapeState::Cold
                    | TapeState::Miss
                    | TapeState::Break
                    | TapeState::Incomplete
            ));
        }
    }
}

#[test]
fn no_ad_hoc_colors_in_embedded_page() {
    // Every color in the page must come from the DESIGN.md token variables;
    // the only hex literals allowed are the token definitions themselves.
    let html = dashboard::DASHBOARD_HTML;
    let hex_lines: Vec<&str> = html
        .lines()
        .filter(|l| l.contains('#') && l.contains(':') && !l.trim_start().starts_with("/*"))
        .filter(|l| l.trim_start().starts_with("--"))
        .collect();
    // Any hex outside a `--token:` definition is a violation.
    for line in html.lines() {
        if line.contains('#') {
            let trimmed = line.trim_start();
            let is_token_def = trimmed.starts_with("--");
            let is_comment = trimmed.starts_with("/*") || trimmed.starts_with("//");
            let is_doctype_or_id = !line.contains(':') || line.contains("color-scheme");
            assert!(
                is_token_def || is_comment || is_doctype_or_id || !line.contains('#'),
                "ad-hoc hex color outside a token: {line}"
            );
        }
    }
    let _ = hex_lines;
}
