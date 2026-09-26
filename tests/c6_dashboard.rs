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

/// A complete cloud record with a cost, for the state-legendity tests.
fn rec(turn: u32, cached: u64, history: u64) -> Record {
    Record {
        session_id: 1,
        turn,
        status: Status::Complete,
        source: SourceLabel::ProviderReported,
        ttft_ms: Some(120.0),
        cached_tokens: cached,
        cache_written_tokens: 0,
        resent_history_tokens: history,
        billed_input_tokens: history + 200,
        broke_prefix: false,
        cost_usd: Some(0.01),
        cost_saved_usd: Some(0.005),
    }
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
        broke_prefix: false,
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

// --- T2: interaction states ---

#[test]
fn t2_empty_state_has_no_rows_and_dash_convention() {
    let v = dashboard::view(&[], false, 0);
    assert_eq!(v.turns.len(), 0);
    assert_eq!(v.hit_rate, "—", "empty state is unexposed, never 0");
    assert_eq!(v.cost_saved, "—");
    assert_eq!(v.session_id, 0);
}

#[test]
fn t2_partial_state_excludes_incomplete_from_cumulative() {
    let mut inc = rec(2, 0, 0);
    inc.status = Status::Incomplete;
    let rs = vec![rec(1, 1000, 2000), inc];
    let v = dashboard::view(&rs, true, 1);
    assert_eq!(v.incomplete_count, 1);
    // cumulative is 1000/2000 = 50%, the incomplete row contributes nothing
    assert_eq!(v.cumulative.cached_over_history, "1,000 / 2,000");
    assert_eq!(v.hit_rate, "50%");
}

#[test]
fn t2_incomplete_turn_is_excluded_from_cost_and_billed_too() {
    // The page states "Complete records only"; cost and billed must honor it.
    let mut inc = rec(2, 0, 0);
    inc.status = Status::Incomplete;
    inc.billed_input_tokens = 9_999;
    inc.cost_usd = Some(9.99);
    inc.cost_saved_usd = Some(9.99);
    let rs = vec![rec(1, 1000, 2000), inc];
    let v = dashboard::view(&rs, true, 2);
    assert!(
        !v.billed_input.contains("9,999"),
        "incomplete billed tokens must be excluded: {}",
        v.billed_input
    );
    assert!(
        !v.cost_saved.contains("9.99"),
        "incomplete cost must be excluded: {}",
        v.cost_saved
    );
}

#[test]
fn t2_page_draws_reset_banner_and_session_break() {
    let html = dashboard::DASHBOARD_HTML;
    assert!(html.contains("Metrics reset"), "must draw the reset banner");
    assert!(
        html.contains("Session break"),
        "must draw the session break"
    );
}

#[test]
fn t2_hero_carries_the_incomplete_badge() {
    let html = dashboard::DASHBOARD_HTML;
    assert!(
        html.contains("hero-badge"),
        "the hero must have a badge slot"
    );
    assert!(
        html.contains("⚠ ") && html.contains("incomplete"),
        "the hero badge reads ⚠ N incomplete"
    );
}

// --- T3: cloud journey liveness + payoff ---

#[test]
fn t3_cold_turn_shows_activity_and_warm_turn_shows_payoff() {
    // Turn 0: activity (a cold tape run and a row), no hit number yet.
    let t0 = rec(0, 0, 0);
    let v0 = dashboard::view(std::slice::from_ref(&t0), true, 1);
    assert_eq!(v0.turns.len(), 1, "turn 0 is an activity row, not blank");
    assert!(
        v0.tape[0].cells.iter().all(|c| *c == TapeState::Cold),
        "turn 0 shows a cold tape run"
    );
    assert_eq!(v0.turns[0].hit, "—", "no hit rate on the cold turn");

    // Turn 1: the payoff — hit rate + the provider_reported label.
    let rs = vec![t0, rec(1, 1020, 1550)];
    let v1 = dashboard::view(&rs, true, 1);
    assert_eq!(v1.provenance, "provider_reported");
    assert_eq!(v1.turns[1].hit, "66%", "first warm turn shows its hit rate");
}

// --- T4: responsive + a11y contract ---

#[test]
fn t4_page_covers_the_responsive_and_a11y_contract() {
    let html = dashboard::DASHBOARD_HTML;
    // 1024px floor, stack below.
    assert!(
        html.contains("@media (max-width:1024px)"),
        "must stack below 1024px"
    );
    // Keyboard-reachable rows and cells.
    assert!(
        html.contains("tabindex=\"0\""),
        "rows/cells must be focusable"
    );
    // Focus ring always visible.
    assert!(html.contains(":focus-visible"), "must define focus-visible");
    assert!(
        html.contains("outline:2px solid var(--accent)"),
        "focus ring in accent"
    );
    // 44px targets on controls.
    assert!(
        html.contains("min-height:44px"),
        "controls need a 44px target"
    );
    assert!(
        html.contains("min-width:44px"),
        "controls need a 44px target"
    );
    // Tabular numerals on all figures.
    assert!(
        html.contains("font-variant-numeric:tabular-nums"),
        "figures must use tabular numerals"
    );
    // Light + dark via prefers-color-scheme.
    assert!(
        html.contains("prefers-color-scheme"),
        "must follow the OS theme"
    );
    // Reduced motion respected.
    assert!(
        html.contains("prefers-reduced-motion"),
        "must respect reduced motion"
    );
}

#[test]
fn provenance_is_the_dominant_source_not_the_first_record() {
    // A cloud session whose first response omitted the provider's cache field
    // must not be mislabeled local by that single turn.
    let mut untruth = rec(1, 0, 0);
    untruth.source = SourceLabel::NoCacheTruth;
    let rs = vec![untruth, rec(2, 1000, 2000), rec(3, 1100, 2100)];
    assert_eq!(
        dashboard::provenance(&rs),
        SourceLabel::ProviderReported,
        "the majority source wins"
    );
}
