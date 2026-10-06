//! `cachemax canary`: a scheduled, self-contained health check for the
//! whole alignment story. It boots a private in-process proxy against a
//! real endpoint and drives a three-request conversation through it —
//! establish, drifted re-send (repaired), exact re-send (warm) — verifying
//! what the product claims actually happened:
//!
//!   - every turn completed (a fault exit is the loudest signal there is);
//!   - the drifted turn WAS repaired (the record says so, not hope);
//!   - the warm turn actually hit cache (the provider reported cached
//!     tokens above zero — the collapse check);
//!   - zero runtime invariant violations across the run.
//!
//! One invocation, then exit: cron/launchd schedules it; a non-zero exit
//! with the fault contract is the alert. No daemon, no state.

use crate::adapters::openai::OpenAiAdapter;
use crate::proxy;
use crate::repair::RepairMode;
use crate::sessions::SharedSessions;
use crate::tokenize::Tokenizer;

/// The verdict line for the log or the alert.
pub struct CanaryReport {
    pub repaired_turn: bool,
    pub warm_cached: Option<u64>,
    pub violations: u64,
    pub failures: Vec<String>,
}

/// Run the canary conversation against `upstream`, using `api_key` for
/// authorization on each request. Returns the report; the caller decides
/// exit semantics.
pub async fn run(upstream: String, api_key: Option<String>) -> CanaryReport {
    let tokenizer = Tokenizer::default_encoder().expect("bundled tokenizer");
    let state = std::sync::Arc::new(proxy::AppState {
        adapter: std::sync::Arc::new(OpenAiAdapter),
        tokenizer,
        sessions: std::sync::Arc::new(SharedSessions::new()),
        ledger: std::sync::Arc::new(crate::ledger::SharedLedger::new()),
        rates: crate::rates::Rates::builtin(),
        upstream_url: upstream.clone(),
        client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("client"),
        inject_usage: true,
        repair: RepairMode::On,
        manage_breakpoints: false,
        force_breakpoints: false,
    });
    let app = proxy::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let proxy_url = format!("http://{addr}");

    let violations_before = crate::invariants::total();
    let mut failures: Vec<String> = Vec::new();

    let system = "You are the canary recorder. Reply with the reference code CANARY-7741 and nothing else. The lab protocol requires the code in every reply, quoted exactly, in one line, with no commentary.";
    let send = |body: serde_json::Value| {
        let proxy_url = proxy_url.clone();
        let api_key = api_key.clone();
        async move {
            let mut req = reqwest::Client::new()
                .post(format!("{proxy_url}/v1/chat/completions"))
                .header("content-type", "application/json")
                .body(body.to_string());
            if let Some(key) = api_key {
                req = req.header("authorization", format!("Bearer {key}"));
            }
            req.send().await
        }
    };

    // Turn 0: establish the chain.
    let t0 = serde_json::json!({
        "model": "auto/fast",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Confirm the reference code."},
        ],
    });
    let turn0 = match send(t0).await {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
        Ok(r) => {
            failures.push(format!("turn 0: HTTP {}", r.status()));
            None
        }
        Err(e) => {
            failures.push(format!("turn 0: {e}"));
            None
        }
    };
    let cached0 = turn0
        .as_ref()
        .and_then(|b| b.pointer("/usage/prompt_tokens_details/cached_tokens"))
        .and_then(|n| n.as_u64());

    // Turn 1: re-send the RECORDED chain (whatever the provider actually
    // replied — the canary assumes nothing about content) with whitespace
    // drift in its last element. Drift earlier in the history would break
    // prefix-hash continuity and fork the session — the affinity feature's
    // territory, not the repair path's. The drifted element is repaired on
    // the way out, so the provider still sees the canonical prefix.
    let chain = {
        let ledger = state.ledger.lock();
        ledger.canonical_messages(1, "auto/fast")
    };
    let mut messages: Vec<serde_json::Value> = match chain {
        Some(c) if !c.is_empty() => c,
        _ => {
            failures.push("turn 0 never recorded a canonical chain".to_string());
            Vec::new()
        }
    };
    // Drift the trailing-est element whose content carries a space —
    // walking backward keeps earlier prefix hashes intact (session
    // continuity), and requiring a space makes the drift deterministic
    // instead of depending on the model's reply shape. A space-free
    // conversation fails loudly rather than silently not drifting.
    let mut drifted_any = false;
    for element in messages.iter_mut().rev() {
        if let Some(text) = element.get("content").and_then(|c| c.as_str()) {
            if text.contains(' ') {
                let drifted = text.replacen(' ', "  ", 1);
                element["content"] = serde_json::Value::String(drifted);
                drifted_any = true;
                break;
            }
        }
    }
    if !drifted_any {
        failures
            .push("no driftable element: the recorded chain carries no spaced content".to_string());
    }
    messages.push(serde_json::json!({"role": "user", "content": "Again."}));
    let t1 = serde_json::json!({
        "model": "auto/fast",
        "messages": messages,
    });
    let turn1 = match send(t1).await {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
        Ok(r) => {
            failures.push(format!("turn 1: HTTP {}", r.status()));
            None
        }
        Err(e) => {
            failures.push(format!("turn 1: {e}"));
            None
        }
    };
    let cached1 = turn1
        .as_ref()
        .and_then(|b| b.pointer("/usage/prompt_tokens_details/cached_tokens"))
        .and_then(|n| n.as_u64());

    // The repair claim: the drifted turn's record says it was rewritten.
    let repaired_turn = {
        let guard = state.sessions.lock();
        guard
            .most_recent()
            .and_then(|s| s.records.get(1))
            .is_some_and(|r| r.repaired)
    };
    if !repaired_turn {
        failures.push("drifted turn was not repaired".to_string());
    }

    // Turn 2: exact re-send of the recorded chain — the decoupled warm
    // check. A cache hit here proves the prefix survived on its own bytes,
    // independent of whether repair mattered for turn 1.
    let chain2 = {
        let ledger = state.ledger.lock();
        ledger.canonical_messages(1, "auto/fast")
    };
    let mut messages2 = chain2.unwrap_or_default();
    messages2.push(serde_json::json!({"role": "user", "content": "Once more."}));
    let t2 = serde_json::json!({
        "model": "auto/fast",
        "messages": messages2,
    });
    let turn2 = match send(t2).await {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
        Ok(r) => {
            failures.push(format!("turn 2: HTTP {}", r.status()));
            None
        }
        Err(e) => {
            failures.push(format!("turn 2: {e}"));
            None
        }
    };
    let cached2 = turn2
        .as_ref()
        .and_then(|b| b.pointer("/usage/prompt_tokens_details/cached_tokens"))
        .and_then(|n| n.as_u64());
    // The cache claim: the provider served cached tokens on the exact
    // re-send (a cold canary session may read zero on turn 0; the exact
    // warm turn is the check).
    match cached2 {
        Some(n) if n > 0 => {}
        Some(_) => failures.push("warm turn read 0 cached tokens (cache collapse)".to_string()),
        None => failures.push("warm turn carried no readable cache figure".to_string()),
    }

    let violations = crate::invariants::total() - violations_before;
    if violations > 0 {
        failures.push(format!(
            "{violations} invariant violation(s) during the run"
        ));
    }

    let _ = (cached0, cached1);
    CanaryReport {
        repaired_turn,
        warm_cached: cached2,
        violations,
        failures,
    }
}

/// The one-line verdict: green, or the failure list. For the log and the
/// alert body alike.
pub fn render(r: &CanaryReport) -> String {
    if r.failures.is_empty() {
        format!(
            "canary green · repaired ✓ · warm cached {} tk · 0 violations",
            r.warm_cached
                .map_or_else(|| "—".to_string(), |n| n.to_string())
        )
    } else {
        format!("canary RED · {}", r.failures.join("; "))
    }
}
