//! C5 micro-profile: where the proxy's per-request time actually goes.
//!
//! A coarse but honest breakdown of one proxied request's hot path:
//! tokenization + prefix hashing, session resolve, record build. These are the
//! only CPU steps the proxy adds besides the extra loopback hop (measured in
//! `c5_latency`). Printed, not asserted, so it runs on every commit without
//! flaking on a busy CI box.

use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::proxy::{build_record, observe, Observation, RequestPlan};
use cachemax::rates::Rates;
use cachemax::sessions::SessionStore;
use cachemax::tokenize::{Message, Tokenizer};
use std::time::Instant;

fn conversation(turns: usize) -> Vec<Message> {
    let mut v = vec![Message {
        role: "system".into(),
        text: "You are a helpful assistant. ".repeat(40),
    }];
    for i in 0..turns {
        v.push(Message {
            role: "user".into(),
            text: format!("Question {i}"),
        });
        v.push(Message {
            role: "assistant".into(),
            text: format!("Answer {i} ").repeat(20),
        });
    }
    v
}

#[test]
fn micro_profile_hot_path_stages() {
    let tokenizer = Tokenizer::default_encoder().unwrap();
    let mut store = SessionStore::new();
    let msgs = conversation(20);
    let body =
        br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
    const N: usize = 500;

    // Tokenize + prefix hash.
    let t = Instant::now();
    for _ in 0..N {
        let _ = tokenizer.prefix_hashes(&msgs);
    }
    let hash_us = t.elapsed().as_secs_f64() * 1e6 / N as f64;

    // Session resolve (store grows each iteration, as in a real session).
    let t = Instant::now();
    for _ in 0..N {
        let hashes = tokenizer.prefix_hashes(&msgs);
        let _ = store.resolve(&hashes);
    }
    let resolve_us = t.elapsed().as_secs_f64() * 1e6 / N as f64;

    // Observe + record build.
    let plan = RequestPlan {
        session_id: 1,
        turn: 1,
        resent_history_tokens: 2000,
        broke_prefix: false,
    };
    let t = Instant::now();
    for _ in 0..N {
        let (cached, written, source) = observe(&OpenAiAdapter, body);
        let obs = Observation {
            ttft_ms: Some(1.0),
            cached_tokens: cached,
            cache_written_tokens: written,
            billed_input_tokens: 2140,
        };
        let _ = build_record(&plan, obs, "gpt-4o", &Rates::builtin(), source, true);
    }
    let build_us = t.elapsed().as_secs_f64() * 1e6 / N as f64;

    eprintln!(
        "micro-profile (per request): prefix-hash {hash_us:.2} us, \
         resolve {resolve_us:.2} us, observe+build {build_us:.2} us"
    );

    // These run under `cargo test` (debug, unoptimized — ~10x the release cost).
    // The bound is generous to avoid CI flakes while still catching a real
    // regression, e.g. accidentally re-tokenizing the model's *output* or
    // rebuilding the prefix hash from scratch each turn.
    let debug_bound_us = 5000.0;
    assert!(
        hash_us < debug_bound_us && resolve_us < debug_bound_us && build_us < debug_bound_us,
        "a hot-path stage exceeded {debug_bound_us} us/request in debug: \
         hash {hash_us:.1}, resolve {resolve_us:.1}, build {build_us:.1} us"
    );
}
