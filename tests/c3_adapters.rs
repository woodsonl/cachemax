//! C3 engineless adapter suite: every backend is exercised through the proxy's
//! observation path against recorded payloads — no engine, no network.
//!
//! This is the mocked-adapter CI the spec requires. Each case asserts the
//! adapter extracts exactly the engine/provider-reported figure and carries the
//! right source label, then that the figure survives `observe` + `build_record`.

use cachemax::adapters::anthropic::AnthropicAdapter;
use cachemax::adapters::llamacpp::LlamaCppAdapter;
use cachemax::adapters::mlxlm::MlxLmAdapter;
use cachemax::adapters::openai::OpenAiAdapter;
use cachemax::adapters::vllm::{PromCounters, VllmAdapter};
use cachemax::proxy::{build_record, observe, Observation, RequestPlan};
use cachemax::rates::Rates;
use cachemax::record::SourceLabel;

#[test]
fn openai_observe_carries_provider_numbers() {
    let body =
        br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
    let (cached, written, source) = observe(&OpenAiAdapter, body);
    assert_eq!((cached, written), (1455, 0));
    assert_eq!(source, SourceLabel::ProviderReported);
}

#[test]
fn anthropic_observe_carries_read_creation_split() {
    let body = br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
    let (cached, written, source) = observe(&AnthropicAdapter, body);
    assert_eq!((cached, written), (900, 300));
    assert_eq!(source, SourceLabel::ProviderReported);
}

#[test]
fn llamacpp_observe_is_engine_measured_ground_truth() {
    let body = br#"{"content":"hi","tokens_cached":1455,"tokens_evaluated":2000}"#;
    let (cached, _written, source) = observe(&LlamaCppAdapter, body);
    assert_eq!(cached, 1455);
    assert_eq!(source, SourceLabel::EngineMeasured);
}

#[test]
fn vllm_delta_is_engine_measured() {
    let before =
        PromCounters::parse("vllm:prefix_cache_hits 1000.0\nvllm:prefix_cache_queries 1500.0\n");
    let after =
        PromCounters::parse("vllm:prefix_cache_hits 1455.0\nvllm:prefix_cache_queries 2000.0\n");
    let delta = PromCounters::delta_hits(before, after);
    let body = format!(r#"{{"cached_tokens":{delta}}}"#);
    let (cached, _written, source) = observe(&VllmAdapter, body.as_bytes());
    assert_eq!(cached, 455);
    assert_eq!(source, SourceLabel::EngineMeasured);
}

#[test]
fn mlxlm_observe_is_no_cache_truth() {
    let (cached, written, source) = observe(&MlxLmAdapter, b"{}");
    assert_eq!((cached, written), (0, 0));
    assert_eq!(source, SourceLabel::NoCacheTruth);
}

#[test]
fn observed_figure_survives_into_the_record() {
    let plan = RequestPlan {
        session_id: 1,
        turn: 1,
        resent_history_tokens: 2000,
    };
    let body =
        br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
    let (cached, written, source) = observe(&OpenAiAdapter, body);
    let obs = Observation {
        ttft_ms: Some(120.0),
        cached_tokens: cached,
        cache_written_tokens: written,
        billed_input_tokens: 2140,
    };
    let r = build_record(&plan, obs, "gpt-4o", &Rates::builtin(), source, true);
    assert_eq!(r.cached_tokens, 1455);
    assert_eq!(r.resent_history_tokens, 2000);
    // 1455/2000 exactly, per the binding formula.
    assert!((r.hit_rate().unwrap() - 1455.0 / 2000.0).abs() < 1e-9);
}
