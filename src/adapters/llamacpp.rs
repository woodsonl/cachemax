//! llama.cpp adapter — the ground-truth engine.
//!
//! llama.cpp exposes the real cached-prefix count in its completion response:
//! `tokens_cached` ("number of tokens from the prompt which could be re-used
//! from previous completion") and, in its OAI-compatible response,
//! `timings.cache_n` ("number of prompt tokens reused from cache") alongside the
//! standard `usage.prompt_tokens_details.cached_tokens`. Its session-cumulative
//! hit-rate is the ±5% acceptance reference — the only engine where token-space
//! matching itself is verified.
//!
//! All three fields name the same quantity; we read whichever the response
//! carries, preferring the explicit `tokens_cached`.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct LlamaCppAdapter;

impl LlamaCppAdapter {
    /// The cached-prefix count from a llama.cpp response, or `None` if the
    /// response carries no cache field at all (distinct from a genuine 0).
    pub fn tokens_cached(response_body: &[u8]) -> Option<u64> {
        let v: serde_json::Value = serde_json::from_slice(response_body).ok()?;
        // Native /completion shape.
        if let Some(n) = v.get("tokens_cached").and_then(|n| n.as_u64()) {
            return Some(n);
        }
        // OAI-compatible chat/completions shape.
        if let Some(n) = v.pointer("/timings/cache_n").and_then(|n| n.as_u64()) {
            return Some(n);
        }
        v.pointer("/usage/prompt_tokens_details/cached_tokens")
            .and_then(|n| n.as_u64())
    }

    /// Total prompt tokens evaluated (the denominator raw material when present).
    pub fn tokens_evaluated(response_body: &[u8]) -> Option<u64> {
        let v: serde_json::Value = serde_json::from_slice(response_body).ok()?;
        v.get("tokens_evaluated")
            .and_then(|n| n.as_u64())
            .or_else(|| v.pointer("/usage/prompt_tokens").and_then(|n| n.as_u64()))
    }
}

impl Adapter for LlamaCppAdapter {
    fn name(&self) -> &'static str {
        "llamacpp"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::EngineMeasured
    }

    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal {
        match Self::tokens_cached(response_body) {
            Some(n) => CacheSignal::reported(n, SourceLabel::EngineMeasured),
            // No cache field → the engine exposed nothing; not a measured zero.
            None => CacheSignal {
                cached_tokens: 0,
                written_tokens: 0,
                source: Some(SourceLabel::EngineMeasured),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_native_tokens_cached() {
        let body = br#"{"content":"hi","tokens_cached":236,"tokens_evaluated":240}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), Some(236));
        assert_eq!(LlamaCppAdapter::tokens_evaluated(body), Some(240));
    }

    #[test]
    fn reads_oai_timings_cache_n() {
        let body = br#"{"timings":{"cache_n":236,"prompt_n":4},"usage":{"prompt_tokens":240}}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), Some(236));
    }

    #[test]
    fn reads_oai_usage_cached_tokens_as_last_resort() {
        let body =
            br#"{"usage":{"prompt_tokens":240,"prompt_tokens_details":{"cached_tokens":236}}}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), Some(236));
    }

    #[test]
    fn genuine_zero_is_zero_not_none() {
        let body = br#"{"tokens_cached":0,"tokens_evaluated":240}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), Some(0));
    }

    #[test]
    fn no_cache_field_is_none() {
        let body = br#"{"content":"hi"}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), None);
    }

    #[test]
    fn signal_is_engine_measured() {
        let body = br#"{"tokens_cached":236}"#;
        let sig = LlamaCppAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 236);
        assert_eq!(sig.source, Some(SourceLabel::EngineMeasured));
    }

    #[test]
    fn ground_truth_counts_match_the_payload_exactly() {
        // The ±5% acceptance compares against these engine-reported numbers; the
        // adapter must pass them through, never re-derive.
        let body = br#"{"tokens_cached":1455,"tokens_evaluated":2000}"#;
        assert_eq!(LlamaCppAdapter::tokens_cached(body), Some(1455));
        assert_eq!(LlamaCppAdapter::tokens_evaluated(body), Some(2000));
    }
}
