//! Adapter registry. An adapter reads the cache signal out of a provider or
//! engine response and reports its source label. Everything above this trait is
//! testable engineless via mocks (the spec's CI requirement).

pub mod anthropic;
pub mod llamacpp;
pub mod mlxlm;
pub mod openai;
pub mod vllm;

use crate::record::SourceLabel;

/// Read a usage count that may arrive in any numeric form. Providers and
/// gateways emit `"900"`, `900.0`, and `1e3` for the same count; under
/// `serde_json`'s `arbitrary_precision` the non-integer forms parse as text
/// and `as_u64` returns `None` — reading that as cache-absent would fake a
/// 0% headline. Integral floats and exponents are accepted.
pub fn usage_count(n: &serde_json::Value) -> Option<u64> {
    n.as_u64().or_else(|| {
        n.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0)
            .map(|f| f as u64)
    })
}

/// A normalized cache observation extracted from one response.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheSignal {
    /// Prefix tokens served from cache, as reported by the provider or engine.
    pub cached_tokens: u64,
    /// Prefix tokens written to cache this turn (Anthropic's creation count;
    /// 0 where the provider exposes none).
    pub written_tokens: u64,
    /// Where the number came from.
    pub source: Option<SourceLabel>,
}

impl CacheSignal {
    pub fn reported(cached_tokens: u64, source: SourceLabel) -> Self {
        Self {
            cached_tokens,
            written_tokens: 0,
            source: Some(source),
        }
    }

    /// A reported signal that also carries a cache-write count (Anthropic).
    pub fn reported_split(cached_tokens: u64, written_tokens: u64) -> Self {
        Self {
            cached_tokens,
            written_tokens,
            source: Some(SourceLabel::ProviderReported),
        }
    }

    /// No signal exposed (e.g. mlx-lm): a valid, complete observation.
    pub fn none() -> Self {
        Self {
            cached_tokens: 0,
            written_tokens: 0,
            source: Some(SourceLabel::NoCacheTruth),
        }
    }
}

/// What a backend adapter can do.
pub trait Adapter: Send + Sync {
    /// Human-readable backend name (the `--backend` value).
    fn name(&self) -> &'static str;

    /// Extract the cache signal from a raw provider/engine response body.
    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal;

    /// The source label this adapter's figures carry.
    fn source(&self) -> SourceLabel;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::SourceLabel;

    #[test]
    fn openai_reads_cached_tokens_dotted_path() {
        let body =
            br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        let sig = openai::OpenAiAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 1455);
        assert_eq!(sig.source, Some(SourceLabel::ProviderReported));
    }

    #[test]
    fn anthropic_reads_read_and_creation_split() {
        let body =
            br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
        let sig = anthropic::AnthropicAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 900);
        assert_eq!(sig.source, Some(SourceLabel::ProviderReported));
    }

    #[test]
    fn mlxlm_exposes_no_cache_truth() {
        let adapter = mlxlm::MlxLmAdapter;
        let sig = adapter.cache_signal(b"{}");
        assert_eq!(sig.cached_tokens, 0);
        assert_eq!(adapter.source(), SourceLabel::NoCacheTruth);
    }
}
