//! OpenAI adapter. Also covers OpenRouter: same wire shape, provider endpoint
//! set to `https://openrouter.ai/api/v1` via `--upstream-url`.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct OpenAiAdapter;

impl Adapter for OpenAiAdapter {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::ProviderReported
    }

    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal {
        let cached = serde_json::from_slice::<serde_json::Value>(response_body)
            .ok()
            .and_then(|v| {
                v.pointer("/usage/prompt_tokens_details/cached_tokens")
                    .and_then(|n| n.as_u64())
            });
        match cached {
            // The provider reported a figure (possibly a real 0): keep it.
            Some(n) => CacheSignal::reported(n, SourceLabel::ProviderReported),
            // No figure at all (uncached model, error body): report no truth
            // rather than fabricating a measured zero.
            None => CacheSignal {
                cached_tokens: 0,
                written_tokens: 0,
                source: Some(SourceLabel::NoCacheTruth),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_matches_provider_usage_exactly() {
        // Never re-derived: the record's cached count equals the provider's.
        let body =
            br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        let sig = OpenAiAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 1455);
        assert_eq!(sig.written_tokens, 0, "OpenAI exposes no write premium");
        assert_eq!(sig.source, Some(SourceLabel::ProviderReported));
    }

    #[test]
    fn openrouter_shape_uses_the_same_path() {
        // OpenRouter mirrors OpenAI's usage shape; the same adapter reads it.
        let body =
            br#"{"usage":{"prompt_tokens":900,"prompt_tokens_details":{"cached_tokens":640}}}"#;
        assert_eq!(OpenAiAdapter.cache_signal(body).cached_tokens, 640);
    }

    #[test]
    fn missing_cached_field_is_no_truth_not_a_fabricated_zero() {
        let body = br#"{"usage":{"prompt_tokens":100}}"#;
        let sig = OpenAiAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 0);
        assert_eq!(
            sig.source,
            Some(SourceLabel::NoCacheTruth),
            "an absent provider figure is not a reported zero"
        );
    }

    #[test]
    fn reported_zero_stays_provider_reported() {
        let body =
            br#"{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":0}}}"#;
        let sig = OpenAiAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 0);
        assert_eq!(sig.source, Some(SourceLabel::ProviderReported));
    }
}
