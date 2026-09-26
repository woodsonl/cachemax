//! Anthropic adapter. Anthropic exposes a write/read split, not a single hit
//! count: `cache_creation_input_tokens` were written to cache (billed at a
//! premium); `cache_read_input_tokens` were served from cache (billed at a
//! discount). The hit-rate is derived as read / (read + creation).

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct AnthropicAdapter;

impl AnthropicAdapter {
    /// The write/read split for display, plus the derived hit-rate inputs.
    /// `None` when neither field is present (no cache activity exposed).
    pub fn split(response_body: &[u8]) -> Option<(u64, u64)> {
        let v: serde_json::Value = serde_json::from_slice(response_body).ok()?;
        let read = v
            .pointer("/usage/cache_read_input_tokens")
            .and_then(|n| n.as_u64());
        let creation = v
            .pointer("/usage/cache_creation_input_tokens")
            .and_then(|n| n.as_u64());
        match (read, creation) {
            (None, None) => None,
            (r, c) => Some((r.unwrap_or(0), c.unwrap_or(0))),
        }
    }

    /// The derived cache-hit share shown beside the write/read split:
    /// `read / (read + creation)`. `None` when no cache activity is exposed.
    pub fn derived_hit_rate(response_body: &[u8]) -> Option<f64> {
        let (read, creation) = Self::split(response_body)?;
        let total = read + creation;
        if total == 0 {
            None
        } else {
            Some(read as f64 / total as f64)
        }
    }
}

impl Adapter for AnthropicAdapter {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::ProviderReported
    }

    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal {
        match Self::split(response_body) {
            Some((read, creation)) => CacheSignal::reported_split(read, creation),
            // No cache fields at all: report no truth rather than a fake zero.
            None => CacheSignal::none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reads_both_fields() {
        let body =
            br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
        assert_eq!(AnthropicAdapter::split(body), Some((900, 300)));
    }

    #[test]
    fn no_cache_fields_is_no_truth_not_a_fabricated_zero() {
        let body = br#"{"usage":{"input_tokens":100}}"#;
        assert_eq!(AnthropicAdapter::split(body), None);
        let sig = AnthropicAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 0);
        assert_eq!(sig.source, Some(SourceLabel::NoCacheTruth));
    }

    #[test]
    fn derived_hit_rate_is_read_over_read_plus_creation() {
        let body =
            br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
        let rate = AnthropicAdapter::derived_hit_rate(body).unwrap();
        assert!((rate - 0.75).abs() < 1e-9, "900/(900+300) = 0.75");
    }

    #[test]
    fn derived_hit_rate_is_none_without_cache_activity() {
        let body = br#"{"usage":{"input_tokens":100}}"#;
        assert_eq!(AnthropicAdapter::derived_hit_rate(body), None);
    }

    #[test]
    fn signal_carries_the_write_split() {
        let body =
            br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
        let sig = AnthropicAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 900);
        assert_eq!(sig.written_tokens, 300);
    }

    #[test]
    fn record_matches_provider_usage_exactly() {
        // The proxy never re-derives a provider figure: the record's cached and
        // written counts must equal the provider's own numbers.
        let body =
            br#"{"usage":{"cache_read_input_tokens":1234,"cache_creation_input_tokens":567}}"#;
        let sig = AnthropicAdapter.cache_signal(body);
        assert_eq!((sig.cached_tokens, sig.written_tokens), (1234, 567));
    }
}
