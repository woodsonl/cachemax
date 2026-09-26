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
    pub fn split(response_body: &[u8]) -> (u64, u64) {
        let v: serde_json::Value = serde_json::from_slice(response_body).unwrap_or_default();
        let read = v
            .pointer("/usage/cache_read_input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0);
        let creation = v
            .pointer("/usage/cache_creation_input_tokens")
            .and_then(|n| n.as_u64())
            .unwrap_or(0);
        (read, creation)
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
        let (read, _creation) = Self::split(response_body);
        CacheSignal::reported(read, SourceLabel::ProviderReported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reads_both_fields() {
        let body = br#"{"usage":{"cache_read_input_tokens":900,"cache_creation_input_tokens":300}}"#;
        assert_eq!(AnthropicAdapter::split(body), (900, 300));
    }
}
