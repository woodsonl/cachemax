//! vLLM adapter — `/metrics` token counters, delta-sampled per request.
//! Measured-with-caveat: record fidelity ±5% at session-aggregate; falls back
//! to log-line parsing if delta-sampling is too noisy. Lands with C3.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct VllmAdapter;

impl Adapter for VllmAdapter {
    fn name(&self) -> &'static str {
        "vllm"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::EngineMeasured
    }

    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal {
        let cached = serde_json::from_slice::<serde_json::Value>(response_body)
            .ok()
            .and_then(|v| v.get("cached_tokens").and_then(|n| n.as_u64()))
            .unwrap_or(0);
        CacheSignal::reported(cached, SourceLabel::EngineMeasured)
    }
}
