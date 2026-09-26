//! llama.cpp adapter — the ground-truth engine. `/slots` exposes real
//! cached-token counts; its session-cumulative hit-rate is the ±5% acceptance
//! reference. Parsing the `/slots` payload lands with C3; the boundary is here.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct LlamaCppAdapter;

impl Adapter for LlamaCppAdapter {
    fn name(&self) -> &'static str {
        "llamacpp"
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
