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
            })
            .unwrap_or(0);
        CacheSignal::reported(cached, SourceLabel::ProviderReported)
    }
}
