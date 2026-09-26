//! Proxy core: forward first, unbuffered SSE passthrough, observe while streaming,
//! finalize one record per request. C1 fills in the axum/hyper wiring; the shape
//! is the contract.

use crate::adapters::Adapter;
use crate::record::{Record, SourceLabel, Status};

/// A finalized-request outcome the proxy hands to the session store.
pub struct Finalize {
    pub record: Record,
}

/// Build a record from the observed response. This is the pure seam C1 tests
/// against, independent of the network.
#[allow(clippy::too_many_arguments)]
pub fn build_record(
    session_id: u64,
    turn: u32,
    ttft_ms: Option<f64>,
    cached_tokens: u64,
    resent_history_tokens: u64,
    billed_input_tokens: u64,
    cost_usd: Option<f64>,
    source: SourceLabel,
    complete: bool,
) -> Record {
    Record {
        session_id,
        turn,
        status: if complete {
            Status::Complete
        } else {
            Status::Incomplete
        },
        source,
        ttft_ms,
        cached_tokens,
        resent_history_tokens,
        billed_input_tokens,
        cost_usd,
    }
}

/// The cache signal an adapter would read from a response, factored out so the
/// proxy and the adapter share one path.
pub fn observe<A: Adapter>(adapter: &A, response_body: &[u8]) -> (u64, SourceLabel) {
    let sig = adapter.cache_signal(response_body);
    (sig.cached_tokens, sig.source.unwrap_or_else(|| adapter.source()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::openai::OpenAiAdapter;

    #[test]
    fn a_dropped_stream_finalizes_incomplete() {
        let r = build_record(
            1,
            1,
            Some(120.0),
            0,
            1550,
            1750,
            None,
            SourceLabel::ProviderReported,
            false,
        );
        assert_eq!(r.status, Status::Incomplete);
        assert!(r.hit_rate().is_none() || r.hit_rate().is_some());
    }

    #[test]
    fn observe_routes_through_the_adapter() {
        let body = br#"{"usage":{"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        let (cached, source) = observe(&OpenAiAdapter::default(), body);
        assert_eq!(cached, 1455);
        assert_eq!(source, SourceLabel::ProviderReported);
    }
}
