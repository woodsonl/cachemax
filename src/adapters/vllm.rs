//! vLLM adapter — `/metrics` Prometheus counters, delta-sampled per request.
//!
//! vLLM exposes cumulative prefix-cache counters (`vllm:prefix_cache_queries`,
//! `vllm:prefix_cache_hits`, `vllm:prompt_tokens_cached`), not per-request
//! figures. The proxy samples the counters around a request and takes the
//! delta: `hits_after - hits_before` is that request's cached-prefix tokens.
//!
//! Measured-with-caveat: ±5% at session-aggregate. Under concurrent load the
//! delta can attribute another request's tokens, so the aggregate is the
//! trustworthy figure, not any single delta.

use super::{Adapter, CacheSignal};
use crate::record::SourceLabel;

#[derive(Default)]
pub struct VllmAdapter;

/// A snapshot of the relevant vLLM counters. Delta between two snapshots gives
/// one request's contribution.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PromCounters {
    /// `vllm:prefix_cache_queries` — queried prefix tokens.
    pub queries: u64,
    /// `vllm:prefix_cache_hits` — cached prefix tokens served.
    pub hits: u64,
    /// `vllm:prompt_tokens_cached` — cached prompt tokens (local + external).
    pub prompt_cached: u64,
}

impl PromCounters {
    /// Parse the Prometheus text exposition. Sums across label sets (engines),
    /// ignoring lines whose metric name isn't one we track.
    pub fn parse(text: &str) -> Self {
        let mut c = PromCounters::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((name, rest)) = split_metric(line) else {
                continue;
            };
            // Prometheus exposition writes counters with a `_total` suffix
            // (`vllm:prefix_cache_hits_total`); the docs name them without it.
            // Strip the suffix so both spellings match.
            let name = name.strip_suffix("_total").unwrap_or(name);
            // `rest` is `{labels} value [timestamp]`; the value is the first
            // whitespace-separated token after any label set, not the last
            // (the last would be the optional timestamp).
            let value = rest
                .split_whitespace()
                .find_map(|v| v.parse::<f64>().ok())
                .map(|v| v as u64);
            let Some(value) = value else { continue };
            match name {
                "vllm:prefix_cache_queries" => c.queries += value,
                "vllm:prefix_cache_hits" => c.hits += value,
                "vllm:prompt_tokens_cached" => c.prompt_cached += value,
                _ => {}
            }
        }
        c
    }

    /// Tokens served from cache between `before` and `after`. Uses hits, and
    /// falls back to the prompt-cached counter when hits is flat.
    pub fn delta_hits(before: Self, after: Self) -> u64 {
        let hits = after.hits.saturating_sub(before.hits);
        if hits > 0 {
            hits
        } else {
            after.prompt_cached.saturating_sub(before.prompt_cached)
        }
    }

    /// Queried prefix tokens between two snapshots.
    pub fn delta_queries(before: Self, after: Self) -> u64 {
        after.queries.saturating_sub(before.queries)
    }
}
/// Split a Prometheus line into `(metric_name, remainder)`.
fn split_metric(line: &str) -> Option<(&str, &str)> {
    let brace = line.find('{');
    let space = line.find(' ');
    let end = match (brace, space) {
        (Some(b), Some(s)) => b.min(s),
        (Some(b), None) => b,
        (None, Some(s)) => s,
        (None, None) => return None,
    };
    let name = &line[..end];
    let rest = &line[end..];
    if name.is_empty() {
        None
    } else {
        Some((name, rest))
    }
}

impl Adapter for VllmAdapter {
    fn name(&self) -> &'static str {
        "vllm"
    }

    fn source(&self) -> SourceLabel {
        SourceLabel::EngineMeasured
    }

    /// A single body here is either a pre-computed delta payload
    /// `{"cached_tokens":N}` (the per-request figure the proxy samples) or raw
    /// Prometheus text. Raw text carries only *cumulative* counters, which are
    /// not a per-turn measurement, so it reports no cache truth rather than
    /// passing an ever-growing total off as this turn's cached tokens. The
    /// per-request number comes from [`PromCounters::delta_hits`] between two
    /// samples (see the module note), not from this single-body path.
    fn cache_signal(&self, response_body: &[u8]) -> CacheSignal {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(response_body) {
            if let Some(n) = v
                .get("cached_tokens")
                .and_then(crate::adapters::usage_count)
            {
                return CacheSignal::reported(n, SourceLabel::EngineMeasured);
            }
        }
        CacheSignal {
            cached_tokens: 0,
            written_tokens: 0,
            source: Some(SourceLabel::NoCacheTruth),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# HELP vllm:prefix_cache_queries Prefix cache queries.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{model_name="llama",engine="0"} 1000.0
vllm:prefix_cache_hits{model_name="llama",engine="0"} 640.0
vllm:prompt_tokens_cached{model_name="llama",engine="0"} 640.0
# HELP other
vllm:num_requests_running{model_name="llama",engine="0"} 3.0
"#;

    #[test]
    fn parses_the_tracked_counters() {
        let c = PromCounters::parse(SAMPLE);
        assert_eq!(c.queries, 1000);
        assert_eq!(c.hits, 640);
        assert_eq!(c.prompt_cached, 640);
    }

    #[test]
    fn parses_counters_exposed_with_the_total_suffix() {
        // Prometheus exposition writes counters as `<name>_total`. This is the
        // form actually on the wire from a real vLLM `/metrics`.
        let text = r#"
vllm:prefix_cache_queries_total{model_name="llama"} 1250.0
vllm:prefix_cache_hits_total{model_name="llama"} 940.0
"#;
        let c = PromCounters::parse(text);
        assert_eq!(c.queries, 1250);
        assert_eq!(c.hits, 940);
    }

    #[test]
    fn ignores_a_trailing_timestamp() {
        // Prometheus allows `<metric> <value> <timestamp_ms>`; the value is not
        // the last token.
        let text = "vllm:prefix_cache_hits_total 940.0 1700000000000\n";
        assert_eq!(PromCounters::parse(text).hits, 940);
    }

    #[test]
    fn delta_between_samples_is_one_requests_contribution() {
        let before = PromCounters::parse(SAMPLE);
        let after =
            PromCounters::parse(&SAMPLE.replace("1000.0", "1250.0").replace("640.0", "940.0"));
        assert_eq!(PromCounters::delta_hits(before, after), 300);
        assert_eq!(PromCounters::delta_queries(before, after), 250);
    }

    #[test]
    fn delta_falls_back_to_prompt_cached_when_hits_flat() {
        let before = PromCounters {
            queries: 0,
            hits: 100,
            prompt_cached: 100,
        };
        let after = PromCounters {
            queries: 0,
            hits: 100,
            prompt_cached: 350,
        };
        assert_eq!(PromCounters::delta_hits(before, after), 250);
    }

    #[test]
    fn counter_reset_saturates_rather_than_underflows() {
        let before = PromCounters {
            hits: 500,
            ..Default::default()
        };
        let after = PromCounters {
            hits: 10,
            ..Default::default()
        };
        assert_eq!(PromCounters::delta_hits(before, after), 0);
    }

    #[test]
    fn sums_across_engines() {
        let text = r#"
vllm:prefix_cache_hits{engine="0"} 100.0
vllm:prefix_cache_hits{engine="1"} 50.0
"#;
        assert_eq!(PromCounters::parse(text).hits, 150);
    }

    #[test]
    fn json_delta_payload_is_read_directly() {
        let body = br#"{"cached_tokens":412}"#;
        let sig = VllmAdapter.cache_signal(body);
        assert_eq!(sig.cached_tokens, 412);
        assert_eq!(sig.source, Some(SourceLabel::EngineMeasured));
    }

    #[test]
    fn raw_prometheus_body_is_not_passed_off_as_a_per_turn_number() {
        // A single metrics scrape is cumulative; reporting it as this turn's
        // cached tokens would be a silent wrong number. It reports no truth.
        let sig = VllmAdapter.cache_signal(SAMPLE.as_bytes());
        assert_eq!(sig.cached_tokens, 0);
        assert_eq!(sig.source, Some(SourceLabel::NoCacheTruth));
    }
}
