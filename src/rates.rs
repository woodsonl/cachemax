//! Per-model token rates and per-provider cache multipliers, for the no-cache
//! cost counterfactual.
//!
//! Published rates change; a static table ships sane defaults and `--rates
//! <file>` overrides the whole table. Cost is deterministic and offline — no
//! network, so engineless tests are exact.
//!
//! Cost saved per the binding definition: the re-sent history priced at the
//! full input rate (the no-cache counterfactual) minus what it actually cost
//! with the provider's cache discount/premium applied.

use serde::Deserialize;
use std::collections::HashMap;

/// A model's published rates, in USD per 1M tokens, plus cache multipliers
/// expressed as a fraction of the full input rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelRate {
    /// USD per 1M fresh input tokens.
    pub input_per_mtok: f64,
    /// USD per 1M output tokens.
    pub output_per_mtok: f64,
    /// Cached-read price as a fraction of `input_per_mtok` (0.5 OpenAI, 0.1 Anthropic).
    pub cached_input_mult: f64,
    /// Cache-write price as a fraction of `input_per_mtok` (0.0 OpenAI, 1.25 Anthropic).
    pub cache_write_mult: f64,
}

impl ModelRate {
    /// Cost of `tokens` fresh input tokens.
    pub fn input_cost(&self, tokens: u64) -> f64 {
        tokens as f64 * self.input_per_mtok / 1_000_000.0
    }

    /// Cost of `tokens` served from cache.
    pub fn cached_cost(&self, tokens: u64) -> f64 {
        self.input_cost(tokens) * self.cached_input_mult
    }

    /// Cost of `tokens` written to cache (Anthropic premium; 0 where no premium).
    pub fn write_cost(&self, tokens: u64) -> f64 {
        self.input_cost(tokens) * self.cache_write_mult
    }
}

/// The rate table. Lookup is by longest model-name prefix, so
/// `gpt-4o-2024-08-06` matches the `gpt-4o` entry.
#[derive(Debug, Clone)]
pub struct Rates {
    /// (prefix, rate), kept sorted longest-prefix-first on construction.
    entries: Vec<(String, ModelRate)>,
}

#[derive(Deserialize)]
struct RateFile {
    models: HashMap<String, RateEntry>,
}

#[derive(Deserialize)]
struct RateEntry {
    input_per_mtok: f64,
    output_per_mtok: f64,
    #[serde(default = "half")]
    cached_input_mult: f64,
    #[serde(default)]
    cache_write_mult: f64,
}

fn half() -> f64 {
    0.5
}

impl Rates {
    /// The built-in table: known OpenAI and Anthropic rates.
    ///
    /// OpenAI generic cached-input discount is 0.5; Anthropic's read discount is
    /// 0.1 and 5-minute-TTL write premium is 1.25.
    pub fn builtin() -> Self {
        let o = |input: f64, output: f64| ModelRate {
            input_per_mtok: input,
            output_per_mtok: output,
            cached_input_mult: 0.5,
            cache_write_mult: 0.0,
        };
        let a = |input: f64, output: f64| ModelRate {
            input_per_mtok: input,
            output_per_mtok: output,
            cached_input_mult: 0.1,
            cache_write_mult: 1.25,
        };
        let mut entries = vec![
            ("gpt-4o".to_string(), o(2.50, 10.00)),
            ("gpt-4o-mini".to_string(), o(0.15, 0.60)),
            ("gpt-4-turbo".to_string(), o(10.00, 30.00)),
            ("gpt-4".to_string(), o(30.00, 60.00)),
            ("gpt-3.5-turbo".to_string(), o(0.50, 1.50)),
            ("claude-3-5-sonnet".to_string(), a(3.00, 15.00)),
            ("claude-3-5-haiku".to_string(), a(0.80, 4.00)),
            ("claude-3-opus".to_string(), a(15.00, 75.00)),
            ("claude-3-sonnet".to_string(), a(3.00, 15.00)),
            ("claude-3-haiku".to_string(), a(0.25, 1.25)),
        ];
        entries.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        Self { entries }
    }

    /// Load overrides from a JSON file (`{"models": {"<prefix>": {...}}}`).
    /// Entries replace same-prefix built-ins; others are added.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let file: RateFile = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let mut rates = Self::builtin();
        for (prefix, e) in file.models {
            let rate = ModelRate {
                input_per_mtok: e.input_per_mtok,
                output_per_mtok: e.output_per_mtok,
                cached_input_mult: e.cached_input_mult,
                cache_write_mult: e.cache_write_mult,
            };
            rates.entries.retain(|(p, _)| p != &prefix);
            rates.entries.push((prefix, rate));
        }
        rates.entries.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        Ok(rates)
    }

    /// Longest-prefix lookup. `None` when no model matches.
    pub fn lookup(&self, model: &str) -> Option<ModelRate> {
        self.entries
            .iter()
            .find(|(prefix, _)| model.starts_with(prefix.as_str()))
            .map(|(_, r)| *r)
    }

    /// Cost saved: the re-sent history at full input rate minus its actual
    /// cached cost. `history_tokens` is the binding denominator; `cached_tokens`
    /// the portion served from cache; `written_tokens` the portion written to
    /// cache this turn (Anthropic; 0 elsewhere).
    pub fn cost_saved(
        &self,
        model: &str,
        cached_tokens: u64,
        history_tokens: u64,
        written_tokens: u64,
    ) -> Option<f64> {
        let rate = self.lookup(model)?;
        let full = rate.input_cost(history_tokens);
        let actual_cached = rate.cached_cost(cached_tokens.min(history_tokens));
        let actual_write = rate.write_cost(written_tokens);
        // Tokens neither cached nor written bill at full rate; they cancel out
        // of the saving, so the saving is full-rate-history minus the cached and
        // written portions' actual cost, plus the write premium is a *cost*.
        Some(full - actual_cached - actual_write)
    }

    /// The repair counterfactual: the span repair actually restored, priced
    /// at the difference between base input and cached-read rates. Used
    /// only when the provider reports no cache truth — an estimate, never
    /// a measurement, and never blended into `cost_saved`.
    pub fn repair_estimated_saved(&self, model: &str, restored_tokens: u64) -> Option<f64> {
        let rate = self.lookup(model)?;
        Some(rate.input_cost(restored_tokens) * (1.0 - rate.cached_input_mult))
    }
}

impl Default for Rates {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_wins() {
        let r = Rates::builtin();
        assert_eq!(
            r.lookup("gpt-4o-mini-2024-07-18").unwrap().input_per_mtok,
            0.15
        );
        assert_eq!(r.lookup("gpt-4o-2024-08-06").unwrap().input_per_mtok, 2.50);
        assert!(r.lookup("unknown-model").is_none());
    }

    #[test]
    fn openai_cached_cost_is_half_of_input() {
        let r = Rates::builtin().lookup("gpt-4o").unwrap();
        assert_eq!(r.cached_cost(1_000_000), 1.25);
        assert_eq!(r.write_cost(1_000_000), 0.0);
    }

    #[test]
    fn anthropic_read_is_tenth_and_write_is_premium() {
        let r = Rates::builtin().lookup("claude-3-5-sonnet").unwrap();
        assert!((r.cached_cost(1_000_000) - 0.30).abs() < 1e-9);
        assert!((r.write_cost(1_000_000) - 3.75).abs() < 1e-9);
    }

    #[test]
    fn cost_saved_prices_history_at_full_rate_minus_cached() {
        // OpenAI gpt-4o at $2.50/1M: 2000-token history, 1500 cached.
        // full = 2000*2.5/1M = 0.005; cached actual = 1500*1.25/1M = 0.001875.
        // saved = 0.005 - 0.001875 = 0.003125.
        let r = Rates::builtin();
        let saved = r.cost_saved("gpt-4o", 1500, 2000, 0).unwrap();
        assert!((saved - 0.003125).abs() < 1e-12, "got {saved}");
    }

    #[test]
    fn unknown_model_has_no_cost() {
        assert_eq!(Rates::builtin().cost_saved("nope", 1, 1, 0), None);
    }

    #[test]
    fn override_replaces_a_prefix() {
        let json = r#"{"models":{"gpt-4o":{"input_per_mtok":99.0,"output_per_mtok":99.0}}}"#;
        let r = Rates::from_json(json).unwrap();
        assert_eq!(r.lookup("gpt-4o").unwrap().input_per_mtok, 99.0);
        // default cached_input_mult 0.5 still applies
        assert_eq!(r.lookup("gpt-4o").unwrap().cached_input_mult, 0.5);
    }
}

#[cfg(test)]
mod estimate_tests {
    use super::*;

    #[test]
    fn repair_estimate_prices_the_span_at_input_minus_cached_read() {
        // Anthropic claude-3-5-sonnet: $3/M input, cached reads at 0.1x.
        // 1000 at-risk tokens save 1000 × 3 × 0.9 / 1M = $0.0027.
        let rates = Rates::builtin();
        let saved = rates
            .repair_estimated_saved("claude-3-5-sonnet-20241022", 1000)
            .unwrap();
        assert!((saved - 0.0027).abs() < 1e-9, "got {saved}");
        // Unknown model: no rate, no estimate — never a guess.
        assert_eq!(rates.repair_estimated_saved("mystery-model", 1000), None);
    }
}
