//! The normalized record type — the shared abstraction every other module reads.

use serde::{Deserialize, Serialize};

use crate::repair::{DriftKind, RepairMode};

/// Whether a request produced a usable measurement.
///
/// Orthogonal to [`SourceLabel`]: a cloud request can be `Complete` with source
/// `ProviderReported`; an mlx-lm request is `Complete` with `NoCacheTruth`.
/// Only `Incomplete` records are excluded from session-cumulative aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Complete,
    Incomplete,
}

/// Where a cache figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLabel {
    /// The cloud provider's own number (OpenAI `cached_tokens`, Anthropic read/creation).
    ProviderReported,
    /// Measured from a local engine (llama.cpp `/slots`, vLLM `/metrics`).
    EngineMeasured,
    /// No cache signal is exposed (mlx-lm); discrimination only, no hit-rate number.
    NoCacheTruth,
}

/// One measured turn.
///
/// `cached_tokens` is the provider- or engine-reported count of prefix tokens
/// served from cache. `resent_history_tokens` is the binding denominator: the
/// token count of the re-sent message list (system + all prior user, assistant,
/// and tool messages), excluding this turn's new content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Session this request belongs to.
    pub session_id: u64,
    /// Turn index within the session. Turn 0 is cold and excluded from the formula.
    pub turn: u32,
    pub status: Status,
    pub source: SourceLabel,
    /// Time to first token, milliseconds.
    pub ttft_ms: Option<f64>,
    /// Provider- or engine-reported cached prefix tokens.
    pub cached_tokens: u64,
    /// Prefix tokens written to cache this turn (Anthropic's creation count).
    /// 0 where the provider exposes no write/premium distinction.
    pub cache_written_tokens: u64,
    /// Binding denominator: system + all prior messages, excluding this turn's new content.
    pub resent_history_tokens: u64,
    /// Billed input tokens for this turn.
    pub billed_input_tokens: u64,
    /// This turn's prefix broke against the tracked session (a cache miss that
    /// stays in the session). Drives the tape's break/miss glyph.
    #[serde(default)]
    pub broke_prefix: bool,
    /// Cost in USD at the provider's published rate, if known.
    pub cost_usd: Option<f64>,
    /// Cost saved versus the no-cache counterfactual, if rates are known.
    pub cost_saved_usd: Option<f64>,
    /// The repair mode this turn ran under. `off` turns carry no drift
    /// claim at all.
    #[serde(default)]
    pub repair_mode: RepairMode,
    /// True when the outgoing history was rewritten to the canonical chain
    /// (`on` mode only; dry-run never rewrites).
    #[serde(default)]
    pub repaired: bool,
    /// `None` when repair did not examine the turn (mode off). `Some(true)`:
    /// the re-sent history byte-matches the canonical chain. `Some(false)`:
    /// drift (see `drift_kind`) or an unrepairable hard stop.
    #[serde(default)]
    pub matches_canonical: Option<bool>,
    /// The classified flavor of the drift, when there was one.
    #[serde(default)]
    pub drift_kind: Option<DriftKind>,
    /// Tokens of drifted history replaced by (`on`) or that would be
    /// replaced by (`dry-run`) the canonical serialization. An estimate for
    /// annotation; repair decisions never consult token counts.
    #[serde(default)]
    pub canonicalized_tokens: u64,
}

impl Record {
    /// Per-turn hit rate per the binding formula.
    ///
    /// `turn == 0` is cold and returns `None`. A zero denominator (empty
    /// history) also returns `None` — rendered as `—`, never `0`. A turn whose
    /// source exposes no cache truth (e.g. mlx-lm) has no rate to show, so it
    /// is unexposed (`None`), never a measured `0%`.
    pub fn hit_rate(&self) -> Option<f64> {
        if self.turn == 0 || self.resent_history_tokens == 0 {
            return None;
        }
        if self.source == SourceLabel::NoCacheTruth {
            return None;
        }
        Some(self.cached_tokens as f64 / self.resent_history_tokens as f64)
    }
}

/// Session-cumulative hit rate: `Σ cached / Σ resent_history` over complete
/// turns only (turn ≥ 1). Returns `None` when no complete turn contributes.
///
/// Turns that expose no cache truth (e.g. mlx-lm) and turns with a zero
/// history (nothing re-sent) contribute nothing: counting their `cached_tokens`
/// against a zero/absent denominator would inflate or fabricate the rate.
pub fn cumulative_hit_rate(records: &[Record]) -> Option<f64> {
    let mut cached: u64 = 0;
    let mut history: u64 = 0;
    for r in records {
        if r.status == Status::Incomplete
            || r.turn == 0
            || r.resent_history_tokens == 0
            || r.source == SourceLabel::NoCacheTruth
        {
            continue;
        }
        cached += r.cached_tokens;
        history += r.resent_history_tokens;
    }
    if history == 0 {
        None
    } else {
        Some(cached as f64 / history as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(turn: u32, status: Status, cached: u64, history: u64) -> Record {
        Record {
            session_id: 1,
            turn,
            status,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: cached,
            resent_history_tokens: history,
            billed_input_tokens: history + 200,
            broke_prefix: false,
            cache_written_tokens: 0,
            cost_usd: None,
            cost_saved_usd: None,
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
        }
    }

    /// The binding formula fixture: a hand-computed multi-turn conversation.
    ///
    /// Session tokens: system+tools S = 800. User turns U = [340, 260], assistant
    /// replies A = [410]. So:
    ///   t1 resent_history = S + U0 + A0        = 800+340+410 = 1550
    ///   t2 resent_history = t1 + U1            = 1550+260    = 1810
    /// Cached: t1 = 1020, t2 = 860.
    ///   t1 hit = 1020/1550 ≈ 0.658
    ///   t2 hit =  860/1810 ≈ 0.475
    ///   cumulative = (1020+860)/(1550+1810) = 1880/3360 ≈ 0.5595
    #[test]
    fn hit_rate_matches_hand_computed_fixture() {
        let t0 = rec(0, Status::Complete, 0, 0);
        let t1 = rec(1, Status::Complete, 1020, 1550);
        let t2 = rec(2, Status::Complete, 860, 1810);

        assert_eq!(t0.hit_rate(), None, "turn 0 is cold and excluded");
        assert!((t1.hit_rate().unwrap() - 1020.0 / 1550.0).abs() < 1e-9);
        assert!((t2.hit_rate().unwrap() - 860.0 / 1810.0).abs() < 1e-9);

        let cum = cumulative_hit_rate(&[t0, t1, t2]).unwrap();
        assert!((cum - 1880.0 / 3360.0).abs() < 1e-9);
    }

    #[test]
    fn zero_denominator_is_none_not_zero() {
        let r = rec(1, Status::Complete, 0, 0);
        assert_eq!(r.hit_rate(), None, "empty history renders as — , never 0");
    }

    #[test]
    fn incomplete_records_are_excluded_from_cumulative() {
        let t1 = rec(1, Status::Complete, 1000, 2000); // 0.5
        let t2 = rec(2, Status::Incomplete, 9999, 9999); // must be ignored
        let cum = cumulative_hit_rate(&[t1, t2]).unwrap();
        assert!((cum - 0.5).abs() < 1e-9);
    }
}
