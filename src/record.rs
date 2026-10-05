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
/// token count of the re-sent message list, excluding this turn's new content.
/// On OpenAI the system prompt is a `role: "system"` message and is counted;
/// on Anthropic it is a top-level `system` field and is not — the denominator
/// is dialect-dependent by construction, so a cached system breakpoint on
/// Anthropic can read above 100%. The floor nets a foreign wrapper, not this;
/// a client's own system-prompt cache is reported as the provider gave it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Session this request belongs to.
    pub session_id: u64,
    /// Turn index within the session. Turn 0 is cold and excluded from the formula.
    pub turn: u32,
    pub status: Status,
    pub source: SourceLabel,
    /// Time to first token, milliseconds — request send to the first
    /// non-empty response byte, so it covers the upstream's whole
    /// time-to-first-token (headers wait included). Streaming requests
    /// measure the first SSE data chunk; JSON bodies the first body byte.
    pub ttft_ms: Option<f64>,
    /// Provider- or engine-reported cached prefix tokens.
    pub cached_tokens: u64,
    /// Prefix tokens written to cache this turn (Anthropic's creation count).
    /// 0 where the provider exposes no write/premium distinction.
    pub cache_written_tokens: u64,
    /// Binding denominator: the history this request re-sent — on OpenAI the
    /// system prompt rides inside `messages` and counts; on Anthropic it is
    /// the top-level `system` field and does not (which is why a cached
    /// system breakpoint can read above 100%, the case the foreign-prefix
    /// floor exists for). Excludes this turn's new content.
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
    /// the re-sent history matches the canonical chain under semantic JSON
    /// equality (object key order is not drift; string leaves are). Some
    /// `false`: drift (see `drift_kind`) or an unrepairable hard stop.
    /// `None` also when the top-level `system` had no recorded baseline to
    /// examine (a chain from before systems were recorded): the claim covers
    /// `messages` only, and an unexamined span is not asserted as matching.
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
    /// `None` when breakpoint management is off (`--manage-breakpoints`
    /// unset, or a non-Anthropic backend). `Some(n)`: the request as
    /// forwarded carried `n` cache hints — the client's own, untouched,
    /// when management declined to touch them.
    #[serde(default)]
    pub breakpoint_count: Option<u64>,
}

impl Record {
    /// Per-turn hit rate with a foreign-prefix floor applied when the raw
    /// reading is impossible: a numerator larger than the whole re-sent
    /// history is a span the denominator cannot contain — the endpoint's
    /// wrapper caching its own prefix — and the floor (that wrapper's
    /// measured span) is subtracted. At or below 100% the raw rate stands:
    /// it is already honest, and subtracting a guessed floor would only
    /// deflate a direct provider's true number.
    ///
    /// `turn == 0` is cold and returns `None`. A zero denominator (empty
    /// history) also returns `None` — rendered as `—`, never `0`. A turn
    /// whose source exposes no cache truth (e.g. mlx-lm) has no rate to
    /// show, so it is unexposed (`None`), never a measured `0%`.
    pub fn hit_rate_net(&self, floor: u64) -> Option<f64> {
        if self.turn == 0 || self.resent_history_tokens == 0 {
            return None;
        }
        if self.source == SourceLabel::NoCacheTruth {
            return None;
        }
        if self.cached_tokens <= self.resent_history_tokens {
            return Some(self.cached_tokens as f64 / self.resent_history_tokens as f64);
        }
        Some(self.cached_tokens.saturating_sub(floor) as f64 / self.resent_history_tokens as f64)
    }
}

/// The foreign-prefix floor of a session: cached tokens reported on a turn
/// that re-sent no history **and wrote nothing to the cache**, therefore
/// attributable to the endpoint's own wrapper (router prompt, injected
/// preamble) rather than this conversation. The minimum over such turns; `0`
/// when none exists.
///
/// The write gate is what separates a wrapper from a client's own cache: a
/// cold turn that *wrote* what it read (Anthropic's `cache_creation_input_tokens`,
/// typically from a `cache_control` breakpoint on the system prompt) cached
/// this conversation's own prefix, so its reading is not foreign. Only a cold
/// turn that read a prefix it did not create reveals a foreign span. Known
/// limit: a wrapper that injects its own preamble WITH a breakpoint also
/// writes on the cold turn and is not learned as a floor; netting then
/// still applies through the >100% pathology gate, capped by whatever
/// floor was learned. Erring toward `0` is deliberate: a false floor would
/// under-report real reuse, the worse error.
pub fn router_prefix_floor(records: &[Record]) -> u64 {
    records
        .iter()
        .filter(|r| {
            r.status == Status::Complete
                && r.turn == 0
                && r.resent_history_tokens == 0
                && r.cache_written_tokens == 0
                && r.source != SourceLabel::NoCacheTruth
        })
        .map(|r| r.cached_tokens)
        .min()
        .unwrap_or(0)
}

/// Cumulative hit rate with the foreign-prefix floor applied where the raw
/// reading is impossible: a turn whose cached tokens exceed its own
/// re-sent history carries a span that history cannot contain — the
/// endpoint's wrapper caching its own prefix ([`router_prefix_floor`]) —
/// and that turn's numerator is netted. Honest turns (at or below 100%)
/// keep their raw numerator, for the same reason
/// [`Record::hit_rate_net`] leaves honest rates alone.
pub fn cumulative_hit_rate_net(records: &[Record], floor: u64) -> Option<f64> {
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
        let numerator = if r.cached_tokens > r.resent_history_tokens {
            r.cached_tokens.saturating_sub(floor)
        } else {
            r.cached_tokens
        };
        cached += numerator;
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
            breakpoint_count: None,
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

        assert_eq!(t0.hit_rate_net(0), None, "turn 0 is cold and excluded");
        assert!((t1.hit_rate_net(0).unwrap() - 1020.0 / 1550.0).abs() < 1e-9);
        assert!((t2.hit_rate_net(0).unwrap() - 860.0 / 1810.0).abs() < 1e-9);

        let cum = cumulative_hit_rate_net(&[t0, t1, t2], 0).unwrap();
        assert!((cum - 1880.0 / 3360.0).abs() < 1e-9);
    }

    #[test]
    fn zero_denominator_is_none_not_zero() {
        let r = rec(1, Status::Complete, 0, 0);
        assert_eq!(
            r.hit_rate_net(0),
            None,
            "empty history renders as — , never 0"
        );
    }

    #[test]
    fn incomplete_records_are_excluded_from_cumulative() {
        let t1 = rec(1, Status::Complete, 1000, 2000); // 0.5
        let t2 = rec(2, Status::Incomplete, 9999, 9999); // must be ignored
        let cum = cumulative_hit_rate_net(&[t1, t2], 0).unwrap();
        assert!((cum - 0.5).abs() < 1e-9);
    }

    #[test]
    fn router_floor_is_the_cold_turn_reading() {
        // A routed endpoint reports 128 cached tokens even on the cold turn
        // (its own wrapper) and wrote none of them — the router cached that
        // prefix before our request.
        let cold = rec(0, Status::Complete, 128, 0);
        let warm = rec(1, Status::Complete, 628, 500);
        let records = [cold, warm];
        assert_eq!(router_prefix_floor(&records), 128);

        // Raw rate exceeds 100%; the netted rate does not.
        let warm = rec(1, Status::Complete, 628, 500);
        assert!((warm.hit_rate_net(0).unwrap() - 628.0 / 500.0).abs() < 1e-9);
        let net = warm.hit_rate_net(128).unwrap();
        assert!(
            (net - 500.0 / 500.0).abs() < 1e-9,
            "628-128 over 500 = 100%"
        );
    }

    #[test]
    fn a_clients_own_cold_cache_is_not_a_floor() {
        // Direct Anthropic: a cold turn whose `cache_control` breakpoint
        // cached the system prompt reports a read AND a creation for that
        // same prefix. That is this conversation's own cache, not a foreign
        // wrapper — netting it would under-report real reuse.
        let mut cold = rec(0, Status::Complete, 4000, 0);
        cold.cache_written_tokens = 4200;
        let warm = rec(1, Status::Complete, 4500, 1000);
        let records = [cold, warm];
        assert_eq!(
            router_prefix_floor(&records),
            0,
            "a cold turn that wrote what it read owns that prefix"
        );
        // With no floor, the warm rate is the provider's own figure.
        let warm = rec(1, Status::Complete, 4500, 1000);
        assert_eq!(warm.hit_rate_net(0), warm.hit_rate_net(0));
    }

    #[test]
    fn direct_providers_have_no_floor_and_are_untouched() {
        // OpenAI reports 0 cached on a cold turn: floor 0, raw == net.
        let cold = rec(0, Status::Complete, 0, 0);
        let warm = rec(1, Status::Complete, 900, 1000);
        let records = [cold, warm];
        assert_eq!(router_prefix_floor(&records), 0);
        let warm = rec(1, Status::Complete, 900, 1000);
        assert_eq!(warm.hit_rate_net(0), warm.hit_rate_net(0));
    }

    #[test]
    fn a_learned_floor_is_not_applied_to_honest_rates() {
        // OpenAI's auto-cache reports the client's OWN system prompt as
        // cached on a cold turn with no write — the floor's exact signature.
        // The floor is learned, but the warm turn reads 90% (at or below
        // 100%): the raw rate stands. Netting there would deflate a direct
        // provider's true number while claiming a routed artifact.
        let cold = rec(0, Status::Complete, 210, 0);
        let warm = rec(1, Status::Complete, 900, 1000);
        assert_eq!(router_prefix_floor(&[cold, warm]), 210);
        let warm = rec(1, Status::Complete, 900, 1000);
        assert_eq!(
            warm.hit_rate_net(210),
            Some(0.9),
            "≤100% keeps the raw rate even with a learned floor"
        );
    }

    #[test]
    fn netting_applies_exactly_when_the_rate_is_impossible() {
        // A numerator larger than the whole re-sent history is a span the
        // denominator cannot contain — the wrapper's cached prefix. That is
        // the only case the floor nets, and the floor caps the subtraction.
        let cold = rec(0, Status::Complete, 210, 0);
        let impossible = rec(1, Status::Complete, 1400, 1000); // raw 140%
        let floor = router_prefix_floor(&[cold, impossible.clone()]);
        assert_eq!(floor, 210);
        let impossible = rec(1, Status::Complete, 1400, 1000);
        assert_eq!(
            impossible.hit_rate_net(floor),
            Some(1190.0 / 1000.0),
            "netted by the learned wrapper span"
        );
        // No floor learned: the honest absurdity stands — no invention.
        let impossible = rec(1, Status::Complete, 1400, 1000);
        assert_eq!(impossible.hit_rate_net(0), Some(1.4));
    }

    #[test]
    fn cumulative_netting_follows_the_same_per_turn_gate() {
        // One impossible turn (140%) and one honest turn (50%): the
        // impossible numerator is netted, the honest one untouched.
        let cold = rec(0, Status::Complete, 210, 0);
        let hot = rec(1, Status::Complete, 1400, 1000);
        let calm = rec(2, Status::Complete, 500, 1000);
        let floor = router_prefix_floor(&[cold, hot.clone(), calm.clone()]);
        let rate = cumulative_hit_rate_net(&[hot, calm], floor).unwrap();
        assert!((rate - (1190.0 + 500.0) / 2000.0).abs() < 1e-9);
    }

    #[test]
    fn floor_is_the_minimum_over_qualifying_cold_turns() {
        // Several qualifying cold turns: the smallest reading wins — it is
        // the wrapper span present on every turn. A writing cold turn is
        // excluded (it cached this conversation's own prefix).
        let cold_a = rec(0, Status::Complete, 128, 0);
        let cold_b = rec(0, Status::Complete, 64, 0);
        let mut writing = rec(0, Status::Complete, 4000, 0);
        writing.cache_written_tokens = 4200;
        assert_eq!(router_prefix_floor(&[cold_a, writing, cold_b]), 64);
    }

    #[test]
    fn no_cold_turn_means_no_floor() {
        // A session whose first observed turn already re-sent history gives
        // no evidence of a foreign prefix: 0, so nothing is netted.
        let warm = rec(1, Status::Complete, 628, 500);
        assert_eq!(router_prefix_floor(&[warm]), 0);
    }

    #[test]
    fn incomplete_cold_turn_does_not_set_the_floor() {
        // A partial cold turn's usage is not trustworthy token truth.
        let cold = rec(0, Status::Incomplete, 128, 0);
        let warm = rec(1, Status::Complete, 628, 500);
        assert_eq!(router_prefix_floor(&[cold, warm]), 0);
    }

    #[test]
    fn netting_clamps_at_zero() {
        // A turn whose reading exceeds its whole history (the impossible
        // case) and whose learned floor exceeds the reading still cannot go
        // negative: saturating subtraction reads 0% net, never negative.
        // An honest ≤100% turn keeps its raw rate even with a learned
        // floor — netting is for the impossible case only.
        let cold = rec(0, Status::Complete, 1280, 0);
        let _ = &cold;
        let honest = rec(1, Status::Complete, 90, 500);
        assert!((honest.hit_rate_net(1280).unwrap() - 0.18).abs() < 1e-9);
        let impossible = rec(1, Status::Complete, 640, 500);
        assert!((impossible.hit_rate_net(1280).unwrap() - 0.0).abs() < 1e-9);
    }
}
