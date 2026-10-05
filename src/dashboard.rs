//! Dashboard (C6): single-file, single-screen, no navigation. Hero-first:
//! status strip, hero band, session view (42%), prefix tape (58%). All color,
//! type, and spacing values come from DESIGN.md tokens — no ad-hoc palette.
//!
//! This module owns two things: the **view model** (records → a serializable
//! snapshot the page polls) and the **embedded HTML** (a single file served at
//! `/`). The page polls `/api/state` every ~500 ms.

use crate::record::{Record, SourceLabel, Status};
use serde::Serialize;

/// Which backend the hero weights toward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cloud,
    Local,
}

impl Backend {
    /// The local path is the engines that expose no cache truth or measure TTFT;
    /// everything provider-reported is the cloud path.
    pub fn from_source(source: SourceLabel) -> Self {
        match source {
            SourceLabel::ProviderReported => Backend::Cloud,
            SourceLabel::EngineMeasured | SourceLabel::NoCacheTruth => Backend::Local,
        }
    }
}

/// The unexposed-value token. Rendered as `—`, never `0`.
pub const UNEXPOSED: &str = "—";

/// The tape cell state for one prefix segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TapeState {
    Hit,
    Resent,
    Cold,
    Miss,
    Break,
    Incomplete,
    /// The resent history drifted from the canonical chain (repairable
    /// flavor) and went out untouched: a dry-run annotation, or repair
    /// declined. Muted — a finding, not an action.
    Drift,
    /// Repair rewrote the history and the provider re-served the cache:
    /// the instrument's own action, marked in amber (DESIGN.md C5
    /// amendment — recovered cache-served data).
    Repaired,
    /// Drift repair refused to touch: hard stop or semantic inequality.
    Unrepairable,
}

impl TapeState {
    /// The canonical glyph (DESIGN.md `tape-cell`). Color-independent.
    pub fn glyph(self) -> &'static str {
        match self {
            TapeState::Hit => "█",
            TapeState::Resent => "▓",
            TapeState::Cold => "░",
            TapeState::Miss => "▚",
            TapeState::Break => "┊",
            TapeState::Incomplete => "?",
            TapeState::Drift => "~",
            TapeState::Repaired => "◆",
            TapeState::Unrepairable => "!",
        }
    }
}

/// A hero figure formatted for display, honoring `—` for unexposed values.
pub fn format_pct(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.0}%", x * 100.0),
        None => UNEXPOSED.to_string(),
    }
}

/// Group digits with commas, e.g. `1550` → `1,550`.
pub fn format_tokens(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

/// USD with two decimals, or `—`.
pub fn format_usd(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("${:.2}", x),
        None => UNEXPOSED.to_string(),
    }
}

/// The provenance tag for the session's figures: the source shared by the
/// session's measured turns. A session whose turns disagree (e.g. one early
/// response omitted the provider's cache field) is tagged by the most frequent
/// source among complete turns, so a single field-less turn cannot mislabel a
/// cloud session as local.
pub fn provenance(records: &[Record]) -> SourceLabel {
    let mut counts: [(SourceLabel, usize); 3] = [
        (SourceLabel::ProviderReported, 0),
        (SourceLabel::EngineMeasured, 0),
        (SourceLabel::NoCacheTruth, 0),
    ];
    for r in records.iter().filter(|r| r.status == Status::Complete) {
        if let Some(slot) = counts.iter_mut().find(|(s, _)| *s == r.source) {
            slot.1 += 1;
        }
    }
    counts
        .iter()
        .filter(|(_, n)| *n > 0)
        // On a count tie, prefer a source that actually exposes cache truth
        // (provider-reported over engine-measured over none), so a single
        // field-less turn cannot drag a cloud session to `no_cache_truth`.
        .max_by_key(|(s, n)| (*n, source_rank(*s)))
        .map(|(s, _)| *s)
        .unwrap_or(SourceLabel::NoCacheTruth)
}

/// Tie-break priority: higher wins. Cloud (provider-reported) is the most
/// specific provenance; `no_cache_truth` is the least.
fn source_rank(s: SourceLabel) -> u8 {
    match s {
        SourceLabel::ProviderReported => 2,
        SourceLabel::EngineMeasured => 1,
        SourceLabel::NoCacheTruth => 0,
    }
}

/// One row of the session table.
#[derive(Debug, Clone, Serialize)]
pub struct TurnRow {
    pub turn: u32,
    /// Formatted hit rate, or `—` for cold/incomplete.
    pub hit: String,
    /// Formatted `cached / resent history`, or `— / —`.
    pub cached_over_history: String,
    /// Formatted cost, or `—`.
    pub cost: String,
    pub cold: bool,
    pub incomplete: bool,
    /// Dry-run drift annotation, when the turn was examined and drifted:
    /// a short tag naming the drift flavor (`tool_args`, `whitespace`,
    /// `truncated`, `reshaped`, `mixed`) or `unrepairable` /
    /// `first-turn` when the chain could not be extended at all.
    pub drift: Option<String>,
    /// At-risk tokens behind the drift annotation, formatted (`1,2xx tk`),
    /// when the turn was examined and drifted.
    pub drift_tokens: Option<String>,
    /// True when repair rewrote this turn's history (`on` mode). The
    /// annotation then names the action, not the finding.
    pub repaired: bool,
}

/// The cumulative summary row.
#[derive(Debug, Clone, Serialize)]
pub struct CumulativeRow {
    pub hit: String,
    pub cached_over_history: String,
    pub cost: String,
}

/// The tape row for one turn: the ordered segment states.
#[derive(Debug, Clone, Serialize)]
pub struct TapeRow {
    pub turn: u32,
    /// Each segment's state; the template maps state → glyph.
    pub cells: Vec<TapeState>,
    pub incomplete: bool,
}

/// The full serializable dashboard state the page polls.
#[derive(Debug, Clone, Serialize)]
pub struct DashboardState {
    pub backend: Backend,
    pub live: bool,
    pub tape_mode: String,
    pub incomplete_count: usize,
    pub session_count: usize,
    /// The session this view summarizes (0 when empty).
    pub session_id: u64,
    /// Cloud hero: cost saved (formatted).
    pub cost_saved: String,
    /// Cloud hero: billed input tokens (formatted).
    pub billed_input: String,
    /// Cloud hero: cache-served tokens (formatted).
    pub cache_served: String,
    /// Headline hit rate (formatted).
    pub hit_rate: String,
    pub provenance: String,
    /// Foreign cached tokens the endpoint reports on a cold turn (its own
    /// wrapper's prefix). Applied to a rate only when that rate reads above
    /// 100% (the numerator carries a span the denominator cannot); token
    /// counts stay provider-raw. `0` when no cold turn exposes a wrapper.
    pub router_prefix_tokens: u64,
    /// Whether any turn actually needed the floor — the disclosure renders
    /// only then, never for a direct provider's honest ≤100% rates.
    pub router_netted: bool,
    /// Anthropic's write/read split, when the session exposes one (secondary to
    /// the binding hit rate). `None` for providers that report no write count.
    pub write_split: Option<WriteSplit>,
    /// Cache-served tokens on repaired turns (`on` mode): what the rewrite
    /// let the provider re-serve. `None` when nothing was repaired — the
    /// line does not exist rather than reading zero.
    pub recovered: Option<String>,
    /// The cold→warm transition always shown beside the hero.
    pub transition: Transition,
    pub turns: Vec<TurnRow>,
    pub cumulative: CumulativeRow,
    pub tape: Vec<TapeRow>,
}

/// Anthropic's cache economics, shown beside the binding hit rate:
/// `cache_creation` (write premium) vs `cache_read` (read discount), plus the
/// derived `read/(read+creation)` share.
#[derive(Debug, Clone, Serialize)]
pub struct WriteSplit {
    pub written: String,
    pub read: String,
    pub derived: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Transition {
    pub resent_history: String,
    pub cached: String,
    pub reuse: String,
}

/// Build the dashboard view for one session's records.
pub fn view(records: &[Record], live: bool, session_count: usize) -> DashboardState {
    let source = provenance(records);
    let backend = Backend::from_source(source);
    let incomplete_count = records
        .iter()
        .filter(|r| r.status == Status::Incomplete)
        .count();

    // Complete turns (turn ≥ 1) drive the numbers; turn 0 and incomplete are
    // shown in the table but excluded from the formula. Turns with no cache
    // truth or a zero denominator contribute nothing (matching
    // `record::cumulative_hit_rate_net`), so the hero cannot be inflated by them.
    let contributes = |r: &&Record| {
        r.status == Status::Complete
            && r.turn >= 1
            && r.resent_history_tokens > 0
            && r.source != SourceLabel::NoCacheTruth
    };
    let complete: Vec<&Record> = records.iter().filter(contributes).collect();

    // Routed endpoints report their own wrapper's cached tokens inside
    // every `cached_tokens` (visible as a cold turn reading >0); the
    // re-sent-history denominator never contains that span, so raw rates
    // can exceed 100%. The floor is learned from the session's cold turn
    // and subtracted from every rate the dashboard derives. The token
    // counts stay provider-raw; only the derived rates are netted.
    let floor = crate::record::router_prefix_floor(records);

    let cached_sum: u64 = complete.iter().map(|r| r.cached_tokens).sum();
    let history_sum: u64 = complete.iter().map(|r| r.resent_history_tokens).sum();
    let net_hit = crate::record::cumulative_hit_rate_net(records, floor);
    // Recovered by repair: cache-served tokens on turns whose history was
    // rewritten (`on` mode). The provider reported these figures — the
    // rewrite is what let the request read a warm prefix at all.
    let recovered_sum: u64 = complete
        .iter()
        .filter(|r| r.repaired)
        .map(|r| r.cached_tokens)
        .sum();
    // Billed and cost follow the same "complete records only" rule the page
    // states: an incomplete turn's partial usage is excluded, as it is from the
    // hit rate. (Turn 0 is complete but carries no history; its own fresh input
    // still bills, so it is included in billed/cost, matching the counterfactual.)
    let counted: Vec<&Record> = records
        .iter()
        .filter(|r| r.status == Status::Complete)
        .collect();
    let billed_sum: u64 = counted.iter().map(|r| r.billed_input_tokens).sum();
    let cost_saved_sum: f64 = counted.iter().filter_map(|r| r.cost_saved_usd).sum();
    let cost_sum: f64 = counted.iter().filter_map(|r| r.cost_usd).sum();
    let has_cost = counted.iter().any(|r| r.cost_usd.is_some());

    // The transition uses the most recent warm turn.
    let last_warm = complete.last();
    let transition = Transition {
        resent_history: last_warm
            .map(|r| format!("{} tk", format_tokens(r.resent_history_tokens)))
            .unwrap_or_else(|| UNEXPOSED.to_string()),
        cached: last_warm
            .map(|r| format!("{} tk", format_tokens(r.cached_tokens)))
            .unwrap_or_else(|| UNEXPOSED.to_string()),
        reuse: last_warm
            .map(|r| format_pct(r.hit_rate_net(floor)))
            .unwrap_or_else(|| UNEXPOSED.to_string()),
    };

    let turns = records.iter().map(|r| turn_row(r, floor)).collect();
    let cumulative = CumulativeRow {
        hit: format_pct(net_hit),
        // Token counts stay provider-raw; the netted rate beside them is
        // disclosed by the session tag ("router prefix N tk subtracted").
        cached_over_history: format!(
            "{} / {}",
            format_tokens(cached_sum),
            format_tokens(history_sum)
        ),
        cost: if has_cost {
            format_usd(Some(cost_sum))
        } else {
            UNEXPOSED.to_string()
        },
    };

    let tape = records.iter().map(|r| tape_row(r, floor)).collect();

    // Anthropic's write/read split, when the provider exposes writes. Derived
    // rate = read / (read + creation), cumulative over complete turns.
    let written_sum: u64 = counted.iter().map(|r| r.cache_written_tokens).sum();
    // Net of the foreign-prefix floor under the same per-turn gate as every
    // rate: only a turn whose numerator exceeds its own history carries the
    // wrapper's span; honest turns keep their raw reads.
    let read_sum: u64 = counted
        .iter()
        .map(|r| {
            if r.cached_tokens > r.resent_history_tokens {
                r.cached_tokens.saturating_sub(floor)
            } else {
                r.cached_tokens
            }
        })
        .sum();
    let write_split = if written_sum > 0 {
        let total = read_sum + written_sum;
        Some(WriteSplit {
            written: format!("{} tk", format_tokens(written_sum)),
            read: format!("{} tk", format_tokens(read_sum)),
            derived: format_pct(if total == 0 {
                None
            } else {
                Some(read_sum as f64 / total as f64)
            }),
        })
    } else {
        None
    };

    DashboardState {
        backend,
        live,
        tape_mode: "hash".to_string(),
        incomplete_count,
        session_count,
        session_id: records.first().map(|r| r.session_id).unwrap_or(0),
        cost_saved: if has_cost {
            format_usd(Some(cost_saved_sum))
        } else {
            UNEXPOSED.to_string()
        },
        billed_input: format!("{} tk", format_tokens(billed_sum)),
        cache_served: format!("{} tk", format_tokens(cached_sum)),
        hit_rate: format_pct(net_hit),
        provenance: source_tag(source).to_string(),
        router_prefix_tokens: floor,
        // The disclosure renders only when netting actually applied: a
        // learned floor with no turn above 100% subtracted nothing, and
        // saying otherwise would describe a direct provider as routed.
        router_netted: floor > 0
            && records.iter().any(|r| {
                r.status == Status::Complete
                    && r.turn >= 1
                    && r.resent_history_tokens > 0
                    && r.source != SourceLabel::NoCacheTruth
                    && r.cached_tokens > r.resent_history_tokens
            }),
        write_split,
        recovered: (recovered_sum > 0).then(|| format!("{} tk", format_tokens(recovered_sum))),
        transition,
        turns,
        cumulative,
        tape,
    }
}

fn turn_row(r: &Record, floor: u64) -> TurnRow {
    let incomplete = r.status == Status::Incomplete;
    let cold = r.turn == 0;
    // The annotation. Mode off = not examined: no claim rendered. Clean
    // turns render nothing (the absence of drift is not news). A repaired
    // turn names the ACTION; everything else names the finding.
    let examined =
        r.repair_mode != crate::repair::RepairMode::Off && r.matches_canonical == Some(false);
    let (drift, drift_tokens) = if !examined {
        (None, None)
    } else {
        let tag = match r.drift_kind {
            Some(crate::repair::DriftKind::ToolArgReserialization) => "tool_args",
            Some(crate::repair::DriftKind::TextNormalization) => "whitespace",
            Some(crate::repair::DriftKind::TruncatedHistory) => "truncated",
            Some(crate::repair::DriftKind::RoleContentReshaped) => "reshaped",
            Some(crate::repair::DriftKind::Mixed) => "mixed",
            // No kind: either a hard stop (tokens at risk were quantified)
            // or nothing canonical to extend (first turn / model switch).
            None if r.canonicalized_tokens > 0 => "unrepairable",
            None => "first-turn",
        };
        (
            Some(tag.to_string()),
            Some(format!("{} tk", format_tokens(r.canonicalized_tokens))),
        )
    };
    TurnRow {
        turn: r.turn,
        hit: if cold || incomplete {
            UNEXPOSED.to_string()
        } else {
            format_pct(r.hit_rate_net(floor))
        },
        cached_over_history: if cold || incomplete {
            format!("{UNEXPOSED} / {UNEXPOSED}")
        } else {
            format!(
                "{} / {}",
                format_tokens(r.cached_tokens),
                format_tokens(r.resent_history_tokens)
            )
        },
        cost: if incomplete {
            UNEXPOSED.to_string()
        } else {
            format_usd(r.cost_usd)
        },
        cold,
        incomplete,
        drift,
        drift_tokens,
        repaired: r.repaired,
    }
}

/// Render a turn's prefix tape. The cell count is proportional to the re-sent
/// history (capped for layout); hit cells are the cached fraction, resent the
/// remainder, cold is a full cold run, incomplete its own glyph.
fn tape_row(r: &Record, floor: u64) -> TapeRow {
    const CELLS: usize = 16;
    let incomplete = r.status == Status::Incomplete;
    let mut cells = if incomplete {
        vec![TapeState::Incomplete; CELLS / 2]
    } else if r.turn == 0 || r.resent_history_tokens == 0 {
        vec![TapeState::Cold; CELLS]
    } else {
        // Exactly the per-turn rate shown in the same row (floor applied
        // only above 100%) — the tape and the number beside it must
        // describe the same fraction.
        let hit = (r.hit_rate_net(floor).unwrap_or(0.0) * CELLS as f64)
            .round()
            .clamp(0.0, CELLS as f64) as usize;
        let mut v = vec![TapeState::Hit; hit];
        v.extend(std::iter::repeat_n(TapeState::Resent, CELLS - hit));
        v
    };
    // A prefix break is a leading miss: the tape records where the prefix
    // diverged from the tracked session, ahead of this turn's hit/resent run.
    if r.broke_prefix && !incomplete {
        cells.insert(0, TapeState::Break);
        cells.insert(1, TapeState::Miss);
    }
    // The drift annotation leads the tape the same way: `◆` when repair
    // rewrote the turn (the action, in amber), `~` when drift went out
    // untouched (repairable flavor), `!` when repair would refuse it. All
    // are glyph-legible without color (DESIGN.md tape-cell rule).
    if !incomplete && r.repair_mode != crate::repair::RepairMode::Off {
        let state = if r.repaired {
            Some(TapeState::Repaired)
        } else if r.matches_canonical == Some(false) {
            Some(if r.drift_kind.is_some() {
                TapeState::Drift
            } else {
                TapeState::Unrepairable
            })
        } else {
            None
        };
        if let Some(state) = state {
            cells.insert(0, state);
        }
    }
    TapeRow {
        turn: r.turn,
        cells,
        incomplete,
    }
}

fn source_tag(source: SourceLabel) -> &'static str {
    match source {
        SourceLabel::ProviderReported => "provider_reported",
        SourceLabel::EngineMeasured => "engine_measured",
        SourceLabel::NoCacheTruth => "no_cache_truth",
    }
}

/// The embedded dashboard page (single file, served at `/`).
pub const DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(turn: u32, cached: u64, history: u64) -> Record {
        Record {
            session_id: 1,
            turn,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: cached,
            cache_written_tokens: 0,
            resent_history_tokens: history,
            billed_input_tokens: history + 200,
            broke_prefix: false,
            cost_usd: Some(0.01),
            cost_saved_usd: Some(0.005),
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
            breakpoint_count: None,
        }
    }

    #[test]
    fn unexposed_renders_as_dash_never_zero() {
        assert_eq!(format_pct(None), "—");
        assert_eq!(format_usd(None), "—");
    }

    #[test]
    fn tape_cells_follow_the_same_gate_as_the_rate() {
        // The tape and the per-turn number describe the same fraction: a
        // ≤100% turn renders raw (no floor applied), an impossible turn
        // renders net of the learned floor.
        let honest = rec(1, 300, 500); // 60% raw and net: 10 hit cells
        let impossible = rec(1, 560, 500); // raw 112% -> net (560-128)/500 = 86.4% -> 14 cells
        let hit_cells = |r: &Record, floor: u64| {
            tape_row(r, floor)
                .cells
                .iter()
                .filter(|c| **c == TapeState::Hit)
                .count()
        };
        assert_eq!(hit_cells(&honest, 128), 10, "honest rate: raw fraction");
        assert_eq!(hit_cells(&impossible, 128), 14, "impossible rate: netted");
    }

    #[test]
    fn mlxlm_no_cache_truth_shows_dash_not_zero_percent() {
        // mlx-lm exposes no cache truth; its rate must be `—`, never a measured
        // `0%` (spec: unexposed fields render `—`).
        let mut r = rec(1, 0, 1550);
        r.source = SourceLabel::NoCacheTruth;
        assert_eq!(r.hit_rate_net(0), None);
        assert_eq!(format_pct(r.hit_rate_net(0)), "—");
        assert_eq!(turn_row(&r, 0).hit, "—");
    }

    #[test]
    fn zero_denominator_turn_does_not_inflate_the_cumulative() {
        // A complete turn with no re-sent history must contribute nothing to
        // the session rate (was: added cached to numerator over a 0 denominator).
        let rs = vec![rec(1, 100, 1000), rec(2, 900, 0)];
        assert_eq!(
            format_pct(crate::record::cumulative_hit_rate_net(&rs, 0)),
            format_pct(Some(0.1))
        );
    }

    #[test]
    fn provenance_tie_prefers_cache_truth_over_no_cache_truth() {
        // One field-less turn + one provider turn must not tag a cloud session
        // as local.
        let mut blank = rec(1, 0, 100);
        blank.source = SourceLabel::NoCacheTruth;
        let reported = rec(2, 50, 100);
        let rs = vec![blank, reported];
        assert_eq!(provenance(&rs), SourceLabel::ProviderReported);
    }

    #[test]
    fn a_routed_session_never_reads_above_one_hundred_percent() {
        // A cold turn that already read 128 cached tokens reveals the
        // endpoint's own wrapper prefix. The warm turn's raw reading
        // (628/500 = 125.6%) must be netted down, per turn and cumulatively,
        // because the denominator never contains the wrapper span.
        let rs = vec![rec(0, 128, 0), rec(1, 628, 500)];
        let raw = 628.0 / 500.0;
        assert!(raw > 1.0, "the artifact this guards against");

        let state = view(&rs, false, 1);
        assert_eq!(state.router_prefix_tokens, 128, "floor is disclosed");
        assert_eq!(
            state.turns[1].hit,
            format_pct(Some((628.0 - 128.0) / 500.0)),
            "per-turn rate is net of the wrapper"
        );
        assert_eq!(state.hit_rate, format_pct(Some((628.0 - 128.0) / 500.0)));
        assert_eq!(state.cumulative.hit, state.hit_rate);
    }

    #[test]
    fn an_unrouted_session_is_untouched_by_netting() {
        // No cold-turn reading → no floor → every figure identical to the
        // raw formula. Direct providers must not be adjusted at all.
        let rs = vec![rec(0, 0, 0), rec(1, 1020, 1550)];
        let state = view(&rs, false, 1);
        assert_eq!(state.router_prefix_tokens, 0);
        assert_eq!(
            state.turns[1].hit,
            format_pct(Some(1020.0 / 1550.0)),
            "raw rate, nothing subtracted"
        );
    }

    #[test]
    fn tokens_group_by_thousands() {
        assert_eq!(format_tokens(1550), "1,550");
        assert_eq!(format_tokens(3880), "3,880");
        assert_eq!(format_tokens(0), "0");
    }

    #[test]
    fn cold_turn_shows_dashes() {
        let mut r = rec(0, 0, 0);
        r.cost_usd = None;
        let row = turn_row(&r, 0);
        assert!(row.cold);
        assert_eq!(row.hit, "—");
        assert_eq!(row.cached_over_history, "— / —");
    }

    #[test]
    fn a_broken_prefix_leads_the_tape_with_break_and_miss() {
        let mut r = rec(3, 100, 200);
        r.broke_prefix = true;
        let row = tape_row(&r, 0);
        assert_eq!(row.cells[0], TapeState::Break, "break glyph leads the tape");
        assert_eq!(
            row.cells[1],
            TapeState::Miss,
            "miss glyph follows the break"
        );
    }

    #[test]
    fn incomplete_turn_shows_its_own_glyph_and_is_excluded() {
        let mut r = rec(3, 0, 0);
        r.status = Status::Incomplete;
        r.cost_usd = None;
        let row = turn_row(&r, 0);
        assert!(row.incomplete);
        assert_eq!(row.hit, "—");
        let t = tape_row(&r, 0);
        assert!(t.cells.iter().all(|c| *c == TapeState::Incomplete));
    }

    #[test]
    fn tape_cells_are_glyph_distinguishable() {
        // Every state must have a distinct glyph (legible without color).
        let states = [
            TapeState::Hit,
            TapeState::Resent,
            TapeState::Cold,
            TapeState::Miss,
            TapeState::Break,
            TapeState::Incomplete,
            TapeState::Drift,
            TapeState::Repaired,
            TapeState::Unrepairable,
        ];
        let mut glyphs: Vec<&str> = states.iter().map(|s| s.glyph()).collect();
        glyphs.sort();
        glyphs.dedup();
        assert_eq!(
            glyphs.len(),
            states.len(),
            "each state needs a unique glyph"
        );
        assert_eq!(TapeState::Repaired.glyph(), "◆");
    }

    /// A record examined in dry-run with the given drift claim.
    fn drifted(turn: u32, kind: Option<crate::repair::DriftKind>, tokens: u64) -> Record {
        let mut r = rec(turn, 100, 200);
        r.repair_mode = crate::repair::RepairMode::DryRun;
        r.matches_canonical = Some(false);
        r.drift_kind = kind;
        r.canonicalized_tokens = tokens;
        r
    }

    #[test]
    fn a_drifted_turn_row_carries_the_annotation_tag() {
        let r = drifted(
            3,
            Some(crate::repair::DriftKind::ToolArgReserialization),
            312,
        );
        let row = turn_row(&r, 0);
        assert_eq!(row.drift.as_deref(), Some("tool_args"));
        assert_eq!(row.drift_tokens.as_deref(), Some("312 tk"));
        // And the tape leads with the drift glyph.
        let tape = tape_row(&r, 0);
        assert_eq!(tape.cells[0], TapeState::Drift);
    }

    #[test]
    fn an_unrepairable_turn_is_tagged_and_banged() {
        // No kind + tokens at risk = the hard-stop / semantic-inequality case.
        let r = drifted(2, None, 900);
        let row = turn_row(&r, 0);
        assert_eq!(row.drift.as_deref(), Some("unrepairable"));
        let tape = tape_row(&r, 0);
        assert_eq!(tape.cells[0], TapeState::Unrepairable);
    }

    #[test]
    fn a_first_turn_or_model_switch_carries_no_at_risk_claim() {
        let r = drifted(1, None, 0);
        assert_eq!(turn_row(&r, 0).drift.as_deref(), Some("first-turn"));
        assert_eq!(turn_row(&r, 0).drift_tokens.as_deref(), Some("0 tk"));
    }

    #[test]
    fn a_repaired_turn_names_the_action_and_leads_the_tape_in_amber() {
        let mut r = drifted(
            3,
            Some(crate::repair::DriftKind::ToolArgReserialization),
            312,
        );
        r.repaired = true;
        r.cached_tokens = 800;
        let row = turn_row(&r, 0);
        assert!(row.repaired);
        assert_eq!(row.drift.as_deref(), Some("tool_args"));
        let tape = tape_row(&r, 0);
        assert_eq!(
            tape.cells[0],
            TapeState::Repaired,
            "the action glyph leads the tape"
        );
    }

    #[test]
    fn the_recovered_line_exists_only_when_something_was_repaired() {
        // Two repaired turns with real cache reads...
        let repaired = |turn: u32| {
            let mut r = rec(turn, 900, 1000);
            r.repair_mode = crate::repair::RepairMode::On;
            r.repaired = true;
            r
        };
        let with = view(&[repaired(1), repaired(2)], true, 1);
        assert_eq!(with.recovered.as_deref(), Some("1,800 tk"));
        // ...and none: the line does not exist, it does not read zero.
        let without = view(&[rec(1, 900, 1000)], true, 1);
        assert_eq!(without.recovered, None);
        // An incomplete repaired turn contributes nothing (aggregates are
        // complete-turns only).
        let mut inc = repaired(3);
        inc.status = Status::Incomplete;
        let partial = view(&[repaired(1), inc], true, 1);
        assert_eq!(partial.recovered.as_deref(), Some("900 tk"));
    }

    #[test]
    fn clean_and_unexamined_turns_carry_no_annotation() {
        // A clean match: nothing to say.
        let mut clean = rec(1, 100, 200);
        clean.repair_mode = crate::repair::RepairMode::DryRun;
        clean.matches_canonical = Some(true);
        assert!(turn_row(&clean, 0).drift.is_none());
        // Unexamined (mode off): no claim at all.
        let off = rec(2, 100, 200);
        assert!(turn_row(&off, 0).drift.is_none());
        assert_eq!(off.matches_canonical, None);
    }

    #[test]
    fn view_marks_incomplete_and_caps_the_count() {
        let rs = vec![rec(1, 1000, 2000), {
            let mut r = rec(2, 0, 0);
            r.status = Status::Incomplete;
            r
        }];
        let v = view(&rs, true, 1);
        assert_eq!(v.incomplete_count, 1);
        assert_eq!(v.hit_rate, "50%");
        assert_eq!(v.session_count, 1);
    }
}
