//! Dashboard (C6): single-file, single-screen, no navigation. Hero-first:
//! status strip, hero band, session view (42%), prefix tape (58%). All color,
//! type, and spacing values come from DESIGN.md tokens — no ad-hoc palette.
//!
//! This module owns two things: the **view model** (records → a serializable
//! snapshot the page polls) and the **embedded HTML** (a single file served at
//! `/`). The page polls `/api/state` every ~500 ms.

use crate::record::{cumulative_hit_rate, Record, SourceLabel, Status};
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

/// The hero's hit-rate figure for a session's records.
pub fn hero_hit_rate(records: &[Record]) -> String {
    format_pct(cumulative_hit_rate(records))
}

/// The provenance tag for the session's figures.
pub fn provenance(records: &[Record]) -> SourceLabel {
    records
        .first()
        .map(|r| r.source)
        .unwrap_or(SourceLabel::NoCacheTruth)
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
    /// Cloud hero: cost saved (formatted).
    pub cost_saved: String,
    /// Cloud hero: billed input tokens (formatted).
    pub billed_input: String,
    /// Cloud hero: cache-served tokens (formatted).
    pub cache_served: String,
    /// Headline hit rate (formatted).
    pub hit_rate: String,
    pub provenance: String,
    /// The cold→warm transition always shown beside the hero.
    pub transition: Transition,
    pub turns: Vec<TurnRow>,
    pub cumulative: CumulativeRow,
    pub tape: Vec<TapeRow>,
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
    // shown in the table but excluded from the formula.
    let complete: Vec<&Record> = records
        .iter()
        .filter(|r| r.status == Status::Complete && r.turn >= 1)
        .collect();

    let cached_sum: u64 = complete.iter().map(|r| r.cached_tokens).sum();
    let history_sum: u64 = complete.iter().map(|r| r.resent_history_tokens).sum();
    let billed_sum: u64 = records.iter().map(|r| r.billed_input_tokens).sum();
    let cost_saved_sum: f64 = records.iter().filter_map(|r| r.cost_saved_usd).sum();
    let cost_sum: f64 = records.iter().filter_map(|r| r.cost_usd).sum();
    let has_cost = records.iter().any(|r| r.cost_usd.is_some());

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
            .map(|r| format_pct(r.hit_rate()))
            .unwrap_or_else(|| UNEXPOSED.to_string()),
    };

    let turns = records.iter().map(turn_row).collect();
    let cumulative = CumulativeRow {
        hit: format_pct(if history_sum == 0 {
            None
        } else {
            Some(cached_sum as f64 / history_sum as f64)
        }),
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

    let tape = records.iter().map(tape_row).collect();

    DashboardState {
        backend,
        live,
        tape_mode: "hash".to_string(),
        incomplete_count,
        session_count,
        cost_saved: if has_cost {
            format_usd(Some(cost_saved_sum))
        } else {
            UNEXPOSED.to_string()
        },
        billed_input: format!("{} tk", format_tokens(billed_sum)),
        cache_served: format!("{} tk", format_tokens(cached_sum)),
        hit_rate: format_pct(if history_sum == 0 {
            None
        } else {
            Some(cached_sum as f64 / history_sum as f64)
        }),
        provenance: source_tag(source).to_string(),
        transition,
        turns,
        cumulative,
        tape,
    }
}

fn turn_row(r: &Record) -> TurnRow {
    let incomplete = r.status == Status::Incomplete;
    let cold = r.turn == 0;
    TurnRow {
        turn: r.turn,
        hit: if cold || incomplete {
            UNEXPOSED.to_string()
        } else {
            format_pct(r.hit_rate())
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
    }
}

/// Render a turn's prefix tape. The cell count is proportional to the re-sent
/// history (capped for layout); hit cells are the cached fraction, resent the
/// remainder, cold is a full cold run, incomplete its own glyph.
fn tape_row(r: &Record) -> TapeRow {
    const CELLS: usize = 16;
    let incomplete = r.status == Status::Incomplete;
    let cells = if incomplete {
        vec![TapeState::Incomplete; CELLS / 2]
    } else if r.turn == 0 || r.resent_history_tokens == 0 {
        vec![TapeState::Cold; CELLS]
    } else {
        let hit = ((r.cached_tokens as f64 / r.resent_history_tokens as f64) * CELLS as f64)
            .round()
            .clamp(0.0, CELLS as f64) as usize;
        let mut v = vec![TapeState::Hit; hit];
        v.extend(std::iter::repeat_n(TapeState::Resent, CELLS - hit));
        v
    };
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
            cost_usd: Some(0.01),
            cost_saved_usd: Some(0.005),
        }
    }

    #[test]
    fn unexposed_renders_as_dash_never_zero() {
        assert_eq!(format_pct(None), "—");
        assert_eq!(format_usd(None), "—");
    }

    #[test]
    fn hero_matches_cumulative_formula() {
        let rs = vec![rec(1, 1020, 1550), rec(2, 860, 1810)];
        assert_eq!(hero_hit_rate(&rs), format_pct(Some(1880.0 / 3360.0)));
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
        let row = turn_row(&r);
        assert!(row.cold);
        assert_eq!(row.hit, "—");
        assert_eq!(row.cached_over_history, "— / —");
    }

    #[test]
    fn incomplete_turn_shows_its_own_glyph_and_is_excluded() {
        let mut r = rec(3, 0, 0);
        r.status = Status::Incomplete;
        r.cost_usd = None;
        let row = turn_row(&r);
        assert!(row.incomplete);
        assert_eq!(row.hit, "—");
        let t = tape_row(&r);
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
        ];
        let mut glyphs: Vec<&str> = states.iter().map(|s| s.glyph()).collect();
        glyphs.sort();
        glyphs.dedup();
        assert_eq!(
            glyphs.len(),
            states.len(),
            "each state needs a unique glyph"
        );
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
