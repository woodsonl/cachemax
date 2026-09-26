//! Dashboard (C6): single-file, single-screen, no navigation. Hero-first:
//! status strip, hero band, session view (42%), prefix tape (58%). All color,
//! type, and spacing values come from DESIGN.md tokens — no ad-hoc palette.
//! The full HTML render lands with C6/T1-T4; this module owns the data model
//! the template binds to.

use crate::record::{cumulative_hit_rate, Record, SourceLabel};

/// Which backend the hero weights toward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cloud,
    Local,
}

/// The unexposed-value token. Rendered as `—`, never `0`.
pub const UNEXPOSED: &str = "—";

/// A hero figure formatted for display, honoring `—` for unexposed values.
pub fn format_pct(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.0}%", x * 100.0),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Status;

    fn rec(turn: u32, cached: u64, history: u64) -> Record {
        Record {
            session_id: 1,
            turn,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: cached,
            resent_history_tokens: history,
            billed_input_tokens: history + 200,
            cost_usd: None,
        }
    }

    #[test]
    fn unexposed_renders_as_dash_never_zero() {
        assert_eq!(format_pct(None), "—");
    }

    #[test]
    fn hero_matches_cumulative_formula() {
        let rs = vec![rec(1, 1020, 1550), rec(2, 860, 1810)];
        assert_eq!(hero_hit_rate(&rs), format_pct(Some(1880.0 / 3360.0)));
    }
}
