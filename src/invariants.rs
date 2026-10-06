//! Runtime invariant checks: the alignment between what the proxy claims
//! and what it did, verified at the moment each claim is made.
//!
//! A violation never blocks a request — measurement must not break serving
//! — but it is counted here, logged (metadata only), and exposed on the
//! dashboard and a text endpoint so alignment is a number, not an
//! assertion. Every check corresponds to a defect class the review
//! process once caught late; a check firing means that class regressed.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

static VIOLATIONS: LazyLock<Mutex<HashMap<&'static str, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record a violation of a named invariant. Idempotent per occurrence.
pub fn note(check: &'static str) {
    let mut map = VIOLATIONS.lock().unwrap();
    *map.entry(check).or_insert(0) += 1;
    tracing::warn!(
        target: "cachemax_invariants",
        check,
        "invariant violated: claims and behavior disagree"
    );
}

/// The current counts, sorted by check name for a stable surface.
pub fn snapshot() -> Vec<(&'static str, u64)> {
    let map = VIOLATIONS.lock().unwrap();
    let mut out: Vec<(&'static str, u64)> = map.iter().map(|(k, v)| (*k, *v)).collect();
    out.sort_unstable();
    out
}

/// Total violations across checks — the dashboard's single alignment
/// number.
pub fn total() -> u64 {
    VIOLATIONS.lock().unwrap().values().sum()
}

/// Prometheus-style text: one line per check, zero-state emits nothing (a
/// scrape that sees no lines has no violations — absence is the healthy
/// signal, and a `total` line would break that).
pub fn render_prom() -> String {
    let mut out = String::new();
    for (check, count) in snapshot() {
        out.push_str(&format!(
            "cachemax_invariant_violations{{check=\"{check}\"}} {count}\n"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_and_prom_render_stably() {
        note("test_check_a");
        note("test_check_a");
        note("test_check_b");
        let snap = snapshot();
        let a = snap.iter().find(|(k, _)| *k == "test_check_a");
        assert_eq!(a.map(|(_, v)| *v), Some(2));
        let text = render_prom();
        assert!(text.contains("check=\"test_check_a\"} 2"));
        assert!(text.contains("check=\"test_check_b\"} 1"));
    }
}
