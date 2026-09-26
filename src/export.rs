//! Observability (C7): metrics-only JSONL export and structured finalize logs.
//! One artifact, two triggers — the CLI writes a file, the dashboard downloads
//! the same schema. Bodies are never written unless the body opt-in is set.

use crate::record::Record;

/// Serialize records as one-JSON-object-per-line. Metrics only: no message
/// content, ever.
pub fn to_jsonl(records: &[Record]) -> Result<String, serde_json::Error> {
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    Ok(out)
}

/// Default export filename for a session: `./cachemax-<session>.jsonl`.
pub fn default_path(session_id: u64) -> String {
    format!("./cachemax-{session_id}.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Record, SourceLabel, Status};

    #[test]
    fn jsonl_is_one_object_per_line_and_round_trips() {
        let r = Record {
            session_id: 7,
            turn: 1,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: 1020,
            resent_history_tokens: 1550,
            billed_input_tokens: 1750,
            cache_written_tokens: 0,
            cost_usd: Some(0.02),
            cost_saved_usd: Some(0.01),
        };
        let s = to_jsonl(std::slice::from_ref(&r)).unwrap();
        assert_eq!(s.lines().count(), 1);
        let back: Record = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(back.cached_tokens, 1020);
    }

    #[test]
    fn default_path_names_the_session() {
        assert_eq!(default_path(7), "./cachemax-7.jsonl");
    }
}
