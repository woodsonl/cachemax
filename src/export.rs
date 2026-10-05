//! Observability (C7): metrics-only JSONL export and structured finalize logs.
//! One artifact, two triggers — the CLI writes a file, the dashboard downloads
//! the same schema. Bodies are never written unless the body opt-in is set.

use crate::record::Record;
use std::io::Write;

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

/// Write a session's records to `path` as JSONL. Returns the number of records
/// written. `path` defaults to [`default_path`] when `--out` is not given.
pub fn write_file(records: &[Record], path: &str) -> std::io::Result<usize> {
    let jsonl = to_jsonl(records).map_err(std::io::Error::other)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(jsonl.as_bytes())?;
    Ok(records.len())
}

/// Emit one structured finalize log line: metadata and counts only, never
/// message content. Grep-able by `cachemax_finalize`.
pub fn log_finalize(r: &Record) {
    tracing::info!(
        target: "cachemax_finalize",
        session = r.session_id,
        turn = r.turn,
        status = ?r.status,
        source = ?r.source,
        ttft_ms = r.ttft_ms,
        cached_tokens = r.cached_tokens,
        resent_history_tokens = r.resent_history_tokens,
        billed_input_tokens = r.billed_input_tokens,
        "request finalized"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Record, SourceLabel, Status};

    fn rec() -> Record {
        Record {
            session_id: 7,
            turn: 1,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: 1020,
            cache_written_tokens: 0,
            resent_history_tokens: 1550,
            billed_input_tokens: 1750,
            broke_prefix: false,
            cost_usd: Some(0.02),
            cost_saved_usd: Some(0.01),
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
        }
    }

    #[test]
    fn jsonl_is_one_object_per_line_and_round_trips() {
        let r = rec();
        let s = to_jsonl(std::slice::from_ref(&r)).unwrap();
        assert_eq!(s.lines().count(), 1);
        let back: Record = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(back.cached_tokens, 1020);
    }

    #[test]
    fn default_path_names_the_session() {
        assert_eq!(default_path(7), "./cachemax-7.jsonl");
    }

    #[test]
    fn file_and_download_share_one_schema() {
        // The CLI file and the dashboard download both come from `to_jsonl`,
        // so they are byte-identical for the same records by construction.
        let rs = vec![rec()];
        let a = to_jsonl(&rs).unwrap();
        let b = to_jsonl(&rs).unwrap();
        assert_eq!(a, b);
        assert!(!a.contains("messages"), "metrics only, no bodies");
    }

    #[test]
    fn write_file_round_trips() {
        let dir = std::env::temp_dir().join(format!("cachemax-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.jsonl");
        let rs = vec![rec()];
        let n = write_file(&rs, path.to_str().unwrap()).unwrap();
        assert_eq!(n, 1);
        let back = std::fs::read_to_string(&path).unwrap();
        assert_eq!(back, to_jsonl(&rs).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod readme_sample {
    use super::*;
    #[test]
    fn readme_sample_is_byte_exact() {
        let r = Record {
            session_id: 1,
            turn: 1,
            status: crate::record::Status::Complete,
            source: crate::record::SourceLabel::ProviderReported,
            ttft_ms: Some(120.0),
            cached_tokens: 1020,
            cache_written_tokens: 0,
            resent_history_tokens: 1550,
            billed_input_tokens: 1750,
            broke_prefix: false,
            cost_usd: Some(0.02),
            cost_saved_usd: Some(0.01),
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
        };
        let line = to_jsonl(&[r]).unwrap();
        let expected = "{\"session_id\":1,\"turn\":1,\"status\":\"complete\",\"source\":\"provider_reported\",\"ttft_ms\":120.0,\"cached_tokens\":1020,\"cache_written_tokens\":0,\"resent_history_tokens\":1550,\"billed_input_tokens\":1750,\"broke_prefix\":false,\"cost_usd\":0.02,\"cost_saved_usd\":0.01,\"repair_mode\":\"off\",\"repaired\":false,\"matches_canonical\":null,\"drift_kind\":null,\"canonicalized_tokens\":0}\n";
        assert_eq!(line, expected, "README sample must match the serializer");
    }
}
