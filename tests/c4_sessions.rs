//! C4 verification: interleaved-request atomicity under a real tokio interleave.
//!
//! Many tasks resolve and append concurrently through `SharedSessions`. The
//! aggregate must equal the serial expectation exactly: no lost appends, no
//! torn reads, no `.await` inside the lock guard (which would deadlock or
//! interleave). Each task owns one session, so the per-session counts are
//! deterministic.

use cachemax::record::{Record, SourceLabel, Status};
use cachemax::sessions::SharedSessions;
use std::sync::Arc;

fn record(session_id: u64, turn: u32) -> Record {
    Record {
        session_id,
        turn,
        status: Status::Complete,
        source: SourceLabel::ProviderReported,
        ttft_ms: Some(10.0),
        cached_tokens: 100,
        cache_written_tokens: 0,
        resent_history_tokens: 200,
        billed_input_tokens: 200,
        broke_prefix: false,
        cost_usd: None,
        cost_saved_usd: None,
        estimated_saved_usd: None,
        repair_mode: cachemax::repair::RepairMode::Off,
        repaired: false,
        matches_canonical: None,
        drift_kind: None,
        canonicalized_tokens: 0,
        breakpoint_count: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_appends_lose_nothing() {
    let store = Arc::new(SharedSessions::new());
    const TASKS: u64 = 32;
    const PER_TASK: u32 = 50;

    // Seed one session per task, then append concurrently.
    let mut ids = Vec::new();
    {
        let mut guard = store.0.lock().unwrap();
        for i in 0..TASKS {
            // Distinct prefixes so each task gets its own session.
            let prefix = vec![i + 1, i + 1000];
            let id = guard.resolve(&prefix).session_id;
            ids.push(id);
        }
    }

    let mut handles = Vec::new();
    for &session_id in ids.iter() {
        let store = store.clone();
        handles.push(tokio::spawn(async move {
            for turn in 1..=PER_TASK {
                // Yield between appends to force real interleaving.
                tokio::task::yield_now().await;
                store.0.lock().unwrap().append(record(session_id, turn));
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let guard = store.0.lock().unwrap();
    let mut total = 0;
    for &id in &ids {
        let s = guard.session(id).unwrap();
        assert_eq!(
            s.records.len(),
            PER_TASK as usize,
            "session {id} lost appends"
        );
        total += s.records.len();
    }
    assert_eq!(total, TASKS as usize * PER_TASK as usize);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resolve_and_append_interleave_without_torn_state() {
    let store = Arc::new(SharedSessions::new());

    // One long-lived session grows as it is re-resolved concurrently with
    // independent sessions being created.
    let grow = {
        let store = store.clone();
        tokio::spawn(async move {
            let id = {
                let mut guard = store.0.lock().unwrap();
                guard.resolve(&[1, 2, 3]).session_id
            };
            for turn in 1..=100 {
                tokio::task::yield_now().await;
                {
                    let mut g = store.0.lock().unwrap();
                    g.resolve(&[1, 2, 3, turn]); // extend
                    g.append(record(id, turn as u32));
                }
            }
            id
        })
    };

    let mut others = Vec::new();
    for i in 0..16u64 {
        let store = store.clone();
        others.push(tokio::spawn(async move {
            for _ in 0..20 {
                tokio::task::yield_now().await;
                {
                    let mut g = store.0.lock().unwrap();
                    let id = g.resolve(&[i + 5000, i + 6000]).session_id;
                    g.append(record(id, 1));
                }
            }
        }));
    }

    let main_id = grow.await.unwrap();
    for o in others {
        o.await.unwrap();
    }

    let guard = store.0.lock().unwrap();
    let s = guard.session(main_id).unwrap();
    assert_eq!(s.records.len(), 100, "the grown session kept every append");
}
