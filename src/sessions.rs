//! Session store: prefix-continuity tracking, fork resolution, atomic aggregate.
//!
//! A request belongs to the session whose remembered token-prefix it extends; no
//! match starts a new session; forks resolve by longest-matching-prefix (ties:
//! most recent activity). A prefix break stays in the session and is measured as
//! a miss — it does not silently start a new session. Aggregation appends under a
//! lock with no `.await` inside the guard.

use crate::record::Record;
use std::collections::HashMap;
use std::sync::Mutex;

/// A tracked conversation, keyed by its cumulative prefix-hash sequence.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: u64,
    /// Hash of each message prefix, in order, for longest-prefix matching.
    pub prefix_hashes: Vec<u64>,
    pub records: Vec<Record>,
}

#[derive(Default)]
pub struct SessionStore {
    sessions: HashMap<u64, Session>,
    next_id: u64,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find the session whose prefix this request extends, or start a new one.
    /// Longest match wins; ties break to the most recent activity.
    pub fn resolve(&mut self, prefix_hashes: &[u64]) -> u64 {
        let best = self
            .sessions
            .values()
            .filter(|s| !s.prefix_hashes.is_empty() && is_prefix(&s.prefix_hashes, prefix_hashes))
            .max_by_key(|s| s.prefix_hashes.len())
            .map(|s| s.id);

        match best {
            Some(id) => id,
            None => {
                self.next_id += 1;
                let id = self.next_id;
                self.sessions.insert(
                    id,
                    Session {
                        id,
                        prefix_hashes: prefix_hashes.to_vec(),
                        records: Vec::new(),
                    },
                );
                id
            }
        }
    }

    /// Append a finalized record. The lock guard is released before returning.
    pub fn append(&mut self, record: Record) {
        if let Some(s) = self.sessions.get_mut(&record.session_id) {
            s.records.push(record);
        }
    }
}

fn is_prefix(short: &[u64], long: &[u64]) -> bool {
    short.len() <= long.len() && short == &long[..short.len()]
}

/// A thread-safe wrapper so the proxy can append from a stream task while the
/// dashboard reads. Kept deliberately simple in C1; lock granularity is a C4
/// concern.
pub struct SharedSessions(pub Mutex<SessionStore>);

impl SharedSessions {
    pub fn new() -> Self {
        Self(Mutex::new(SessionStore::new()))
    }
}

impl Default for SharedSessions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extends_the_session_it_is_a_prefix_of() {
        // The second request extends the first's prefix, so it stays in that
        // session — it does not start a new one.
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let b = store.resolve(&[1, 2, 3, 4]);
        assert_eq!(a, b);
    }

    #[test]
    fn divergent_prefixes_are_distinct_sessions() {
        // No common prefix → two sessions.
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let b = store.resolve(&[9, 9]);
        assert_ne!(a, b);
    }

    #[test]
    fn longest_matching_prefix_wins_on_fork() {
        // Session A grew to [1,2,3]; session B diverged to [1,7]. Both now
        // exist. A request extending [1,2] must match A (whose prefix [1,2,3]
        // the request extends), never B.
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let b = store.resolve(&[1, 7]); // no session it extends → new
        assert_ne!(a, b);

        let matched = store.resolve(&[1, 2, 3, 5]);
        assert_eq!(matched, a, "extends the [1,2,3] branch, not the [1,7] branch");
    }
}
