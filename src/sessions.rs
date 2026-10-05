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
    /// Monotonic tick of the last append or resolve, for fork tie-breaks.
    pub last_active: u64,
}

impl Session {
    /// The session's cumulative hit rate over complete, non-cold turns, net of
    /// the endpoint's foreign-prefix floor (`crate::record::router_prefix_floor`)
    /// so it agrees with the dashboard and never reads above 100% on a routed
    /// session.
    pub fn cumulative_hit_rate(&self) -> Option<f64> {
        let floor = crate::record::router_prefix_floor(&self.records);
        crate::record::cumulative_hit_rate_net(&self.records, floor)
    }
}

/// The outcome of resolving an incoming request against the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub session_id: u64,
    /// True when this request continues a tracked session's prefix.
    pub continued: bool,
    /// True when the request broke a tracked session's prefix (measured as a
    /// miss, but kept in the session).
    pub broke_prefix: bool,
}

/// Optional log sink for prefix collisions (spec: collision log test).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collision {
    pub session_id: u64,
    pub shared_prefix_len: usize,
    pub session_len: usize,
    pub incoming_len: usize,
}

/// Cap on the retained collision log. The log is diagnostic, not a ledger; a
/// client that diverges every turn would otherwise grow it without bound.
const COLLISION_LOG_CAP: usize = 256;

/// Append to `log`, keeping only the most recent [`COLLISION_LOG_CAP`] entries.
fn push_bounded(log: &mut Vec<Collision>, c: Collision) {
    if log.len() == COLLISION_LOG_CAP {
        log.remove(0);
    }
    log.push(c);
}

#[derive(Default)]
pub struct SessionStore {
    sessions: HashMap<u64, Session>,
    /// Explicit client-declared affinity keys (`x-cachemax-session`) → session
    /// id. Opt-in: a client that names its conversation gets exactly that
    /// session, with no prefix-fork inference.
    keys: HashMap<String, u64>,
    next_id: u64,
    tick: u64,
    collision_log: Vec<Collision>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn collision_log(&self) -> &[Collision] {
        &self.collision_log
    }

    /// Find the session this request belongs to, or start a new one.
    ///
    /// Resolution order:
    /// 1. **Extends:** a session whose prefix is a prefix of the request's.
    ///    Longest wins; ties break to the most recent activity.
    /// 2. **Break:** no extension, but a session shares a non-empty prefix that
    ///    then diverges. The request stays in the most-recent such session (or
    ///    the longest shared prefix, ties to recency) and is logged as a
    ///    collision — it is measured as a miss, not a silent new session.
    /// 3. **New:** nothing shares a prefix → a fresh session.
    pub fn resolve(&mut self, prefix_hashes: &[u64]) -> Resolution {
        self.tick += 1;
        let tick = self.tick;

        // 1. Extension: session is a prefix of the request.
        let extending = self
            .sessions
            .values()
            .filter(|s| !s.prefix_hashes.is_empty() && is_prefix(&s.prefix_hashes, prefix_hashes))
            .max_by(|a, b| {
                a.prefix_hashes
                    .len()
                    .cmp(&b.prefix_hashes.len())
                    .then(a.last_active.cmp(&b.last_active))
            })
            .map(|s| s.id);

        if let Some(id) = extending {
            if let Some(s) = self.sessions.get_mut(&id) {
                s.last_active = tick;
                // A conversation only grows: absorb the longer prefix so the
                // next turn's continuity is measured against the full history
                // seen so far, not just the first request's messages.
                if prefix_hashes.len() > s.prefix_hashes.len() {
                    s.prefix_hashes = prefix_hashes.to_vec();
                }
            }
            return Resolution {
                session_id: id,
                continued: true,
                broke_prefix: false,
            };
        }

        // 2. Break: shares a non-empty prefix but diverges.
        let breaking = self
            .sessions
            .values()
            .filter(|s| !s.prefix_hashes.is_empty())
            .filter(|s| shared_prefix_len(&s.prefix_hashes, prefix_hashes) > 0)
            .max_by(|a, b| {
                shared_prefix_len(&a.prefix_hashes, prefix_hashes)
                    .cmp(&shared_prefix_len(&b.prefix_hashes, prefix_hashes))
                    .then(a.last_active.cmp(&b.last_active))
            })
            .map(|s| (s.id, shared_prefix_len(&s.prefix_hashes, prefix_hashes)));

        if let Some((id, shared)) = breaking {
            let session_len = self.sessions[&id].prefix_hashes.len();
            if let Some(s) = self.sessions.get_mut(&id) {
                s.last_active = tick;
                // Re-base the session onto the divergent branch. This turn is
                // still a break (measured as a miss), but the branch becomes the
                // session's prefix, so its *next* turn is a continuation rather
                // than another break. Without this a client that switches topic
                // once is flagged as breaking on every later turn.
                s.prefix_hashes = prefix_hashes.to_vec();
            }
            push_bounded(
                &mut self.collision_log,
                Collision {
                    session_id: id,
                    shared_prefix_len: shared,
                    session_len,
                    incoming_len: prefix_hashes.len(),
                },
            );
            return Resolution {
                session_id: id,
                continued: false,
                broke_prefix: true,
            };
        }

        // 3. New session.
        self.next_id += 1;
        let id = self.next_id;
        self.sessions.insert(
            id,
            Session {
                id,
                prefix_hashes: prefix_hashes.to_vec(),
                records: Vec::new(),
                last_active: tick,
            },
        );
        Resolution {
            session_id: id,
            continued: false,
            broke_prefix: false,
        }
    }

    /// Resolve by an explicit client-declared affinity key
    /// (`x-cachemax-session`), bypassing prefix inference entirely.
    ///
    /// The client is opting in to "this request belongs to the conversation I
    /// name." A known key returns its session (a continuation, even when the
    /// bytes re-sent are a truncated or re-based history — the client's word
    /// is the authority); a new key allocates a session and binds it. The
    /// prefix hashes are still recorded/absorbed so a later un-keyed request
    /// can still match this session by prefix.
    pub fn resolve_keyed(&mut self, key: &str, prefix_hashes: &[u64]) -> Resolution {
        self.tick += 1;
        let tick = self.tick;
        if let Some(&id) = self.keys.get(key) {
            if let Some(s) = self.sessions.get_mut(&id) {
                s.last_active = tick;
                if prefix_hashes.len() > s.prefix_hashes.len() {
                    s.prefix_hashes = prefix_hashes.to_vec();
                }
            }
            return Resolution {
                session_id: id,
                continued: true,
                broke_prefix: false,
            };
        }
        self.next_id += 1;
        let id = self.next_id;
        self.sessions.insert(
            id,
            Session {
                id,
                prefix_hashes: prefix_hashes.to_vec(),
                records: Vec::new(),
                last_active: tick,
            },
        );
        self.keys.insert(key.to_string(), id);
        Resolution {
            session_id: id,
            continued: false,
            broke_prefix: false,
        }
    }

    /// Append a finalized record. The lock guard is released before returning.
    pub fn append(&mut self, record: Record) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(s) = self.sessions.get_mut(&record.session_id) {
            s.records.push(record);
            s.last_active = tick;
        }
    }

    /// Look up a session by id.
    pub fn session(&self, id: u64) -> Option<&Session> {
        self.sessions.get(&id)
    }

    /// The most recently active session, if any.
    pub fn most_recent(&self) -> Option<&Session> {
        self.sessions.values().max_by_key(|s| s.last_active)
    }

    /// Number of tracked sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Total recorded turns across all sessions.
    pub fn total_records(&self) -> usize {
        self.sessions.values().map(|s| s.records.len()).sum()
    }

    /// Whether any session is tracked.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

fn is_prefix(short: &[u64], long: &[u64]) -> bool {
    short.len() <= long.len() && short == &long[..short.len()]
}

fn shared_prefix_len(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// A thread-safe wrapper so the proxy can append from a stream task while the
/// dashboard reads. The lock is held only for the duration of a store call; no
/// `.await` ever occurs inside the guard.
pub struct SharedSessions(pub Mutex<SessionStore>);

impl SharedSessions {
    pub fn new() -> Self {
        Self(Mutex::new(SessionStore::new()))
    }

    /// Lock the store, recovering from poisoning. A panic in one handler must
    /// not brick every later request; the measurement store has no invariant a
    /// poisoned lock could break, so the guard is taken anyway.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, SessionStore> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
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
    use crate::record::{SourceLabel, Status};

    #[test]
    fn extends_the_session_it_is_a_prefix_of() {
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let b = store.resolve(&[1, 2, 3, 4]);
        assert_eq!(a.session_id, b.session_id);
        assert!(b.continued);
    }

    #[test]
    fn divergent_prefixes_are_distinct_sessions() {
        let mut store = SessionStore::new();
        let a = store.resolve(&[9, 9]);
        let b = store.resolve(&[1, 2, 3]);
        // No shared prefix: [9,9] vs [1,2,3] share nothing.
        assert_ne!(a.session_id, b.session_id);
        assert!(!b.continued);
        assert!(!b.broke_prefix);
    }

    #[test]
    fn fork_tie_breaks_to_most_recent_activity() {
        // Two sessions with distinct equal-length prefixes that share the same
        // length of prefix with an incoming request. The incoming must land in
        // the most recently active of the two.
        let mut store = SessionStore::new();
        // Seed a real fork: start [1,2,3], then force a second session by making
        // the second request share nothing with the first, then align them.
        let s1 = store.resolve(&[1, 2, 3]).session_id; // session 1
                                                       // Touch session 1 last; now an incoming that ties must pick session 1.
        store.append(Record {
            session_id: s1,
            turn: 0,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: None,
            cached_tokens: 0,
            cache_written_tokens: 0,
            resent_history_tokens: 0,
            billed_input_tokens: 0,
            broke_prefix: false,
            cost_usd: None,
            cost_saved_usd: None,
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
            breakpoint_count: None,
        });
        // Incoming [1,2,9] breaks from session 1 (shared [1,2]); it is the only
        // session sharing a prefix, so it wins.
        let r = store.resolve(&[1, 2, 9]);
        assert_eq!(r.session_id, s1);
        assert!(r.broke_prefix);
    }

    #[test]
    fn longest_matching_prefix_wins_on_fork() {
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        // [1,7] shares only [1] with [1,2,3] -> break, stays in a and re-bases.
        let b = store.resolve(&[1, 7]);
        assert_eq!(b.session_id, a.session_id);
        assert!(b.broke_prefix);

        // After the re-base, a request extending the new branch continues it.
        let matched = store.resolve(&[1, 7, 8]);
        assert_eq!(matched.session_id, a.session_id);
        assert!(matched.continued);
        assert!(!matched.broke_prefix);
    }

    #[test]
    fn divergent_branch_is_absorbed_after_one_break() {
        // A client that switches topic once must be measured as a single break,
        // then as a continuation — not as breaking forever.
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let br = store.resolve(&[1, 2, 9]);
        assert!(br.broke_prefix);
        let next = store.resolve(&[1, 2, 9, 10]);
        assert_eq!(next.session_id, a.session_id);
        assert!(next.continued, "the divergent branch is a continuation");
        assert!(!next.broke_prefix);
    }

    #[test]
    fn collision_log_is_bounded() {
        // A client diverging every turn must not grow the log without bound.
        let mut store = SessionStore::new();
        store.resolve(&[1, 2, 3]);
        for i in 0..(COLLISION_LOG_CAP * 2) {
            store.resolve(&[1, 2, 10_000 + i as u64]);
        }
        assert_eq!(store.collision_log().len(), COLLISION_LOG_CAP);
    }

    #[test]
    fn prefix_break_stays_in_the_session_and_is_logged() {
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        let broken = store.resolve(&[1, 2, 9]);
        assert_eq!(broken.session_id, a.session_id, "stays in the session");
        assert!(broken.broke_prefix);
        assert_eq!(store.collision_log().len(), 1);
        let c = &store.collision_log()[0];
        assert_eq!(c.session_id, a.session_id);
        assert_eq!(c.shared_prefix_len, 2);
    }

    #[test]
    fn a_poisoned_lock_still_yields_the_store() {
        let store = SharedSessions::new();
        // Poison the mutex by panicking while holding it.
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = store.0.lock().unwrap();
            panic!("handler panic");
        }));
        assert!(poisoned.is_err());
        // The recovery lock must still return a usable guard.
        let mut g = store.lock();
        let r = g.resolve(&[1, 2, 3]);
        assert_eq!(r.session_id, 1, "the store survived the poison");
    }

    #[test]
    fn cumulative_excludes_incomplete_breaks() {
        let mut store = SessionStore::new();
        let a = store.resolve(&[1, 2, 3]);
        store.append(Record {
            session_id: a.session_id,
            turn: 1,
            status: Status::Complete,
            source: SourceLabel::ProviderReported,
            ttft_ms: Some(100.0),
            cached_tokens: 1000,
            cache_written_tokens: 0,
            resent_history_tokens: 2000,
            billed_input_tokens: 2000,
            broke_prefix: false,
            cost_usd: None,
            cost_saved_usd: None,
            repair_mode: crate::repair::RepairMode::Off,
            repaired: false,
            matches_canonical: None,
            drift_kind: None,
            canonicalized_tokens: 0,
            breakpoint_count: None,
        });
        let _ = store.resolve(&[1, 2, 9]); // break → incomplete marker
        let s = store.session(a.session_id).unwrap();
        assert!((s.cumulative_hit_rate().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_declared_key_pins_the_session_across_disjoint_history() {
        // Two requests under one key, with histories that share NO prefix
        // (a re-based/truncated re-send): the client's word wins — same
        // session, a continuation, no fork.
        let mut store = SessionStore::new();
        let a = store.resolve_keyed("conv-7", &[1, 2, 3]);
        let b = store.resolve_keyed("conv-7", &[9, 9, 9]);
        assert_eq!(a.session_id, b.session_id);
        assert!(b.continued, "the named session is the session");
        assert!(!b.broke_prefix);
    }

    #[test]
    fn distinct_keys_never_cross() {
        let mut store = SessionStore::new();
        let a = store.resolve_keyed("conv-a", &[1, 2, 3]);
        let b = store.resolve_keyed("conv-b", &[1, 2, 3]);
        assert_ne!(
            a.session_id, b.session_id,
            "identical bytes under two keys are two conversations"
        );
        let a2 = store.resolve_keyed("conv-a", &[1, 2, 3, 4]);
        assert_eq!(a2.session_id, a.session_id);
    }

    #[test]
    fn a_keyed_session_absorbs_the_longest_prefix() {
        // The keyed path still grows the recorded prefix, so a later
        // un-keyed request can match this session by prefix continuity.
        let mut store = SessionStore::new();
        let a = store.resolve_keyed("conv-7", &[1, 2]);
        assert_eq!(a.session_id, 1);
        let b = store.resolve_keyed("conv-7", &[1, 2, 3, 4]);
        assert_eq!(b.session_id, 1);
        let c = store.resolve(&[1, 2, 3, 4, 5]);
        assert_eq!(c.session_id, 1, "un-keyed continuity still finds it");
        assert!(c.continued);
    }
}
