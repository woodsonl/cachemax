//! Proxy core: forward first, unbuffered SSE passthrough, observe while
//! streaming, finalize one record per request.
//!
//! The design rule (spec §Streaming): the SSE bytes the client receives are the
//! bytes we observed; the stored record must byte-match reassembly of the stream
//! by concatenation. So the hot path never parses-to-forward — it forwards
//! chunks untouched and runs observation on a side copy. Only at finalize do we
//! look at the buffered copy (bounded: usage arrives in the terminal SSE event).
//!
//! The network wiring is in [`serve`]; the pure seams ([`build_record`],
//! [`observe`], [`finalize`], [`RequestPlan`]) are tested without a socket.

use crate::adapters::Adapter;
use crate::rates::Rates;
use crate::record::{Record, SourceLabel, Status};
use crate::sessions::{SessionStore, SharedSessions};
use crate::tokenize::{Message, Tokenizer};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

/// What the proxy knows about a request *before* forwarding: which session it
/// extends, its turn index, and the binding denominator.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestPlan {
    pub session_id: u64,
    pub turn: u32,
    /// `resent_history_tokens`: token count of system + all prior messages,
    /// excluding this turn's new content.
    pub resent_history_tokens: u64,
    /// This request broke a tracked session's prefix (measured as a miss).
    pub broke_prefix: bool,
}

/// Compute the plan for an incoming request from its messages.
///
/// `prefix_hashes` is this request's cumulative per-message hash sequence. The
/// matched session tells us the turn index (its record count) and the longest
/// shared prefix; everything past that shared prefix is this turn's new
/// content, so the denominator is the token count up to the shared boundary.
///
/// Token counts are measured locally here only to size the denominator. On the
/// cloud path the *provider's* figure stays authoritative for `cached_tokens`;
/// this local count is the re-sent-history measure the dashboard divides by.
pub fn plan_request(
    store: &mut SessionStore,
    tokenizer: &Tokenizer,
    messages: &[Message],
) -> RequestPlan {
    let hashes = tokenizer.prefix_hashes(messages);
    let resolution = store.resolve(&hashes);
    let session_id = resolution.session_id;
    let session = store.session(session_id);
    let turn = session.map(|s| s.records.len() as u32).unwrap_or(0);
    // Binding denominator = the re-sent history: system + all prior user,
    // assistant, and tool messages, excluding this turn's new content. This
    // turn's new content is the final message, so history is every message
    // before it. Turn 0 is cold and has no history.
    let resent_history_tokens = if turn == 0 || messages.len() <= 1 {
        0
    } else {
        tokenizer.count_messages(&messages[..messages.len() - 1]) as u64
    };

    RequestPlan {
        session_id,
        turn,
        resent_history_tokens,
        broke_prefix: resolution.broke_prefix,
    }
}

/// The raw observed figures for one turn, before cost is applied.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Observation {
    pub ttft_ms: Option<f64>,
    pub cached_tokens: u64,
    pub cache_written_tokens: u64,
    pub billed_input_tokens: u64,
}

/// Build a record from the observed response. The pure seam C1/C2 test against,
/// independent of the network. `rates` prices the no-cache counterfactual when
/// the model is known; unknown models simply carry no cost.
pub fn build_record(
    plan: &RequestPlan,
    obs: Observation,
    model: &str,
    rates: &Rates,
    source: SourceLabel,
    complete: bool,
) -> Record {
    // Local engines report cached tokens in the engine's own token space, which
    // can drift from our tokenizer; clamp to the history span so the ratio never
    // exceeds 100%. Cloud counts are provider-reported and shown as-is.
    let cached = match source {
        SourceLabel::EngineMeasured => obs.cached_tokens.min(plan.resent_history_tokens),
        SourceLabel::ProviderReported => obs.cached_tokens,
        SourceLabel::NoCacheTruth => 0,
    };
    let cost_usd = rates
        .lookup(model)
        .map(|r| r.input_cost(obs.billed_input_tokens));
    let cost_saved_usd = rates.cost_saved(
        model,
        cached,
        plan.resent_history_tokens,
        obs.cache_written_tokens,
    );
    Record {
        session_id: plan.session_id,
        turn: plan.turn,
        status: if complete {
            Status::Complete
        } else {
            Status::Incomplete
        },
        source,
        ttft_ms: obs.ttft_ms,
        cached_tokens: cached,
        cache_written_tokens: obs.cache_written_tokens,
        resent_history_tokens: plan.resent_history_tokens,
        billed_input_tokens: obs.billed_input_tokens,
        broke_prefix: plan.broke_prefix,
        cost_usd,
        cost_saved_usd,
    }
}

/// The cache signal an adapter reads from a response, factored out so the proxy
/// and adapter share one path. `response_body` may be a single JSON document
/// (non-streaming) or a buffered SSE tail; the usage-bearing event wins.
pub fn observe<A: Adapter>(adapter: &A, response_body: &[u8]) -> (u64, u64, SourceLabel) {
    observe_doc(adapter, &last_json_event(response_body))
}

/// [`observe`] over an already-reduced JSON document, so a caller that also
/// needs the document (e.g. for billing) reduces the tail only once.
fn observe_doc<A: Adapter>(adapter: &A, doc: &[u8]) -> (u64, u64, SourceLabel) {
    let sig = adapter.cache_signal(doc);
    (
        sig.cached_tokens,
        sig.written_tokens,
        sig.source.unwrap_or_else(|| adapter.source()),
    )
}

/// Reduce a body to the most informative JSON document: the whole body if it is
/// one JSON object, else the last `data:` SSE event. Prefers an event that
/// carries `usage` (cache/ billing figures live there); falls back to the last
/// parseable event. Usage arrives in the terminal event, so scanning from the
/// end is both correct and cheap.
fn last_json_event(body: &[u8]) -> Vec<u8> {
    if body.first().map(|&b| b == b'{').unwrap_or(false)
        && serde_json::from_slice::<serde_json::Value>(body).is_ok()
    {
        return body.to_vec();
    }
    let mut fallback: Option<Vec<u8>> = None;
    for line in body.split(|&b| b == b'\n').rev() {
        // SSE allows `data:` with or without the space.
        let line = line
            .strip_prefix(b"data: ")
            .or_else(|| line.strip_prefix(b"data:"))
            .unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line == b"[DONE]" || line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<serde_json::Value>(line) {
            Ok(v) => {
                if v.get("usage").is_some() {
                    return line.to_vec();
                }
                if fallback.is_none() {
                    fallback = Some(line.to_vec());
                }
            }
            Err(_) => continue,
        }
    }
    fallback.unwrap_or_default()
}

/// Incremental stream observer. Fed each forwarded chunk in order, untouched;
/// tracks TTFT and accumulates a bounded copy for the finalize parse.
///
/// The client stream is *never* delayed by this: chunk bytes are cloned for
/// observation and the originals are forwarded as-is. See [`serve`].
pub struct StreamObserver {
    started: Instant,
    ttft_ms: Option<f64>,
    saw_first_byte: bool,
    tail: Vec<u8>,
    /// Cap the retained copy; usage always arrives in the terminal event, so
    /// retaining the tail is sufficient and bounded (spec: no unbounded buffer).
    cap: usize,
}

impl StreamObserver {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            ttft_ms: None,
            saw_first_byte: false,
            tail: Vec::new(),
            cap: 64 * 1024,
        }
    }

    /// Observe one forwarded chunk. Called with the exact bytes being sent to
    /// the client; must not mutate them.
    pub fn on_chunk(&mut self, chunk: &[u8]) {
        if !self.saw_first_byte && !chunk.is_empty() {
            self.saw_first_byte = true;
            self.ttft_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
        }
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > self.cap {
            let drop = self.tail.len() - self.cap;
            self.tail.drain(..drop);
        }
    }

    /// Finalize the observation into a record. `complete` is false when the
    /// stream ended early (client disconnect, upstream error mid-stream).
    pub fn finalize<A: Adapter>(
        &self,
        plan: &RequestPlan,
        adapter: &A,
        model: &str,
        rates: &Rates,
        complete: bool,
    ) -> Record {
        // Reduce the tail once; both the cache signal and the billed count read
        // the same usage-bearing event.
        let doc = last_json_event(&self.tail);
        let (cached_tokens, cache_written_tokens, source) = observe_doc(adapter, &doc);
        let billed = billed_from_doc(&doc).unwrap_or(plan.resent_history_tokens);
        let obs = Observation {
            ttft_ms: self.ttft_ms,
            cached_tokens,
            cache_written_tokens,
            billed_input_tokens: billed,
        };
        build_record(plan, obs, model, rates, source, complete)
    }
}

impl Default for StreamObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Read `billed_input_tokens` from an already-reduced JSON document.
fn billed_from_doc(doc: &[u8]) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(doc).ok()?;
    v.pointer("/usage/prompt_tokens").and_then(|n| n.as_u64())
}

/// State shared by every request handler.
pub struct AppState<A: Adapter> {
    pub adapter: Arc<A>,
    pub tokenizer: Tokenizer,
    pub sessions: Arc<SharedSessions>,
    pub rates: Rates,
    pub upstream_url: String,
    pub client: reqwest::Client,
}

/// Run the proxy. Forwards to `upstream_url`, streams the response through
/// unbuffered, and finalizes one record per request.
pub async fn serve<A: Adapter + 'static>(
    adapter: A,
    tokenizer: Tokenizer,
    rates: Rates,
    upstream_url: String,
    bind: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        adapter: Arc::new(adapter),
        tokenizer,
        sessions: Arc::new(SharedSessions::new()),
        rates,
        upstream_url,
        client: reqwest::Client::new(),
    });

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(addr = %listener.local_addr()?, "cachemax listening");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

/// The proxy's axum router. Shared by [`serve`] and integration tests so both
/// exercise one construction path. The dashboard (`/`) and its data endpoints
/// live on the same server as the API.
pub fn router<A: Adapter + 'static>(state: Arc<AppState<A>>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat::<A>))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route("/", axum::routing::get(serve_dashboard))
        .route("/api/state", axum::routing::get(dashboard_state::<A>))
        .route("/api/export", axum::routing::get(export_session::<A>))
        .with_state(state)
}

/// Serve the embedded single-file dashboard.
async fn serve_dashboard() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(crate::dashboard::DASHBOARD_HTML))
        .unwrap()
}

/// The live dashboard snapshot: the most recently active session's view.
async fn dashboard_state<A: Adapter + 'static>(State(state): State<Arc<AppState<A>>>) -> Response {
    let (records, session_count, live) = {
        let guard = state.sessions.0.lock().unwrap();
        let live = guard
            .most_recent()
            .map(|s| !s.records.is_empty())
            .unwrap_or(false);
        let records = guard
            .most_recent()
            .map(|s| s.records.clone())
            .unwrap_or_default();
        (records, guard.len(), live)
    };
    let view = crate::dashboard::view(&records, live, session_count);
    match serde_json::to_vec(&view) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(bytes))
            .unwrap(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Export the most recently active session as metrics-only JSONL.
async fn export_session<A: Adapter + 'static>(State(state): State<Arc<AppState<A>>>) -> Response {
    let (records, id) = {
        let guard = state.sessions.0.lock().unwrap();
        match guard.most_recent() {
            Some(s) => (s.records.clone(), s.id),
            None => (Vec::new(), 0),
        }
    };
    match crate::export::to_jsonl(&records) {
        Ok(jsonl) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/x-ndjson")
            .header(
                "content-disposition",
                format!("attachment; filename=\"cachemax-{id}.jsonl\""),
            )
            .body(Body::from(jsonl))
            .unwrap(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Request headers forwarded to the upstream verbatim. Auth must pass through
/// or every cloud call 401s; the rest are the provider-identification headers
/// OpenRouter and similar gateways read. Everything else is dropped.
const FORWARD_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "anthropic-version",
    "openai-organization",
    "openai-project",
    "http-referer",
    "x-title",
];

/// The request handler: plan, forward, stream through, observe, finalize.
pub async fn handle_chat<A: Adapter + 'static>(
    State(state): State<Arc<AppState<A>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let messages = match messages_from(&body) {
        Some(m) => m,
        None => {
            return (StatusCode::BAD_REQUEST, "could not parse messages").into_response();
        }
    };
    let model = model_from(&body);

    let plan = {
        let mut guard = state.sessions.0.lock().unwrap();
        plan_request(&mut guard, &state.tokenizer, &messages)
    };

    // Forward first. The request body is passed through untouched; auth and
    // provider-identification headers are forwarded so cloud keys keep working.
    let url = format!(
        "{}/v1/chat/completions",
        state.upstream_url.trim_end_matches('/')
    );
    let mut req = state
        .client
        .post(&url)
        .header("content-type", "application/json");
    for name in FORWARD_HEADERS {
        if let Some(value) = headers.get(*name) {
            req = req.header(*name, value);
        }
    }
    let upstream = match req.body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            let record = build_record(
                &plan,
                Observation::default(),
                &model,
                &state.rates,
                state.adapter.source(),
                false,
            );
            crate::export::log_finalize(&record);
            state.sessions.0.lock().unwrap().append(record);
            return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response();
        }
    };
    let status = upstream.status();
    let mut upstream_stream = upstream.bytes_stream();

    // Unbuffered passthrough: each chunk is forwarded as it arrives; a clone is
    // handed to the observer, which never blocks the forward path.
    let adapter = state.adapter.clone();
    let sessions = state.sessions.clone();
    let rates = state.rates.clone();
    let stream = async_stream::stream! {
        let mut observer = StreamObserver::new();
        let mut complete = true;
        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    observer.on_chunk(&bytes);
                    yield Ok::<Bytes, std::io::Error>(bytes);
                }
                Err(_) => {
                    complete = false;
                    break;
                }
            }
        }
        let record = observer.finalize(&plan, adapter.as_ref(), &model, &rates, complete);
        crate::export::log_finalize(&record);
        sessions.0.lock().unwrap().append(record);
    };

    Response::builder()
        .status(status)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Extract messages from an OpenAI-dialect request body. Dialect-neutral output.
fn messages_from(body: &[u8]) -> Option<Vec<Message>> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let arr = v.get("messages")?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for m in arr {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let text = flatten_content(m.get("content"));
        out.push(Message {
            role: role.to_string(),
            text,
        });
    }
    Some(out)
}

/// Extract the model name from a request body (for rate lookup).
fn model_from(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_default()
}

/// Flatten OpenAI content — a string, or an array of `{type,text}` parts.
fn flatten_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::openai::OpenAiAdapter;

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            text: text.into(),
        }
    }

    #[test]
    fn a_dropped_stream_finalizes_incomplete() {
        let plan = RequestPlan {
            session_id: 1,
            turn: 1,
            resent_history_tokens: 1550,
            broke_prefix: false,
        };
        let obs = Observation {
            ttft_ms: Some(120.0),
            ..Default::default()
        };
        let r = build_record(
            &plan,
            obs,
            "gpt-4o",
            &Rates::builtin(),
            SourceLabel::ProviderReported,
            false,
        );
        assert_eq!(r.status, Status::Incomplete);
    }

    #[test]
    fn observe_routes_through_the_adapter() {
        let body = br#"{"usage":{"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        let (cached, written, source) = observe(&OpenAiAdapter, body);
        assert_eq!(cached, 1455);
        assert_eq!(written, 0);
        assert_eq!(source, SourceLabel::ProviderReported);
    }

    #[test]
    fn observer_records_ttft_at_first_nonempty_chunk() {
        let mut o = StreamObserver::new();
        assert!(o.ttft_ms.is_none());
        o.on_chunk(b"");
        assert!(o.ttft_ms.is_none(), "empty chunk is not first byte");
        o.on_chunk(b"data: {}\n\n");
        assert!(o.ttft_ms.is_some(), "first nonempty chunk sets TTFT");
    }

    #[test]
    fn observer_finalizes_cached_and_billed_from_the_tail() {
        let plan = RequestPlan {
            session_id: 7,
            turn: 2,
            resent_history_tokens: 1810,
            broke_prefix: false,
        };
        let mut o = StreamObserver::new();
        o.on_chunk(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n");
        o.on_chunk(
            b"data: {\"usage\":{\"prompt_tokens\":2140,\"prompt_tokens_details\":{\"cached_tokens\":1455}}}\n\n",
        );
        let r = o.finalize(&plan, &OpenAiAdapter, "gpt-4o", &Rates::builtin(), true);
        assert_eq!(r.cached_tokens, 1455);
        assert_eq!(r.billed_input_tokens, 2140);
        assert_eq!(r.resent_history_tokens, 1810);
        assert_eq!(r.turn, 2);
        assert!(r.cost_usd.is_some(), "known model carries cost");
    }

    #[test]
    fn tail_is_bounded() {
        let mut o = StreamObserver::new();
        for _ in 0..2000 {
            o.on_chunk(&[b'x'; 100]);
        }
        assert!(o.tail.len() <= o.cap, "tail must not grow unbounded");
    }

    #[test]
    fn plan_tracks_turns_and_denominator_across_a_session() {
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        let conv1 = vec![msg("system", "You are helpful."), msg("user", "Hi")];
        let p0 = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &conv1);
        assert_eq!(p0.turn, 0, "first request is cold");
        assert_eq!(p0.resent_history_tokens, 0, "turn 0 has no history");

        // Simulate the finalized turn 0 so the next request sees turn 1.
        let r = build_record(
            &p0,
            Observation {
                ttft_ms: Some(100.0),
                billed_input_tokens: 10,
                ..Default::default()
            },
            "gpt-4o",
            &Rates::builtin(),
            SourceLabel::ProviderReported,
            true,
        );
        store.0.lock().unwrap().append(r);

        let conv2 = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "More"),
        ];
        let p1 = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &conv2);
        assert_eq!(p1.session_id, p0.session_id, "same session");
        assert_eq!(p1.turn, 1);
        assert!(
            p1.resent_history_tokens > 0,
            "history excludes this turn's new content but includes the prefix"
        );
    }

    #[test]
    fn anthropic_write_split_lands_in_the_record() {
        use crate::adapters::anthropic::AnthropicAdapter;
        let plan = RequestPlan {
            session_id: 1,
            turn: 1,
            resent_history_tokens: 2000,
            broke_prefix: false,
        };
        let mut o = StreamObserver::new();
        o.on_chunk(
            b"data: {\"usage\":{\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":300,\"input_tokens\":2100}}\n\n",
        );
        let r = o.finalize(
            &plan,
            &AnthropicAdapter,
            "claude-3-5-sonnet",
            &Rates::builtin(),
            true,
        );
        assert_eq!(r.cached_tokens, 900);
        assert_eq!(r.cache_written_tokens, 300, "creation split recorded");
        assert!(r.cost_saved_usd.is_some(), "anthropic rates known");
    }

    #[test]
    fn denominator_grows_with_each_turn() {
        // Regression: the re-sent history must grow as the conversation does
        // (system + prior messages), not stay pinned at the first request's
        // span. Turn t's history is every message before the final one.
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        let mut histories = Vec::new();
        let mut convo: Vec<Message> = vec![msg("system", "You are a helpful assistant.")];
        for turn in 0..4 {
            convo.push(msg("user", &format!("Question number {turn} please")));
            // resolve/append so turn indexing advances
            let p = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &convo);
            let r = build_record(
                &p,
                Observation {
                    billed_input_tokens: 10,
                    ..Default::default()
                },
                "gpt-4o",
                &Rates::builtin(),
                SourceLabel::ProviderReported,
                true,
            );
            histories.push(p.resent_history_tokens);
            store.0.lock().unwrap().append(r);
            convo.push(msg("assistant", &format!("Answer number {turn} here")));
        }
        assert_eq!(histories[0], 0, "turn 0 is cold");
        // Each subsequent turn's history strictly grows.
        for w in histories.windows(2) {
            if w[0] != 0 {
                assert!(w[1] > w[0], "history must grow: {:?}", histories);
            }
        }
    }

    #[test]
    fn plan_flags_a_broken_prefix() {
        let tokenizer = Tokenizer::default_encoder().unwrap();
        let store = Arc::new(SharedSessions::new());
        // Seed a session with a two-message prefix.
        let seed = vec![msg("system", "You are helpful."), msg("user", "First")];
        let _ = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &seed);
        // A request that shares only the system message then diverges.
        let broken = vec![msg("system", "You are helpful."), msg("user", "Different")];
        let p = plan_request(&mut store.0.lock().unwrap(), &tokenizer, &broken);
        assert!(p.broke_prefix, "a divergent prefix is flagged on the plan");
        assert_eq!(
            p.session_id, 1,
            "the break stays in the tracked session, not a new one"
        );
    }

    #[test]
    fn messages_from_flattens_string_and_parts() {
        let body = br#"{"messages":[
            {"role":"system","content":"sys"},
            {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}
        ]}"#;
        let msgs = messages_from(body).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "sys");
        assert_eq!(msgs[1].text, "ab");
    }
}
