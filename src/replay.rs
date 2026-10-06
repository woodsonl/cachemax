//! `cachemax replay --execute`: drive the recorded A/B pairs against a real
//! endpoint and report measurably what each form costs.
//!
//! The printed JSONL pairs (`a_drifted`, `b_canonical`) are the *input* for a
//! measurement; this module is that measurement. It sends each form directly
//! to the endpoint — bypassing the proxy, so the number is the raw provider
//! cache, not our own — `n` times, and reports per-form cached tokens
//! (median and max), the prompt size, and how many distinct upstream
//! instances answered.
//!
//! Measurement-validity rules this module keeps:
//!   - the two forms are sent interleaved (a, b, a, b, …) and each send gets
//!     a fresh connection, so form B's readings are not warmed by form A's
//!     sends and a connection-pinned router cannot answer every send from
//!     one instance by accident;
//!   - single sends are not a measurement on a routed endpoint: the same
//!     bytes landed in different cache namespaces across instances (observed
//!     live, a 2.3%↔99.7% flip on identical bodies). `n` samples and per-form
//!     max expose that; the instance fingerprint (a hash of the response's
//!     stable routing headers) names how many distinct backends answered;
//!   - a recovery figure is printed only when both forms were measured AND
//!     at least one instance answered both — a delta across two cache
//!     namespaces is a routing artifact, not a repair effect;
//!   - a form that got no readable send is *unmeasured*, never a zero: its
//!     medians are `null` and it drives no recovery claim. A response whose
//!     usage carries no cache figure at all (a non-caching model, a gateway
//!     that strips the details) is likewise unmeasured — the provider did
//!     not report a number, so there is no number.

use crate::adapters::anthropic::AnthropicAdapter;
use crate::adapters::openai::OpenAiAdapter;
use crate::adapters::Adapter;
use crate::record::SourceLabel;

/// Anthropic's Messages API requires `max_tokens`; the ledger records none.
/// Every replayed Anthropic body carries this documented default.
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 1024;

/// Which wire shape the endpoint speaks, for auth and usage parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    OpenAi,
    Anthropic,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Some(Backend::OpenAi),
            "anthropic" => Some(Backend::Anthropic),
            _ => None,
        }
    }

    /// The default environment variable carrying this backend's API key.
    pub fn default_key_env(self) -> &'static str {
        match self {
            Backend::OpenAi => "OPENAI_API_KEY",
            Backend::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    /// The full URL to POST a completion to, given the endpoint base. The base
    /// follows the same convention as `--upstream-url` for `serve`: it may or
    /// may not already carry the `/v1` version prefix ([`crate::proxy::
    /// versioned_base`]). OpenAI and every OpenAI-compatible gateway take
    /// `/v1/chat/completions`; Anthropic's native endpoint is `/v1/messages`.
    pub fn chat_url(self, endpoint: &str) -> String {
        let base = crate::proxy::versioned_base(endpoint);
        match self {
            Backend::OpenAi => format!("{base}/chat/completions"),
            Backend::Anthropic => format!("{base}/messages"),
        }
    }

    /// The request body for one form on this backend. OpenAI bodies go as
    /// recorded; Anthropic's Messages API requires a `max_tokens` and reads
    /// the system prompt from a top-level `system`, which the ledger recorded
    /// separately from `messages` — both are restored here, so a recorded
    /// chain replays as a valid request rather than a guaranteed 400.
    pub fn prepare_body(
        self,
        form_body: &serde_json::Value,
        system: &serde_json::Value,
        tools: &serde_json::Value,
    ) -> serde_json::Value {
        let mut body = form_body.clone();
        if let Some(obj) = body.as_object_mut() {
            // The recorded tools ride along in both dialects: they are part
            // of the cached prefix, so a replay without them measures the
            // wrong root.
            if !tools.is_null() {
                obj.insert("tools".into(), tools.clone());
            }
            if let Backend::Anthropic = self {
                obj.insert("max_tokens".into(), ANTHROPIC_DEFAULT_MAX_TOKENS.into());
                if !system.is_null() {
                    obj.insert("system".into(), system.clone());
                }
            }
        }
        body
    }

    /// Read the cache signal out of one response body with this backend's
    /// adapter. No allocation: both adapters are zero-sized.
    fn cache_signal(self, body: &[u8]) -> crate::adapters::CacheSignal {
        match self {
            Backend::OpenAi => OpenAiAdapter.cache_signal(body),
            Backend::Anthropic => AnthropicAdapter.cache_signal(body),
        }
    }
}

/// One A/B form sampled: the cached-token readings and the instance
/// fingerprints seen for that form.
///
/// The summary figures are `Option<u64>`: a form that never got a successful,
/// readable response has `None` — *unmeasured*, rendered `—`, never a
/// fabricated `0`. `sends` counts sends whose usage was readable; a 200 with
/// a streamed or otherwise unreadable body is not a reading and is not
/// counted (its reason lands in `failures`).
#[derive(Debug, Clone)]
pub struct FormSample {
    pub form: &'static str,
    pub sends: usize,
    /// Every readable reading, in send order (for transparency; the summary
    /// figures compress them). Empty when unmeasured.
    pub cached_readings: Vec<u64>,
    pub prompt_readings: Vec<u64>,
    pub median_cached: Option<u64>,
    pub max_cached: Option<u64>,
    /// Distinct upstream instances (by identifying-header fingerprint).
    pub instances: Vec<String>,
    /// Why each non-reading send produced no reading, in send order. Kept so
    /// a total-failure run can say what actually happened (HTTP 400 vs a
    /// streamed body vs a connection error) instead of a generic shrug.
    pub failures: Vec<String>,
}

impl FormSample {
    /// Whether this form has a usable measurement at all.
    pub fn measured(&self) -> bool {
        self.sends > 0
    }
}

/// The measurement for one recorded chain.
#[derive(Debug, Clone)]
pub struct ChainReport {
    pub session_id: u64,
    pub turn: u32,
    pub model: String,
    pub a_drifted: FormSample,
    pub b_canonical: FormSample,
}

/// Compute the median of a reading list (lower of the two middles for an
/// even count — a reading is one of the seen values, never an invented one).
/// `None` for an empty list: no readings is no median, not `0`.
pub fn median(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    Some(v[(v.len() - 1) / 2])
}

/// A stable fingerprint of the upstream that answered: its *static routing*
/// headers, sorted, hashed. Only headers that identify an instance/namespace
/// independent of the response are used — never per-response values like
/// `Date`, a request id, or `x-cache` (whose HIT/MISS flips reply to reply
/// and would split one stable instance into many).
pub fn instance_fingerprint(headers: &reqwest::header::HeaderMap) -> String {
    use std::collections::BTreeMap;
    const KEYS: &[&str] = &["server", "cf-ray", "x-upstream", "x-instance-id"];
    let mut map: BTreeMap<&str, String> = BTreeMap::new();
    for key in KEYS {
        if let Some(v) = headers.get(*key).and_then(|v| v.to_str().ok()) {
            map.insert(key, v.to_string());
        }
    }
    // `cf-ray` embeds a per-request suffix (`...-SJC`); keep only the
    // datacenter tag, which identifies the edge instance stably.
    if let Some(ray) = map.get("cf-ray").cloned() {
        if let Some((_, dc)) = ray.rsplit_once('-') {
            map.insert("cf-ray", dc.to_string());
        }
    }
    let joined = map
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";");
    short_hash(&joined)
}

/// A stable short hash of a string, hex, 8 chars. Uses the standard hasher
/// (already the house style for prefix hashing), truncated only for display;
/// the fingerprint only ever has to distinguish, not resist collisions.
fn short_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:08x}", (h.finish() & 0xffff_ffff) as u32)
}

/// Build the auth and framing headers for one form, per backend.
pub fn auth_headers(backend: Backend, api_key: Option<&str>) -> Vec<(&'static str, String)> {
    let mut out = vec![("content-type", "application/json".to_string())];
    match backend {
        Backend::OpenAi => {
            if let Some(k) = api_key {
                out.push(("authorization", format!("Bearer {k}")));
            }
        }
        Backend::Anthropic => {
            if let Some(k) = api_key {
                // Subscription OAuth tokens authenticate as Bearer with the
                // OAuth beta header; console API keys use x-api-key.
                if k.starts_with("sk-ant-oat") {
                    out.push(("authorization", format!("Bearer {k}")));
                    out.push(("anthropic-beta", "oauth-2025-04-20".to_string()));
                } else {
                    out.push(("x-api-key", k.to_string()));
                }
            }
            out.push(("anthropic-version", "2023-06-01".to_string()));
        }
    }
    out
}

/// Read the cached and prompt token counts out of a provider response body.
/// `None` when the body is not a JSON usage payload at all — an SSE stream,
/// an error shape — or when the usage carries NO cache figure: the provider
/// reported nothing, so there is nothing to measure, and reporting a measured
/// `0` would fabricate a cache miss. A genuinely reported `0` stays a `0`.
pub fn read_usage(backend: Backend, body: &[u8]) -> Option<(u64, u64)> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = v
        .get("usage")
        .or_else(|| v.pointer("/message/usage"))
        .filter(|u| u.is_object())?;
    let sig = backend.cache_signal(body);
    if sig.source == Some(SourceLabel::NoCacheTruth) {
        return None;
    }
    // Provider-reported prompt size; absent on some shapes, which is 0 prompt
    // tokens *known*, not a missing measurement (the cache reading is the
    // measurement, prompt is context).
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    Some((sig.cached_tokens, prompt))
}

/// How to drive a replay: endpoint, auth, sample count.
#[derive(Debug, Clone)]
pub struct ExecuteConfig {
    pub endpoint: String,
    pub backend: Backend,
    pub api_key: Option<String>,
    pub samples: usize,
}

/// What one send reported.
struct SendOutcome {
    cached: u64,
    prompt: u64,
    fingerprint: String,
}

/// Send one body once and read the cache usage out of the reply. A non-2xx
/// reply, or a 2xx whose usage cannot be read, is not a cache reading: it is
/// reported as an error, never counted as `0`.
async fn send_once(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    body: &serde_json::Value,
) -> Result<SendOutcome, String> {
    let url = cfg.backend.chat_url(&cfg.endpoint);
    let mut req = client.post(&url).body(body.to_string());
    for (k, v) in auth_headers(cfg.backend, cfg.api_key.as_deref()) {
        req = req.header(k, v);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    let fingerprint = instance_fingerprint(resp.headers());
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("body read failed: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "HTTP {status}: {}",
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(200)
                .collect::<String>()
        ));
    }
    let (cached, prompt) = read_usage(cfg.backend, &bytes).ok_or_else(|| {
        "response carried no readable cache figure (streamed body, or usage with no cache field)"
            .to_string()
    })?;
    Ok(SendOutcome {
        cached,
        prompt,
        fingerprint,
    })
}

/// One form's accumulated readings across its sends.
struct SampleAcc {
    form: &'static str,
    cached: Vec<u64>,
    prompt: Vec<u64>,
    instances: Vec<String>,
    failures: Vec<String>,
}

impl SampleAcc {
    fn new(form: &'static str) -> Self {
        SampleAcc {
            form,
            cached: Vec::new(),
            prompt: Vec::new(),
            instances: Vec::new(),
            failures: Vec::new(),
        }
    }

    async fn send(
        &mut self,
        client: &reqwest::Client,
        cfg: &ExecuteConfig,
        body: &serde_json::Value,
    ) {
        match send_once(client, cfg, body).await {
            Ok(o) => {
                self.cached.push(o.cached);
                self.prompt.push(o.prompt);
                if !self.instances.contains(&o.fingerprint) {
                    self.instances.push(o.fingerprint);
                }
            }
            Err(e) => self.failures.push(e),
        }
    }

    fn finish(self) -> FormSample {
        FormSample {
            form: self.form,
            sends: self.cached.len(),
            median_cached: median(&self.cached),
            max_cached: self.cached.iter().copied().max(),
            cached_readings: self.cached,
            prompt_readings: self.prompt,
            instances: self.instances,
            failures: self.failures,
        }
    }
}

/// Drive one A/B pair for a recorded chain: the drifted and canonical forms
/// are reconstructed from the request, shaped for the backend, and sampled
/// interleaved — a, b, a, b, … — so neither form's sends warm the other's
/// prefix advantage.
pub async fn execute_pair(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    request: &crate::ledger::ReplayRequest,
) -> ChainReport {
    let pair = crate::repair::replay_pair(request);
    execute_bodies(
        client,
        cfg,
        &request.request_system,
        &request.request_tools,
        &pair["a_drifted"],
        &pair["b_canonical"],
    )
    .await
}

/// Drive one A/B pair from raw bodies — the same interleaved sampling as
/// [`execute_pair`], for callers that hold their own body pair (the
/// drift-cost matrix runs built-in fixtures). `system` is the recorded
/// top-level system for the Anthropic dialect (injected with `max_tokens`);
/// `Null` for dialects where the system rides inside `messages`.
pub async fn execute_bodies(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    system: &serde_json::Value,
    tools: &serde_json::Value,
    a_body: &serde_json::Value,
    b_body: &serde_json::Value,
) -> ChainReport {
    let a_body = cfg.backend.prepare_body(a_body, system, tools);
    let b_body = cfg.backend.prepare_body(b_body, system, tools);
    let mut a = SampleAcc::new("a_drifted");
    let mut b = SampleAcc::new("b_canonical");
    for _ in 0..cfg.samples {
        a.send(client, cfg, &a_body).await;
        b.send(client, cfg, &b_body).await;
    }
    ChainReport {
        session_id: 0,
        turn: 0,
        model: String::new(),
        a_drifted: a.finish(),
        b_canonical: b.finish(),
    }
}

/// Render the measurement table for one chain. Plain text, aligned, honest:
/// unmeasured forms show `—`; a recovery figure is printed only when both
/// forms were measured AND an instance answered both (a delta across two
/// cache namespaces is routing noise, not a repair effect).
pub fn render_report(r: &ChainReport) -> String {
    fn cell(v: Option<u64>) -> String {
        v.map_or_else(|| "—".to_string(), |n| n.to_string())
    }
    let mut out = String::new();
    out.push_str(&format!(
        "session {} turn {} · {}\n",
        r.session_id, r.turn, r.model
    ));
    out.push_str(&format!(
        "  {:<12} {:>6} {:>8} {:>8} {:>8} {:>10}\n",
        "form", "sends", "median", "max", "prompt", "instances"
    ));
    for f in [&r.a_drifted, &r.b_canonical] {
        out.push_str(&format!(
            "  {:<12} {:>6} {:>8} {:>8} {:>8} {:>10}\n",
            f.form,
            f.sends,
            cell(f.median_cached),
            cell(f.max_cached),
            // Prompt size is context for the reading, not the measurement;
            // render-time only.
            cell(median(&f.prompt_readings)),
            f.instances.len()
        ));
    }
    let shared_instance = !r.a_drifted.instances.is_empty()
        && r.a_drifted
            .instances
            .iter()
            .any(|i| r.b_canonical.instances.contains(i));
    match (r.a_drifted.median_cached, r.b_canonical.median_cached) {
        (Some(a), Some(b)) if b > a && shared_instance => {
            out.push_str(&format!(
                "  → canonical form recovers {} cached tokens (median) over drifted\n",
                b - a
            ));
        }
        (Some(_), Some(_)) if !shared_instance => {
            out.push_str(
                "  · forms answered by different upstream instances — no like-for-like delta\n",
            );
        }
        (Some(_), Some(_)) => {}
        _ => out.push_str(
            "  · not a measurement: a form had no readable reading (see the logged failures)\n",
        ),
    }
    if r.a_drifted.instances.len() > 1 || r.b_canonical.instances.len() > 1 {
        out.push_str(
            "  ! more than one upstream instance answered — single readings are routing-lottery sensitive; read max-of-N\n",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_is_none_for_no_readings_never_zero() {
        assert_eq!(median(&[1455, 20, 99]), Some(99));
        assert_eq!(
            median(&[10, 20]),
            Some(10),
            "lower middle for an even count"
        );
        assert_eq!(median(&[]), None, "no readings is unmeasured, not 0");
        assert_eq!(median(&[7]), Some(7));
    }

    #[test]
    fn backend_parses_and_defaults_its_key_env() {
        assert_eq!(Backend::parse("openai"), Some(Backend::OpenAi));
        assert_eq!(Backend::parse("Anthropic"), Some(Backend::Anthropic));
        assert_eq!(Backend::parse("claude"), None);
        assert_eq!(Backend::parse("nope"), None);
        assert_eq!(Backend::OpenAi.default_key_env(), "OPENAI_API_KEY");
        assert_eq!(Backend::Anthropic.default_key_env(), "ANTHROPIC_API_KEY");
    }

    #[test]
    fn chat_url_honors_an_endpoint_that_already_carries_v1() {
        // Both the bare host and the documented `/v1` form must yield one
        // `/v1/...`, the same convention `serve` uses.
        assert_eq!(
            Backend::OpenAi.chat_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            Backend::OpenAi.chat_url("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            Backend::OpenAi.chat_url("https://gateway.example/v1/"),
            "https://gateway.example/v1/chat/completions"
        );
        // Anthropic's native endpoint is `/v1/messages`, not chat/completions.
        assert_eq!(
            Backend::Anthropic.chat_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn usage_without_a_cache_field_is_unmeasured_not_a_measured_zero() {
        // Non-caching model, or a gateway that strips the details: usage is
        // readable, the cache figure was never exposed. There is no number —
        // reporting 0 would fabricate a cache miss.
        assert_eq!(
            read_usage(Backend::OpenAi, br#"{"usage":{"prompt_tokens":100}}"#),
            None
        );
        assert_eq!(
            read_usage(Backend::Anthropic, br#"{"usage":{"input_tokens":100}}"#),
            None
        );
    }

    #[test]
    fn a_reported_zero_stays_a_measured_zero() {
        let body =
            br#"{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":0}}}"#;
        assert_eq!(read_usage(Backend::OpenAi, body), Some((0, 100)));
    }

    #[test]
    fn openai_usage_reads_cached_and_prompt() {
        let body =
            br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        assert_eq!(read_usage(Backend::OpenAi, body), Some((1455, 2140)));
    }

    #[test]
    fn anthropic_usage_reads_read_and_input() {
        let body = br#"{"usage":{"input_tokens":100,"cache_read_input_tokens":900}}"#;
        assert_eq!(read_usage(Backend::Anthropic, body), Some((900, 100)));
    }

    #[test]
    fn anthropic_nested_message_usage_is_read() {
        let body = br#"{"message":{"usage":{"input_tokens":2100,"cache_read_input_tokens":900}}}"#;
        assert_eq!(read_usage(Backend::Anthropic, body), Some((900, 2100)));
    }

    #[test]
    fn a_body_with_no_usage_is_unmeasured_not_zero() {
        // An SSE stream, an error shape, a plain-text body: no JSON usage,
        // so no reading — reported as such, never as `0` cached.
        assert_eq!(read_usage(Backend::OpenAi, b"{}"), None);
        assert_eq!(read_usage(Backend::OpenAi, b"data: {\"x\":1}\n\n"), None);
        assert_eq!(read_usage(Backend::OpenAi, b"not json"), None);
    }

    #[test]
    fn anthropic_bodies_carry_the_required_max_tokens_and_system() {
        // Anthropic's Messages API rejects a body without `max_tokens`, and
        // reads the system prompt from the top level, not from `messages`.
        let form =
            serde_json::json!({"model": "claude", "messages": [{"role": "user", "content": "hi"}]});
        let system = serde_json::json!("You call tools.");
        let body = Backend::Anthropic.prepare_body(&form, &system, &serde_json::Value::Null);
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["system"], "You call tools.");
        assert_eq!(body["messages"][0]["content"], "hi");

        // A chain recorded with no system stays valid; no null "system" key.
        let body = Backend::Anthropic.prepare_body(
            &form,
            &serde_json::Value::Null,
            &serde_json::Value::Null,
        );
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("system").is_none());

        // OpenAI bodies go as recorded — no injected fields.
        let body = Backend::OpenAi.prepare_body(&form, &system, &serde_json::Value::Null);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn subscription_oauth_tokens_authenticate_as_bearer() {
        // Claude subscription tokens (sk-ant-oat…) ride Authorization with
        // the OAuth beta; console keys ride x-api-key. The executor must
        // send each the way its flow expects or the upstream 401s.
        let oat = auth_headers(Backend::Anthropic, Some("sk-ant-oat01-abc"));
        assert!(oat.contains(&("authorization", "Bearer sk-ant-oat01-abc".to_string())));
        assert!(oat.contains(&("anthropic-beta", "oauth-2025-04-20".to_string())));
        assert!(!oat.contains(&("x-api-key", "sk-ant-oat01-abc".to_string())));
        let key = auth_headers(Backend::Anthropic, Some("sk-ant-api03-xyz"));
        assert!(key.contains(&("x-api-key", "sk-ant-api03-xyz".to_string())));
        assert!(!key.iter().any(|(k, _)| *k == "authorization"));
    }

    #[test]
    fn auth_headers_carry_the_right_scheme_per_backend() {
        let oai = auth_headers(Backend::OpenAi, Some("sk-x"));
        assert!(oai.contains(&("authorization", "Bearer sk-x".to_string())));
        let anth = auth_headers(Backend::Anthropic, Some("sk-y"));
        assert!(anth.contains(&("x-api-key", "sk-y".to_string())));
        assert!(anth.contains(&("anthropic-version", "2023-06-01".to_string())));
    }

    #[test]
    fn fingerprints_use_routing_headers_but_ignore_volatile_ones() {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut a = HeaderMap::new();
        a.insert("server", HeaderValue::from_static("nginx"));
        a.insert("cf-ray", HeaderValue::from_static("abc123-SJC"));
        a.insert(
            "date",
            HeaderValue::from_static("Mon, 05 Oct 2026 09:00:00 GMT"),
        );
        a.insert("x-request-id", HeaderValue::from_static("req-1"));
        let mut b = HeaderMap::new();
        b.insert("server", HeaderValue::from_static("nginx"));
        b.insert("cf-ray", HeaderValue::from_static("def456-SJC"));
        b.insert(
            "date",
            HeaderValue::from_static("Mon, 05 Oct 2026 09:00:01 GMT"),
        );
        b.insert("x-request-id", HeaderValue::from_static("req-2"));
        // Same edge (datacenter tag SJC); Date and request id differ but are
        // volatile — the fingerprint must not treat them as instances.
        assert_eq!(instance_fingerprint(&a), instance_fingerprint(&b));

        // A genuinely different edge is a different instance.
        let mut c = HeaderMap::new();
        c.insert("server", HeaderValue::from_static("nginx"));
        c.insert("cf-ray", HeaderValue::from_static("ghi789-IAD"));
        assert_ne!(instance_fingerprint(&a), instance_fingerprint(&c));

        // A different server string too.
        let mut d = HeaderMap::new();
        d.insert("server", HeaderValue::from_static("cloudflare"));
        d.insert("cf-ray", HeaderValue::from_static("abc123-SJC"));
        assert_ne!(instance_fingerprint(&a), instance_fingerprint(&d));
    }

    #[test]
    fn cache_state_headers_do_not_split_one_instance() {
        // `x-cache` flips MISS→HIT reply to reply on many CDNs. One stable
        // instance must not read as a routed spread just because its cache
        // warmed up mid-run.
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut miss = HeaderMap::new();
        miss.insert("server", HeaderValue::from_static("nginx"));
        miss.insert("cf-ray", HeaderValue::from_static("abc123-SJC"));
        miss.insert("x-cache", HeaderValue::from_static("MISS"));
        let mut hit = HeaderMap::new();
        hit.insert("server", HeaderValue::from_static("nginx"));
        hit.insert("cf-ray", HeaderValue::from_static("abc123-SJC"));
        hit.insert("x-cache", HeaderValue::from_static("HIT"));
        assert_eq!(instance_fingerprint(&miss), instance_fingerprint(&hit));
    }
}
