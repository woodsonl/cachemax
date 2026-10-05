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
//! Single sends are not a measurement on a routed endpoint: the same bytes
//! landed in different cache namespaces across instances (observed live, a
//! 2.3%↔99.7% flip on identical bodies). `n` samples and per-form max expose
//! that; the instance fingerprint (a hash of the response's identifying
//! headers) names how many distinct backends answered, so a reader can weigh
//! the spread.
//!
//! Honesty rules this module keeps:
//!   - a form that got no successful send is *unmeasured*, never a zero;
//!     its medians are `null` and it drives no recovery claim;
//!   - a response whose usage cannot be read (an SSE body, a shape we do not
//!     recognize) is *unmeasured*, never counted as `0` cached;
//!   - the recovery figure is only printed when both forms were measured.

use crate::adapters::anthropic::AnthropicAdapter;
use crate::adapters::openai::OpenAiAdapter;
use crate::adapters::Adapter;
use serde::Serialize;

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

    /// Read the cache signal out of one response body with this backend's
    /// adapter. No allocation: both adapters are zero-sized.
    fn cache_signal(self, body: &[u8]) -> crate::adapters::CacheSignal {
        match self {
            Backend::OpenAi => OpenAiAdapter.cache_signal(body),
            Backend::Anthropic => AnthropicAdapter.cache_signal(body),
        }
    }
}

/// One A/B pair sampled: the cached-token readings and the instance
/// fingerprints seen for each form.
///
/// The summary figures are `Option<u64>`: a form that never got a successful,
/// readable response has `None` — *unmeasured*, rendered `—`, never a
/// fabricated `0`. `sends` is the count of successful sends, which may be
/// fewer than requested when some failed; `sends == 0` is the unmeasured case.
#[derive(Debug, Clone, Serialize)]
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
}

impl FormSample {
    /// An unsent form: no readings, no summary. `execute_pair` fills it in.
    pub fn unmeasured(form: &'static str) -> Self {
        FormSample {
            form,
            sends: 0,
            cached_readings: Vec::new(),
            prompt_readings: Vec::new(),
            median_cached: None,
            max_cached: None,
            instances: Vec::new(),
        }
    }

    /// Whether this form has a usable measurement at all.
    pub fn measured(&self) -> bool {
        self.sends > 0
    }
}

/// The measurement for one recorded chain.
#[derive(Debug, Clone, Serialize)]
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

/// A stable fingerprint of the upstream that answered: its *routing* headers,
/// sorted, hashed. Only headers that identify an instance/namespace statically
/// are used — never per-response values like `Date` or a request id, which
/// change on every reply from one instance and would inflate the instance
/// count the longer a run lasts.
pub fn instance_fingerprint(headers: &reqwest::header::HeaderMap) -> String {
    use std::collections::BTreeMap;
    const KEYS: &[&str] = &[
        "server",
        "cf-ray",
        "x-served-by",
        "x-upstream",
        "x-cache",
        "via",
        "x-instance-id",
    ];
    let mut map: BTreeMap<&str, String> = BTreeMap::new();
    for key in KEYS {
        // A routing header that names a *cluster* is fine; the value shape
        // varies per provider, so keep it verbatim.
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
                out.push(("x-api-key", k.to_string()));
            }
            out.push(("anthropic-version", "2023-06-01".to_string()));
        }
    }
    out
}

/// Read the cached and prompt token counts out of a provider response body.
/// `None` when the body is not a JSON usage payload at all — an SSE stream, an
/// error shape, a content type we cannot parse. That is *unmeasured*, reported
/// as such, never as `0` cached.
pub fn read_usage(backend: Backend, body: &[u8]) -> Option<(u64, u64)> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = v
        .get("usage")
        .or_else(|| v.pointer("/message/usage"))
        .filter(|u| u.is_object())?;
    let sig = backend.cache_signal(body);
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
        "response carried no readable usage (streamed or non-JSON body)".to_string()
    })?;
    Ok(SendOutcome {
        cached,
        prompt,
        fingerprint,
    })
}

/// Sample one form `cfg.samples` times. Failed or unreadable sends are
/// reported on stderr and omitted from the readings: they are no measurement,
/// not a cache miss. The summary is `None` when no send succeeded.
async fn sample_form(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    label: &'static str,
    body: &serde_json::Value,
) -> FormSample {
    let mut cached_readings = Vec::new();
    let mut prompt_readings = Vec::new();
    let mut instances: Vec<String> = Vec::new();
    for _ in 0..cfg.samples {
        match send_once(client, cfg, body).await {
            Ok(o) => {
                cached_readings.push(o.cached);
                prompt_readings.push(o.prompt);
                if !instances.contains(&o.fingerprint) {
                    instances.push(o.fingerprint);
                }
            }
            Err(e) => eprintln!("replay: {label} send failed: {e}"),
        }
    }
    FormSample {
        form: label,
        sends: cached_readings.len(),
        median_cached: median(&cached_readings),
        max_cached: cached_readings.iter().copied().max(),
        cached_readings,
        prompt_readings,
        instances,
    }
}

/// Drive one A/B pair: sample each form `cfg.samples` times and summarize.
pub async fn execute_pair(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    session_id: u64,
    turn: u32,
    model: &str,
    a_body: &serde_json::Value,
    b_body: &serde_json::Value,
) -> ChainReport {
    let a_drifted = sample_form(client, cfg, "a_drifted", a_body).await;
    let b_canonical = sample_form(client, cfg, "b_canonical", b_body).await;
    ChainReport {
        session_id,
        turn,
        model: model.to_string(),
        a_drifted,
        b_canonical,
    }
}

/// Render the measurement table for one chain. Plain text, aligned, honest:
/// unmeasured forms show `—`, and the recovery figure is printed only when
/// both forms were actually measured.
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
            cell(median(&f.prompt_readings)),
            f.instances.len()
        ));
    }
    match (r.a_drifted.median_cached, r.b_canonical.median_cached) {
        (Some(a), Some(b)) if b > a => {
            out.push_str(&format!(
                "  → canonical form recovers {} cached tokens (median) over drifted\n",
                b - a
            ));
        }
        (Some(_), Some(_)) => {}
        _ => out.push_str(
            "  · not a measurement: a form had no readable reading (see stderr for the failed sends)\n",
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
    fn a_body_with_no_usage_is_unmeasured_not_zero() {
        // An SSE stream, an error shape, a plain-text body: no JSON usage,
        // so no reading — reported as such, never as `0` cached.
        assert_eq!(read_usage(Backend::OpenAi, b"{}"), None);
        assert_eq!(read_usage(Backend::OpenAi, b"data: {\"x\":1}\n\n"), None);
        assert_eq!(read_usage(Backend::OpenAi, b"not json"), None);
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
}
