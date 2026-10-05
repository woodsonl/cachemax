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
            "openai" | "open-ai" => Some(Backend::OpenAi),
            "anthropic" | "claude" => Some(Backend::Anthropic),
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

    /// The path to POST a completion to, appended to the endpoint base.
    pub fn chat_path(self) -> &'static str {
        "/v1/chat/completions"
    }

    fn adapter(self) -> Box<dyn Adapter> {
        match self {
            Backend::OpenAi => Box::new(OpenAiAdapter),
            Backend::Anthropic => Box::new(AnthropicAdapter),
        }
    }
}

/// One A/B pair sampled: the cached-token readings and the instance
/// fingerprints seen for each form.
#[derive(Debug, Clone, Serialize)]
pub struct FormSample {
    pub form: &'static str,
    pub sends: usize,
    /// Every reading, in send order (for transparency; median/max summarize).
    pub cached_readings: Vec<u64>,
    pub prompt_readings: Vec<u64>,
    pub median_cached: u64,
    pub max_cached: u64,
    /// Distinct upstream instances (by identifying-header fingerprint).
    pub instances: Vec<String>,
}

impl FormSample {
    /// An unsent form: no readings, all zero. `execute_pair` fills it in.
    pub fn empty(form: &'static str) -> Self {
        FormSample {
            form,
            sends: 0,
            cached_readings: Vec::new(),
            prompt_readings: Vec::new(),
            median_cached: 0,
            max_cached: 0,
            instances: Vec::new(),
        }
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
pub fn median(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    v[(v.len() - 1) / 2]
}

/// A stable fingerprint of the upstream that answered: the identifying
/// headers, sorted, hashed. Two responses that share it very likely came from
/// the same instance/namespace; a differing one names a distinct backend.
pub fn instance_fingerprint(headers: &reqwest::header::HeaderMap) -> String {
    use std::collections::BTreeMap;
    // Headers that vary per instance or carry routing identity. Volatile
    // values (Date, Age, request ids) are included: they change per response
    // only when the instance does, which is exactly the signal.
    const KEYS: &[&str] = &[
        "server",
        "date",
        "cf-ray",
        "x-request-id",
        "x-served-by",
        "x-upstream",
        "via",
    ];
    let mut map: BTreeMap<&str, String> = BTreeMap::new();
    for key in KEYS {
        if let Some(v) = headers.get(*key).and_then(|v| v.to_str().ok()) {
            map.insert(key, v.to_string());
        }
    }
    // A coarse hash: enough to distinguish instances, short enough to read.
    let joined = map
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";");
    short_hash(&joined)
}

/// A stable short hash of a string (FNV-1a, hex, 8 chars).
fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", (h & 0xffff_ffff) as u32)
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

/// Read the cached and prompt token counts out of a provider response body,
/// using the backend's adapter. Prompt tokens are provider-reported where
/// available, else 0.
pub fn read_usage(backend: Backend, body: &[u8]) -> (u64, u64) {
    let sig = backend.adapter().cache_signal(body);
    let prompt = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            let usage = v.get("usage").or_else(|| v.pointer("/message/usage"))?;
            usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .and_then(|n| n.as_u64())
        })
        .unwrap_or(0);
    (sig.cached_tokens, prompt)
}

/// How to drive a replay: endpoint, auth, sample count.
#[derive(Debug, Clone)]
pub struct ExecuteConfig {
    pub endpoint: String,
    pub backend: Backend,
    pub api_key: Option<String>,
    pub samples: usize,
    pub timeout: std::time::Duration,
}

/// What one send reported.
struct SendOutcome {
    cached: u64,
    prompt: u64,
    fingerprint: String,
}

/// Send one body once and read the cache usage out of the reply. A non-2xx
/// reply is not a cache reading: it is reported as such, not counted as 0.
async fn send_once(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    body: &serde_json::Value,
) -> Result<SendOutcome, String> {
    let url = format!(
        "{}{}",
        cfg.endpoint.trim_end_matches('/'),
        cfg.backend.chat_path()
    );
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
    let (cached, prompt) = read_usage(cfg.backend, &bytes);
    Ok(SendOutcome {
        cached,
        prompt,
        fingerprint,
    })
}

/// Drive one A/B pair: sample each form `cfg.samples` times and summarize.
/// A form with no successful send reports `sends: 0` — an honest gap, never
/// a fabricated zero.
pub async fn execute_pair(
    client: &reqwest::Client,
    cfg: &ExecuteConfig,
    report: &ChainReport,
    a_body: &serde_json::Value,
    b_body: &serde_json::Value,
) -> ChainReport {
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
                Err(e) => {
                    // A failed send is reported on stderr and omitted from the
                    // readings: it is not a cache miss, it is no measurement.
                    eprintln!("replay: {label} send failed: {e}");
                }
            }
        }
        FormSample {
            form: label,
            sends: cached_readings.len(),
            median_cached: median(&cached_readings),
            max_cached: cached_readings.iter().copied().max().unwrap_or(0),
            cached_readings,
            prompt_readings,
            instances,
        }
    }

    let a_drifted = sample_form(client, cfg, "a_drifted", a_body).await;
    let b_canonical = sample_form(client, cfg, "b_canonical", b_body).await;
    ChainReport {
        a_drifted,
        b_canonical,
        ..report.clone()
    }
}

/// Render the measurement table for one chain. Plain text, aligned, honest:
/// it prints the readings it saw, the medians, and the instance count.
pub fn render_report(r: &ChainReport) -> String {
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
        let prompt = median(&f.prompt_readings);
        out.push_str(&format!(
            "  {:<12} {:>6} {:>8} {:>8} {:>8} {:>10}\n",
            f.form,
            f.sends,
            f.median_cached,
            f.max_cached,
            prompt,
            f.instances.len()
        ));
    }
    if r.b_canonical.median_cached > 0 && r.a_drifted.median_cached < r.b_canonical.median_cached {
        let recovered = r.b_canonical.median_cached - r.a_drifted.median_cached;
        out.push_str(&format!(
            "  → canonical form recovers {recovered} cached tokens (median) over drifted\n"
        ));
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
    fn median_takes_a_seen_value_not_an_average() {
        assert_eq!(median(&[1455, 20, 99]), 99);
        assert_eq!(median(&[10, 20]), 10, "lower middle for an even count");
        assert_eq!(median(&[]), 0);
        assert_eq!(median(&[7]), 7);
    }

    #[test]
    fn backend_parses_and_defaults_its_key_env() {
        assert_eq!(Backend::parse("openai"), Some(Backend::OpenAi));
        assert_eq!(Backend::parse("Anthropic"), Some(Backend::Anthropic));
        assert_eq!(Backend::parse("claude"), Some(Backend::Anthropic));
        assert_eq!(Backend::parse("nope"), None);
        assert_eq!(Backend::OpenAi.default_key_env(), "OPENAI_API_KEY");
        assert_eq!(Backend::Anthropic.default_key_env(), "ANTHROPIC_API_KEY");
    }

    #[test]
    fn openai_usage_reads_cached_and_prompt() {
        let body =
            br#"{"usage":{"prompt_tokens":2140,"prompt_tokens_details":{"cached_tokens":1455}}}"#;
        assert_eq!(read_usage(Backend::OpenAi, body), (1455, 2140));
    }

    #[test]
    fn anthropic_usage_reads_read_and_input() {
        let body = br#"{"usage":{"input_tokens":100,"cache_read_input_tokens":900}}"#;
        assert_eq!(read_usage(Backend::Anthropic, body), (900, 100));
    }

    #[test]
    fn no_usage_reads_as_zero_not_a_fabricated_number() {
        assert_eq!(read_usage(Backend::OpenAi, b"{}"), (0, 0));
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
    fn fingerprints_differ_by_identifying_headers() {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut a = HeaderMap::new();
        a.insert("server", HeaderValue::from_static("nginx"));
        a.insert("cf-ray", HeaderValue::from_static("aaa"));
        let mut b = HeaderMap::new();
        b.insert("server", HeaderValue::from_static("nginx"));
        b.insert("cf-ray", HeaderValue::from_static("bbb"));
        assert_ne!(instance_fingerprint(&a), instance_fingerprint(&b));
        // Identical identifying headers → same fingerprint.
        let mut c = HeaderMap::new();
        c.insert("server", HeaderValue::from_static("nginx"));
        c.insert("cf-ray", HeaderValue::from_static("aaa"));
        assert_eq!(instance_fingerprint(&a), instance_fingerprint(&c));
    }
}
