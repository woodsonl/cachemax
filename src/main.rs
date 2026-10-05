//! cachemax — OpenAI-compatible prompt-cache measurement proxy.
//!
//! CLI surface (C8): `serve`, `--backend`, `--upstream-url`, `--check`,
//! `export`, `--verbose` (metadata only), `--rates`, `--tokenizer`.

use cachemax::adapters::{
    anthropic::AnthropicAdapter, llamacpp::LlamaCppAdapter, mlxlm::MlxLmAdapter,
    openai::OpenAiAdapter, vllm::VllmAdapter,
};
use cachemax::ledger::SharedLedger;
use cachemax::{export, proxy, rates::Rates, tokenize::Tokenizer};
use clap::{Parser, Subcommand};
use std::sync::Arc;

/// Default loopback address the proxy binds.
const DEFAULT_BIND: &str = "127.0.0.1:8787";

/// Docs base for error-contract links.
const DOCS: &str = "https://github.com/woodsonl/cachemax/blob/main/docs/troubleshooting.md";

/// The D3 error contract: every failure names the problem, the cause, the fix,
/// and a docs link. No raw panic by default.
#[derive(Debug)]
pub struct Fault {
    pub problem: &'static str,
    pub cause: String,
    pub fix: &'static str,
    pub docs: &'static str,
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "problem: {}\n  cause: {}\n  fix: {}\n  docs: {}#{}",
            self.problem, self.cause, self.fix, DOCS, self.docs
        )
    }
}

impl std::error::Error for Fault {}

impl Fault {
    fn upstream_unreachable(cause: String) -> Self {
        Fault {
            problem: "upstream unreachable",
            cause,
            fix: "check --upstream-url and your network; a reachable host that returns 401 is fine",
            docs: "upstream-unreachable",
        }
    }
    fn upstream_required() -> Self {
        Fault {
            problem: "no upstream configured",
            cause: "--upstream-url was not provided".into(),
            fix: "pass --upstream-url, e.g. https://api.openai.com/v1 or https://openrouter.ai/api/v1",
            docs: "upstream-unreachable",
        }
    }
    fn tokenizer_unavailable(name: &str, cause: String) -> Self {
        Fault {
            problem: "tokenizer unavailable",
            cause: format!("'{name}': {cause}"),
            fix: "use --tokenizer cl100k_base (the default), another encoding name, or a known model name",
            docs: "tokenizer-unavailable",
        }
    }
    fn unknown_backend(other: &str) -> Self {
        Fault {
            problem: "unknown backend",
            cause: format!("'{other}' is not a known adapter"),
            fix: "use one of: openai, anthropic, llamacpp, vllm, mlxlm",
            docs: "no-cache-signal",
        }
    }
    fn no_proxy(base: &str, cause: String) -> Self {
        Fault {
            problem: "no running proxy",
            cause: format!("nothing answered at {base}: {cause}"),
            fix: "start the proxy first with `cachemax serve --upstream-url ...`",
            docs: "no-running-proxy",
        }
    }
    fn rates_load(path: &str, cause: String) -> Self {
        Fault {
            problem: "rates file unusable",
            cause: format!("{path}: {cause}"),
            fix: "provide a JSON file of the form {\"models\": {\"<model-prefix>\": {\"input_per_mtok\": N, \"output_per_mtok\": N}}}",
            docs: "rates-file",
        }
    }
    fn bind_unavailable(bind: &str, cause: String) -> Self {
        Fault {
            problem: "address unavailable",
            cause: format!("could not bind {bind}: {cause}"),
            fix: "another process may hold the port; pass a different --bind, e.g. --bind 127.0.0.1:8788",
            docs: "address-unavailable",
        }
    }
    fn ledger_unavailable(path: &str, cause: String) -> Self {
        Fault {
            problem: "ledger directory unusable",
            cause: format!("{path}: {cause}"),
            fix: "pass a writable --ledger-dir, or --no-ledger to keep the ledger in memory only",
            docs: "ledger",
        }
    }
}

#[derive(Parser)]
#[command(
    name = "cachemax",
    version,
    about = "Measure prompt-cache reuse in front of an LLM"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Backend adapter: openai | anthropic | llamacpp | vllm | mlxlm.
    #[arg(long, global = true, default_value = "openai")]
    backend: String,

    /// Provider endpoint. OpenRouter: https://openrouter.ai/api/v1
    #[arg(long, global = true)]
    upstream_url: Option<String>,

    /// Loopback address to bind.
    #[arg(long, global = true, default_value = DEFAULT_BIND)]
    bind: String,

    /// Prefix-hash tokenizer: an encoding name or a model name.
    #[arg(long, global = true, default_value = "cl100k_base")]
    tokenizer: String,

    /// Override the built-in rate table with a JSON file.
    #[arg(long, global = true)]
    rates: Option<String>,

    /// Verbose logging. Metadata only — never message content.
    #[arg(long, global = true)]
    verbose: bool,

    /// Do not add `stream_options.include_usage` to OpenAI-dialect streaming
    /// requests. Injecting it is what lets cachemax see the terminal usage
    /// chunk (and thus cache figures); disable only for strict pass-through.
    #[arg(long, global = true)]
    no_inject_usage: bool,

    /// Directory for the repair ledger (default: ~/.cache/cachemax/ledger).
    /// The ledger stores the exact message content the proxy forwards and
    /// receives — locally only, never exported — so repair can extend the
    /// provider-seen prefix. Delete the directory to purge it.
    #[arg(long, global = true)]
    ledger_dir: Option<String>,

    /// Keep the repair ledger in memory only; nothing is written to disk.
    #[arg(long, global = true)]
    no_ledger: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Run the proxy.
    Serve,
    /// Check that the upstream is reachable; fail loudly if not.
    Check,
    /// Export the running proxy's current session as metrics-only JSONL.
    Export {
        /// Write to this path (default: ./cachemax-<session>.jsonl).
        #[arg(long)]
        out: Option<String>,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    if let Err(e) = run(cli).await {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let rates = load_rates(cli.rates.as_deref())?;
    let tokenizer = Tokenizer::resolve(&cli.tokenizer)
        .map_err(|e| Fault::tokenizer_unavailable(&cli.tokenizer, e))?;
    // Built before the command is taken out of `cli` (a partial move).
    let ledger = build_ledger(&cli)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let upstream = cli
                .upstream_url
                .clone()
                .ok_or_else(Fault::upstream_required)?;
            dispatch_serve(
                &cli.backend,
                tokenizer,
                rates,
                ledger,
                upstream,
                &cli.bind,
                // Dry-run is the product default: classify and annotate,
                // never rewrite. The `--repair` flag lands with the rewrite
                // batch; until then the mode is fixed here.
                cachemax::proxy::ServeOptions {
                    repair: cachemax::repair::RepairMode::DryRun,
                    inject_usage: !cli.no_inject_usage,
                },
            )
            .await
        }
        Command::Check => {
            let upstream = cli
                .upstream_url
                .as_deref()
                .ok_or_else(Fault::upstream_required)?;
            check_upstream(upstream).await
        }
        Command::Export { out } => {
            let base = format!("http://{}", cli.bind);
            let jsonl = fetch_export(&base).await?;
            let path = match out {
                Some(p) => p,
                None => {
                    // Name the file after the session actually exported. Faulting
                    // to id 0 would silently write `cachemax-0.jsonl` (and clobber
                    // a prior one) when the session id is unknown.
                    let id = first_session_id(&jsonl).ok_or_else(|| Fault {
                        problem: "nothing to export",
                        cause: "the proxy has no finalized requests yet".into(),
                        fix: "send a request through the proxy, then export; or pass --out to choose a path",
                        docs: "no-running-proxy",
                    })?;
                    export::default_path(id)
                }
            };
            std::fs::write(&path, jsonl)?;
            println!("wrote {path}");
            Ok(())
        }
    }
}

/// Build the canonical ledger from `--ledger-dir` / `--no-ledger`. Default:
/// persisted under the user's cache directory, so the canonical chain
/// survives a proxy restart. `--no-ledger` keeps it in memory only.
fn build_ledger(cli: &Cli) -> Result<Arc<SharedLedger>, Box<dyn std::error::Error>> {
    let dir = match &cli.ledger_dir {
        Some(p) => std::path::PathBuf::from(p),
        None => default_ledger_dir(),
    };
    let ledger = if cli.no_ledger {
        SharedLedger::new()
    } else {
        SharedLedger::on_disk(dir.clone())
            .map_err(|e| Fault::ledger_unavailable(&dir.display().to_string(), e.to_string()))?
    };
    Ok(Arc::new(ledger))
}

/// The default ledger directory: `~/.cache/cachemax/ledger` (honoring
/// `XDG_CACHE_HOME` when set). Written without a platform crate: two env
/// vars cover the supported platforms.
fn default_ledger_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let cache = std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| format!("{home}/.cache"));
    std::path::PathBuf::from(cache)
        .join("cachemax")
        .join("ledger")
}

/// Build the adapter for `--backend` and run the proxy with it. The listener is
/// bound here so a bad/unavailable address surfaces as the D3 error contract
/// rather than a raw OS error.
async fn dispatch_serve(
    backend: &str,
    tokenizer: Tokenizer,
    rates: Rates,
    ledger: Arc<SharedLedger>,
    upstream: String,
    bind: &str,
    options: cachemax::proxy::ServeOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .map_err(|e| Fault::bind_unavailable(bind, e.to_string()))?;
    match backend {
        "openai" => {
            proxy::serve(
                OpenAiAdapter,
                tokenizer,
                rates,
                ledger,
                upstream.clone(),
                options.clone(),
                listener,
            )
            .await
        }
        "anthropic" => {
            proxy::serve(
                AnthropicAdapter,
                tokenizer,
                rates,
                ledger,
                upstream.clone(),
                options.clone(),
                listener,
            )
            .await
        }
        "llamacpp" => {
            proxy::serve(
                LlamaCppAdapter,
                tokenizer,
                rates,
                ledger,
                upstream.clone(),
                options.clone(),
                listener,
            )
            .await
        }
        "vllm" => {
            proxy::serve(
                VllmAdapter,
                tokenizer,
                rates,
                ledger,
                upstream.clone(),
                options.clone(),
                listener,
            )
            .await
        }
        "mlxlm" => {
            proxy::serve(
                MlxLmAdapter,
                tokenizer,
                rates,
                ledger,
                upstream.clone(),
                options.clone(),
                listener,
            )
            .await
        }
        other => Err(Fault::unknown_backend(other).into()),
    }
}

fn load_rates(path: Option<&str>) -> Result<Rates, Box<dyn std::error::Error>> {
    match path {
        None => Ok(Rates::builtin()),
        Some(p) => {
            let json =
                std::fs::read_to_string(p).map_err(|e| Fault::rates_load(p, e.to_string()))?;
            Rates::from_json(&json).map_err(|e| Fault::rates_load(p, e).into())
        }
    }
}

/// Reachability check: a GET to the upstream's models endpoint. Any HTTP
/// response (even 401) proves the host is reachable; a transport error fails.
async fn check_upstream(upstream: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("{}/models", proxy::versioned_base(upstream));
    // Bounded: a check that hangs is a failure, not a wait.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    match client.get(&url).send().await {
        Ok(resp) => {
            println!(
                "cachemax --check: upstream {} reachable (HTTP {})",
                upstream,
                resp.status()
            );
            Ok(())
        }
        Err(e) => Err(Fault::upstream_unreachable(e.to_string()).into()),
    }
}

async fn fetch_export(base: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("{base}/api/export");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Fault::no_proxy(base, e.to_string()))?;
    if !resp.status().is_success() {
        return Err(Fault {
            problem: "export failed",
            cause: format!("HTTP {}", resp.status()),
            fix: "check the proxy logs for the failed request",
            docs: "no-running-proxy",
        }
        .into());
    }
    Ok(resp.text().await?)
}

/// The session id of the first record in an exported JSONL body, if any.
fn first_session_id(jsonl: &str) -> Option<u64> {
    let line = jsonl.lines().find(|l| !l.trim().is_empty())?;
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    v.get("session_id").and_then(|n| n.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_session_id_reads_the_first_record() {
        let jsonl = "{\"session_id\":7,\"turn\":0}\n{\"session_id\":7,\"turn\":1}\n";
        assert_eq!(first_session_id(jsonl), Some(7));
    }

    #[test]
    fn first_session_id_is_none_for_empty_export() {
        // Empty (no records): the caller must surface that as a fault, not
        // silently write `cachemax-0.jsonl`.
        assert_eq!(first_session_id(""), None);
        assert_eq!(first_session_id("\n  \n"), None);
    }
}
