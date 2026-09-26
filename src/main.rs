//! cachemax — OpenAI-compatible prompt-cache measurement proxy.
//!
//! CLI surface (C8): `serve`, `--backend`, `--upstream-url`, `--check`,
//! `export`, `--verbose` (metadata only), `--rates`, `--tokenizer`.

use cachemax::adapters::{
    anthropic::AnthropicAdapter, llamacpp::LlamaCppAdapter, mlxlm::MlxLmAdapter,
    openai::OpenAiAdapter, vllm::VllmAdapter,
};
use cachemax::{export, proxy, rates::Rates, tokenize::Tokenizer};
use clap::{Parser, Subcommand};

/// Default loopback address the proxy binds.
const DEFAULT_BIND: &str = "127.0.0.1:8787";

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

    /// Prefix-hash tokenizer: an encoding name, a model name, or a vocab path.
    #[arg(long, global = true, default_value = "cl100k_base")]
    tokenizer: String,

    /// Override the built-in rate table with a JSON file.
    #[arg(long, global = true)]
    rates: Option<String>,

    /// Verbose logging. Metadata only — never message content.
    #[arg(long, global = true)]
    verbose: bool,
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
        .map_err(|e| format!("tokenizer '{}' unavailable: {e}", cli.tokenizer))?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let upstream = cli.upstream_url.clone().ok_or(
                "cachemax serve: --upstream-url is required (e.g. https://api.openai.com/v1)",
            )?;
            tracing::info!(
                backend = %cli.backend,
                upstream = %upstream,
                bind = %cli.bind,
                "cachemax listening",
            );
            dispatch_serve(&cli.backend, tokenizer, rates, upstream, &cli.bind).await
        }
        Command::Check => {
            let upstream = cli
                .upstream_url
                .as_deref()
                .ok_or("cachemax --check: --upstream-url is required")?;
            check_upstream(upstream).await
        }
        Command::Export { out } => {
            let base = format!("http://{}", cli.bind);
            let path = match out {
                Some(p) => p,
                None => {
                    let id = fetch_current_session_id(&base).await.unwrap_or(0);
                    export::default_path(id)
                }
            };
            let jsonl = fetch_export(&base).await?;
            std::fs::write(&path, jsonl)?;
            println!("wrote {path}");
            Ok(())
        }
    }
}

/// Build the adapter for `--backend` and run the proxy with it.
async fn dispatch_serve(
    backend: &str,
    tokenizer: Tokenizer,
    rates: Rates,
    upstream: String,
    bind: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        "openai" => proxy::serve(OpenAiAdapter, tokenizer, rates, upstream, bind).await,
        "anthropic" => proxy::serve(AnthropicAdapter, tokenizer, rates, upstream, bind).await,
        "llamacpp" => proxy::serve(LlamaCppAdapter, tokenizer, rates, upstream, bind).await,
        "vllm" => proxy::serve(VllmAdapter, tokenizer, rates, upstream, bind).await,
        "mlxlm" => proxy::serve(MlxLmAdapter, tokenizer, rates, upstream, bind).await,
        other => Err(format!(
            "unknown backend '{other}': expected openai | anthropic | llamacpp | vllm | mlxlm"
        )
        .into()),
    }
}

fn load_rates(path: Option<&str>) -> Result<Rates, Box<dyn std::error::Error>> {
    match path {
        None => Ok(Rates::builtin()),
        Some(p) => {
            let json = std::fs::read_to_string(p).map_err(|e| format!("--rates {p}: {e}"))?;
            Rates::from_json(&json).map_err(|e| format!("--rates {p}: {e}").into())
        }
    }
}

/// Reachability check: a HEAD/GET to the upstream's models endpoint. Any HTTP
/// response (even 401) proves the host is reachable; a transport error fails.
async fn check_upstream(upstream: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("{}/models", upstream.trim_end_matches('/'));
    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) => {
            println!(
                "cachemax --check: upstream {} reachable (HTTP {})",
                upstream,
                resp.status()
            );
            Ok(())
        }
        Err(e) => Err(format!("cachemax --check: upstream {upstream} not reachable: {e}").into()),
    }
}

async fn fetch_export(base: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("{base}/api/export");
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("no running proxy at {base}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("export failed: HTTP {}", resp.status()).into());
    }
    Ok(resp.text().await?)
}

async fn fetch_current_session_id(base: &str) -> Option<u64> {
    let url = format!("{base}/api/state");
    let v: serde_json::Value = reqwest::get(&url).await.ok()?.json().await.ok()?;
    v.get("session_id").and_then(|n| n.as_u64())
}
