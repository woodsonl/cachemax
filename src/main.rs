//! cachemax — OpenAI-compatible prompt-cache measurement proxy.
//!
//! CLI surface (C8): `serve`, `--backend`, `--upstream-url`, `--check`,
//! `export`, `--verbose` (metadata only).

// The module skeleton is in place before the C1-C8 wiring uses it, so most of
// the API is currently unreferenced. Remove this once the proxy and dashboard
// are wired end to end.
#![allow(dead_code)]

mod adapters;
mod dashboard;
mod export;
mod proxy;
mod record;
mod sessions;
mod tokenize;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cachemax", version, about = "Measure prompt-cache reuse in front of an LLM")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Backend adapter: openai | anthropic | llamacpp | vllm | mlxlm.
    #[arg(long, global = true, default_value = "openai")]
    backend: String,

    /// Provider endpoint. OpenRouter: https://openrouter.ai/api/v1
    #[arg(long, global = true)]
    upstream_url: Option<String>,

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
    /// Export a session's metrics as JSONL.
    Export {
        /// Session id to export.
        session: u64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            tracing::info!(
                backend = %cli.backend,
                upstream = ?cli.upstream_url,
                "cachemax serve — not implemented yet (see docs/designs/cachemax-measurement-core.md)"
            );
            Err("serve is not implemented yet".into())
        }
        Command::Check => {
            let url = cli
                .upstream_url
                .as_deref()
                .unwrap_or("(unset — provide --upstream-url)");
            Err(format!("cachemax --check: upstream {url} not reachable (not implemented yet)").into())
        }
        Command::Export { session } => {
            let path = export::default_path(session);
            Err(format!("cachemax export: would write {path} (not implemented yet)").into())
        }
    }
}
