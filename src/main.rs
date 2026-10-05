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
    fn repair_mode_unknown(other: &str) -> Self {
        Fault {
            problem: "unknown repair mode",
            cause: format!("'{other}' is not a repair mode"),
            fix: "use one of: dry-run (default), on, off",
            docs: "repair-mode",
        }
    }

    fn replay_execute_requires(flag: &str, cause: String) -> Self {
        Fault {
            problem: "replay --execute is missing a required flag",
            cause: format!("{flag}: {cause}"),
            fix: "pass --upstream-url pointing at the endpoint, and --backend openai|anthropic",
            docs: "replay-execute",
        }
    }

    fn replay_execute_no_measurement(endpoint: String) -> Self {
        Fault {
            problem: "replay --execute measured nothing",
            cause: format!("no send to {endpoint} returned a readable cache reading"),
            fix: "check --upstream-url and --backend (the path differs: openai /v1/chat/completions, anthropic /v1/messages); a streaming-only endpoint answers with a body this command cannot read",
            docs: "replay-execute",
        }
    }

    fn breakpoints_require_anthropic() -> Self {
        Fault {
            problem: "breakpoint management is anthropic-only",
            cause: "--manage-breakpoints was set without --backend anthropic".into(),
            fix: "pass --backend anthropic, or drop --manage-breakpoints",
            docs: "breakpoints",
        }
    }

    fn force_requires_manage() -> Self {
        Fault {
            problem: "nowhere to force breakpoints",
            cause: "--force-breakpoints was set without --manage-breakpoints".into(),
            fix: "add --manage-breakpoints (force overrides client-placed hints)",
            docs: "breakpoints",
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

    /// Repair mode: `dry-run` (default; detect and annotate drift, never
    /// touch a byte), `on` (rewrite drifted history to the canonical
    /// serialization), `off` (no classification). Every rewrite is logged.
    #[arg(long, global = true, default_value = "dry-run")]
    repair: String,

    /// Directory for the repair ledger (default: ~/.cache/cachemax/ledger).
    /// The ledger stores the exact message content the proxy forwards and
    /// receives — locally only, never exported — so repair can extend the
    /// provider-seen prefix. Purge it with `cachemax purge`.
    #[arg(long, global = true)]
    ledger_dir: Option<String>,

    /// Keep the repair ledger in memory only; nothing is written to disk.
    #[arg(long, global = true)]
    no_ledger: bool,

    /// Anthropic only: manage cache breakpoints (`cache_control` markers)
    /// per Anthropic's incremental-breakpoint guidance — the last system
    /// block plus the last user/tool-result blocks, at most 4 per request.
    /// Requests carrying client-placed breakpoints pass through untouched.
    #[arg(long, global = true)]
    manage_breakpoints: bool,

    /// With --manage-breakpoints: re-derive breakpoints even over
    /// client-placed ones (still never exceeding the 4-block limit).
    #[arg(long, global = true)]
    force_breakpoints: bool,
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
    /// Delete the on-disk repair ledger (the exact messages the proxy
    /// forwarded and received): every `<session>.jsonl` file directly
    /// inside the ledger directory. Other files, subdirectories, and the
    /// directory itself stay; a running proxy's in-memory ledger is not
    /// touched — restart to drop it.
    Purge,
    /// Replay the recorded ledger. By default, print per-chain A/B request
    /// bodies (drifted vs canonical) as JSONL — the input for an A/B cache
    /// measurement. With --execute, drive those pairs against a real
    /// endpoint instead and report what each form measurably costs.
    Replay {
        /// Actually send each pair's two forms to the endpoint, `--n` times
        /// each, and report cached tokens (median and max) per form. Without
        /// this flag nothing is sent; the A/B bodies are only printed.
        #[arg(long)]
        execute: bool,
        /// Endpoint base URL to POST to (with --execute). The backend's
        /// chat path is appended.
        #[arg(long)]
        upstream_url: Option<String>,
        /// Wire shape of the endpoint (with --execute): `openai` or
        /// `anthropic`. Defaults to `openai`.
        #[arg(long, default_value = "openai")]
        backend: String,
        /// Environment variable holding the API key (with --execute).
        /// Defaults per backend: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
        #[arg(long)]
        api_key_env: Option<String>,
        /// Samples per form (with --execute). One reading on a routed
        /// endpoint is a routing lottery; ≥3 exposes the spread.
        #[arg(long, default_value_t = 3)]
        n: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    // Purge needs nothing but the ledger directory: no tokenizer, no
    // adapter, and it must not CREATE a ledger merely by running. It
    // ignores --no-ledger by design: it targets what a previous `serve`
    // (with or without the flag) may have written, on disk.
    if matches!(cli.command, Some(Command::Purge)) {
        let dir = ledger_dir_of(&cli);
        let report = match purge_ledger(&dir) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        print_purge(&dir, &report);
        return Ok(());
    }

    // Replay reads the ledger directory directly: it has no use for a
    // tokenizer (token columns come from the line's own counts) and must
    // not touch the running proxy — or create the directory by running.
    let replay_dir = ledger_dir_of(&cli);
    if let Some(Command::Replay {
        execute,
        upstream_url,
        backend,
        api_key_env,
        n,
    }) = cli.command
    {
        let dir = replay_dir;
        if !dir.is_dir() {
            return Err(Fault::ledger_unavailable(
                &dir.display().to_string(),
                "the directory does not exist".to_string(),
            )
            .into());
        }
        let requests = cachemax::ledger::Ledger::replay_requests(&dir)
            .map_err(|e| Fault::ledger_unavailable(&dir.display().to_string(), e.to_string()))?;
        if requests.is_empty() {
            println!(
                "no chains recorded in {}; send traffic through `cachemax serve` first",
                dir.display()
            );
            return Ok(());
        }
        if execute {
            return run_replay_execute(&requests, upstream_url, &backend, api_key_env, n.max(1))
                .await;
        }
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        use std::io::Write;
        for request in &requests {
            let line = cachemax::repair::replay_pair(request);
            // A closed pipe (`| head`) is a normal end for a stdout
            // stream, not an error to report. serde_json wraps io errors
            // — check the cause chain, not the top-level kind.
            let io_kind =
                |e: &serde_json::Error| e.io_error_kind() == Some(std::io::ErrorKind::BrokenPipe);
            if let Err(e) = serde_json::to_writer(&mut out, &line) {
                if io_kind(&e) {
                    return Ok(());
                }
                return Err(e.into());
            }
            if let Err(e) = out.write_all(b"\n") {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        return Ok(());
    }
    if let Err(e) = run(cli).await {
        eprintln!("{e}");
        std::process::exit(1);
    }
    Ok(())
}

/// What one purge removed, and what it could not.
#[derive(Debug, Default)]
struct PurgeReport {
    dir_existed: bool,
    removed: usize,
    failed: usize,
    bytes: u64,
    /// The messages for removals that failed (stderr material).
    failures: Vec<String>,
}

/// Print the purge outcome. A purge that could not remove everything is
/// a failure: exit non-zero rather than claim a clean directory.
fn print_purge(dir: &std::path::Path, report: &PurgeReport) {
    if !report.dir_existed {
        println!("nothing to purge: {} does not exist", dir.display());
    } else if report.failed > 0 {
        for f in &report.failures {
            eprintln!("{f}");
        }
        println!(
            "purged {} of {} ledger file(s) from {}; the rest could not be removed",
            report.removed,
            report.removed + report.failed,
            dir.display()
        );
        std::process::exit(1);
    } else if report.removed == 0 {
        println!("ledger {} already empty", dir.display());
    } else {
        println!(
            "purged {} ledger file(s), {} bytes, from {}",
            report.removed,
            report.bytes,
            dir.display()
        );
    }
}

/// Delete the ledger's own session files from `dir`. Only files named
/// `<digits>.jsonl` — the exact name `serve` writes (`<session>.jsonl`,
/// see `crate::ledger`) — are removed. Symlinks are never followed (a
/// link named like a session file is skipped, its target untouched);
/// other files, subdirectories, and the directory itself stay.
fn purge_ledger(dir: &std::path::Path) -> Result<PurgeReport, Box<dyn std::error::Error>> {
    if !dir.exists() {
        return Ok(PurgeReport {
            dir_existed: false,
            ..PurgeReport::default()
        });
    }
    if !dir.is_dir() {
        return Err(Fault {
            problem: "ledger directory unusable",
            cause: format!("{} is not a directory", dir.display()),
            fix: "pass the --ledger-dir the proxy runs with (default: ~/.cache/cachemax/ledger)",
            docs: "ledger",
        }
        .into());
    }
    let mut report = PurgeReport {
        dir_existed: true,
        ..PurgeReport::default()
    };
    let entries = std::fs::read_dir(dir)
        .map_err(|e| Fault::ledger_unavailable(&dir.display().to_string(), e.to_string()))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| Fault::ledger_unavailable(&dir.display().to_string(), e.to_string()))?;
        let path = entry.path();
        // A session file is a regular file named `<digits>.jsonl`. A
        // symlink named so is not its target: skipping it can never lose
        // data, and following it could.
        let is_session_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false)
            && path.extension().is_some_and(|e| e == "jsonl")
            && path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
        if !is_session_file {
            continue;
        }
        let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                report.removed += 1;
                report.bytes += bytes;
            }
            // A file that cannot be removed is reported, not fatal: the
            // rest of the purge still happened.
            Err(e) => {
                report.failed += 1;
                report
                    .failures
                    .push(format!("could not remove {}: {e}", path.display()));
            }
        }
    }
    Ok(report)
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let rates = load_rates(cli.rates.as_deref())?;
    let tokenizer = Tokenizer::resolve(&cli.tokenizer)
        .map_err(|e| Fault::tokenizer_unavailable(&cli.tokenizer, e))?;
    // Built before the command is taken out of `cli` (a partial move).
    let ledger = build_ledger(&cli)?;
    let repair = match cli.repair.as_str() {
        "off" => cachemax::repair::RepairMode::Off,
        "dry-run" => cachemax::repair::RepairMode::DryRun,
        "on" => cachemax::repair::RepairMode::On,
        other => return Err(Fault::repair_mode_unknown(other).into()),
    };

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
                // Dry-run is the default; `on` requires the operator to ask
                // for it explicitly with --repair on (or the per-request
                // x-cachemax-repair header).
                cachemax::proxy::ServeOptions {
                    repair,
                    inject_usage: !cli.no_inject_usage,
                    manage_breakpoints: cli.manage_breakpoints,
                    force_breakpoints: cli.force_breakpoints,
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
        // Handled before `run` (they need no tokenizer or upstream); the
        // compiler still wants the arms here.
        Command::Purge | Command::Replay { .. } => Ok(()),
    }
}

/// The ledger directory the command targets: `--ledger-dir` or the
/// default. Shared by `serve`, `purge`, and `replay` so all three agree
/// on the location.
fn ledger_dir_of(cli: &Cli) -> std::path::PathBuf {
    cli.ledger_dir
        .clone()
        .map_or_else(default_ledger_dir, std::path::PathBuf::from)
}

/// Drive each recorded A/B pair against a real endpoint and print what each
/// form measurably costs. Requires `--upstream-url`; auth comes from the
/// backend's environment variable (omitted only if the endpoint needs none).
async fn run_replay_execute(
    requests: &[cachemax::ledger::ReplayRequest],
    upstream_url: Option<String>,
    backend: &str,
    api_key_env: Option<String>,
    samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = upstream_url.ok_or_else(|| {
        Fault::replay_execute_requires(
            "--upstream-url",
            "--execute needs an endpoint to send to".to_string(),
        )
    })?;
    let backend = cachemax::replay::Backend::parse(backend).ok_or_else(|| {
        Fault::replay_execute_requires("--backend", "expected `openai` or `anthropic`".to_string())
    })?;
    let env_name = api_key_env.unwrap_or_else(|| backend.default_key_env().to_string());
    let api_key = std::env::var(&env_name).ok().filter(|s| !s.is_empty());

    let cfg = cachemax::replay::ExecuteConfig {
        endpoint,
        backend,
        api_key,
        samples,
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut out = std::io::stdout().lock();
    use std::io::Write;
    let mut any_measured = false;
    for request in requests {
        let pair = cachemax::repair::replay_pair(request);
        let report = cachemax::replay::execute_pair(
            &client,
            &cfg,
            request.session_id,
            request.turn,
            &request.model,
            &pair["a_drifted"],
            &pair["b_canonical"],
        )
        .await;
        any_measured |= report.a_drifted.measured() || report.b_canonical.measured();
        write!(out, "{}", cachemax::replay::render_report(&report))?;
        out.flush().ok();
    }
    if !any_measured {
        // Every send failed or was unreadable. Rendering a table of `—` and
        // exiting 0 would look like a valid null measurement; the CLI's own
        // contract is to fault loudly instead.
        return Err(Fault::replay_execute_no_measurement(cfg.endpoint.clone()).into());
    }
    Ok(())
}

/// Build the canonical ledger from `--ledger-dir` / `--no-ledger`. Default:
/// persisted under the user's cache directory, so the canonical chain
/// survives a proxy restart. `--no-ledger` keeps it in memory only.
fn build_ledger(cli: &Cli) -> Result<Arc<SharedLedger>, Box<dyn std::error::Error>> {
    let dir = ledger_dir_of(cli);
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
    // Validate the breakpoint flags before serving: a flag that silently
    // does nothing is a doc-lie, not a default.
    if (options.manage_breakpoints || options.force_breakpoints) && backend != "anthropic" {
        return Err(Fault::breakpoints_require_anthropic().into());
    }
    if options.force_breakpoints && !options.manage_breakpoints {
        return Err(Fault::force_requires_manage().into());
    }
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

    #[test]
    fn purge_removes_only_session_files() {
        let dir = std::env::temp_dir().join(format!("cachemax-purge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("7.jsonl");
        let keep = dir.join("notes.txt");
        let named = dir.join("backup.jsonl"); // not a session name
        let nested = dir.join("sub.jsonl"); // a directory, not a file
        std::fs::write(&a, "{\"turn\":1}\n").unwrap();
        std::fs::write(&keep, "user data").unwrap();
        std::fs::write(&named, "my precious data").unwrap();
        std::fs::create_dir_all(&nested).unwrap();

        let report = purge_ledger(&dir).unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.failed, 0);
        assert!(report.bytes > 0);
        assert!(!a.exists(), "session files are gone");
        assert!(named.exists(), "non-session jsonl stays");
        assert!(keep.exists(), "non-ledger files stay");
        assert!(nested.exists(), "subdirectories stay");
        assert!(dir.exists(), "the directory itself stays");

        // A second purge is a clean no-op; a missing dir reports honestly.
        let again = purge_ledger(&dir).unwrap();
        assert_eq!(again.removed, 0);
        let missing = purge_ledger(&dir.join("nope")).unwrap();
        assert!(!missing.dir_existed && missing.removed == 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn purge_skips_a_symlink_named_like_a_session_file() {
        // A link named 7.jsonl pointing at user data must be skipped, not
        // followed-then-removed: the target survives, the report is clean.
        let dir = std::env::temp_dir().join(format!("cachemax-purge-sym-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("target.txt");
        std::fs::write(&target, "keep me").unwrap();
        std::os::unix::fs::symlink(&target, dir.join("7.jsonl")).unwrap();

        let report = purge_ledger(&dir).unwrap();
        assert_eq!(report.removed, 0, "a link is not a session file");
        assert!(target.exists(), "the link's target is untouched");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_of_a_file_path_faults() {
        let f = std::env::temp_dir().join(format!("cachemax-purge-file-{}", std::process::id()));
        std::fs::write(&f, "x").unwrap();
        assert!(purge_ledger(&f).is_err());
        std::fs::remove_file(&f).ok();
    }
}
