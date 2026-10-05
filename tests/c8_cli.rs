//! C8 verification: the CLI surface. `--help` lists everything, `check` fails
//! loudly on an unreachable upstream and passes on a reachable one. Run via the
//! built binary so the whole arg parser + dispatch is exercised.

use std::process::Command;

fn cachemax() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cachemax"))
}

#[test]
fn help_lists_every_flag_and_command() {
    let out = cachemax().arg("--help").output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "serve",
        "check",
        "export",
        "--backend",
        "--upstream-url",
        "--bind",
        "--tokenizer",
        "--rates",
        "--verbose",
    ] {
        assert!(s.contains(needle), "--help missing {needle}");
    }
}

#[test]
fn check_fails_loudly_on_unreachable_upstream() {
    let out = cachemax()
        .args(["check", "--upstream-url", "http://127.0.0.1:1/v1"])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "unreachable upstream must exit non-zero"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("upstream unreachable"),
        "error must name the problem: {err}"
    );
    // D3 contract: problem + cause + fix + docs link.
    assert!(err.contains("cause:"), "must name the cause: {err}");
    assert!(err.contains("fix:"), "must give a fix: {err}");
    assert!(err.contains("docs:"), "must link docs: {err}");
}

#[test]
fn export_without_a_proxy_fails_clearly() {
    // No proxy listening on this bind → a clear error, no panic.
    let out = cachemax()
        .args(["export", "--bind", "127.0.0.1:1"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no running proxy"), "got: {err}");
}

#[test]
fn unknown_backend_fails_with_the_valid_list() {
    let out = cachemax()
        .args([
            "serve",
            "--backend",
            "bogus",
            "--upstream-url",
            "http://127.0.0.1:1",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown backend"), "got: {err}");
    assert!(err.contains("openai"), "must list valid backends: {err}");
}

#[test]
fn serve_on_a_taken_port_fails_with_the_error_contract() {
    // Hold a port, then ask the proxy to bind it. This must surface the D3
    // error contract (problem, cause, fix, docs link), not a raw OS error.
    let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = held.local_addr().unwrap();
    let out = cachemax()
        .args([
            "serve",
            "--upstream-url",
            "http://127.0.0.1:1",
            "--bind",
            &addr.to_string(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("address unavailable"), "got: {err}");
    assert!(err.contains("--bind"), "must offer a fix: {err}");
    assert!(err.contains("docs"), "must link docs: {err}");
}

#[test]
fn purge_reports_honestly_at_the_cli_level() {
    let dir = std::env::temp_dir().join(format!("cachemax-cli-purge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ledger = dir.join("42.jsonl");
    std::fs::write(&ledger, "{\"turn\":1}\n").unwrap();

    // A real purge names what it removed and exits 0.
    let out = cachemax()
        .args(["purge", "--ledger-dir", dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out_text.contains("purged 1 ledger file(s)"),
        "got: {out_text}"
    );

    // An already-empty dir says so; a missing dir does not pretend.
    let empty = cachemax()
        .args(["purge", "--ledger-dir", dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(empty.status.success());
    assert!(String::from_utf8_lossy(&empty.stdout).contains("already empty"));

    let missing = cachemax()
        .args(["purge", "--ledger-dir", dir.join("nope").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stdout).contains("does not exist"),
        "got: {missing:?}"
    );

    // A path that is not a directory faults with the contract.
    let file = dir.join("afile");
    std::fs::write(&file, "x").unwrap();
    let fault = cachemax()
        .args(["purge", "--ledger-dir", file.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!fault.status.success());
    let err = String::from_utf8_lossy(&fault.stderr);
    assert!(err.contains("problem:"), "D3 contract, got: {err}");
    assert!(err.contains("docs:"), "got: {err}");

    std::fs::remove_dir_all(&dir).ok();
}
