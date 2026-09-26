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
        "serve", "check", "export", "--backend", "--upstream-url", "--bind",
        "--tokenizer", "--rates", "--verbose",
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
    assert!(!out.status.success(), "unreachable upstream must exit non-zero");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("not reachable"), "error must name the problem: {err}");
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
        .args(["serve", "--backend", "bogus", "--upstream-url", "http://127.0.0.1:1"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown backend"), "got: {err}");
    assert!(err.contains("openai"), "must list valid backends: {err}");
}
