"""CLI argument handling: subcommand routing and the bare-invocation default."""

from cache_max.main import _build_parser, _resolve_argv


def test_bare_invocation_defaults_to_serve():
    assert _resolve_argv([]) == ["serve"]


def test_leading_flags_route_to_serve():
    assert _resolve_argv(["--port", "9000"]) == ["serve", "--port", "9000"]
    assert _resolve_argv(["--backend", "fake"]) == ["serve", "--backend", "fake"]


def test_explicit_subcommands_are_left_alone():
    assert _resolve_argv(["serve", "--port", "9000"]) == ["serve", "--port", "9000"]
    assert _resolve_argv(["measure", "--turns", "3"]) == ["measure", "--turns", "3"]


def test_help_is_not_rewritten():
    assert _resolve_argv(["--help"]) == ["--help"]
    assert _resolve_argv(["-h"]) == ["-h"]


def test_parsed_bare_flags_run_serve():
    args = _build_parser().parse_args(_resolve_argv(["--backend", "fake", "--port", "9"]))
    assert args.command == "serve"
    assert args.backend == "fake"
    assert args.port == 9


def test_parsed_measure():
    args = _build_parser().parse_args(_resolve_argv(["measure", "--turns", "2"]))
    assert args.command == "measure"
    assert args.turns == 2
