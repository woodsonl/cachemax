"""Entry point for the cachemax mlx-lm sidecar.

Usage:
    cachemax-sidecar serve --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
    cachemax-sidecar measure --model ...    # opt-in warm/cold TTFT check

`serve` is the default and the only command the proxy needs: point cachemax at
it with `--backend mlxlm --upstream-url http://127.0.0.1:8080/v1`.
"""

from __future__ import annotations

import argparse
import sys

from .engine import FakeEngine, MlxEngine
from .server import build_app


def _engine(args: argparse.Namespace):
    if args.backend == "fake":
        return FakeEngine()
    return MlxEngine(model_name=args.model)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    app = build_app(_engine(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _measure(args: argparse.Namespace) -> int:
    """Opt-in warm/cold TTFT report (no speed threshold; see measure.py)."""
    from .measure import warm_cold_discrimination

    engine = _engine(args)
    result = warm_cold_discrimination(engine, turns=args.turns)
    for i, (warm, cold, ratio) in enumerate(
        zip(result.warm_ms, result.cold_ms, result.ratios)
    ):
        print(f"turn {i + 1}: warm {warm:.1f} ms  cold {cold:.1f} ms  ratio {ratio:.2f}x")
    print(f"median cold/warm ratio: {result.median_ratio:.2f}x (>1 means warm was faster)")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cachemax-sidecar")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=["mlx", "fake"], default="mlx")
        p.add_argument(
            "--model",
            default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            help="Hugging Face model id for the mlx backend",
        )

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    add_common(serve)
    serve.set_defaults(func=_serve)

    measure = sub.add_parser("measure", help="opt-in warm/cold TTFT check")
    measure.add_argument("--turns", type=int, default=5)
    add_common(measure)
    measure.set_defaults(func=_measure)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Bare invocation serves, matching the proxy's expectation.
        args = parser.parse_args(["serve", *(argv or [])])
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
