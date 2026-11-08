"""Command-line interface (DESIGN §11).

Built on argparse (decision D7). Every command prints one JSON object on
stdout; progress goes to stderr. Exit codes:

0 ok · 1 failed quality checks · 2 bad config/input · 3 resource limits ·
4 internal error.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from globerode import __version__
from globerode.config import ConfigError, load_config
from globerode.pipeline import ResourceError, estimate, make_run_dir, run
from globerode.testworlds import WORLDS, write_world

EXIT_OK, EXIT_QUALITY, EXIT_CONFIG, EXIT_RESOURCES, EXIT_INTERNAL = 0, 1, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="globerode",
        description="Spherical terrain erosion for world maps.",
    )
    parser.add_argument("--version", action="version", version=f"globerode {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", type=Path, help="world TOML config")
        p.add_argument("--preview", action="store_true", help="2048-wide output, N=512 only")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set resolution.output_width=4096",
        )

    p_run = sub.add_parser("run", help="run the pipeline")
    add_config_args(p_run)
    p_run.add_argument("--name", help="run name (default: config name)")
    p_run.add_argument("--runs-dir", type=Path, default=Path("runs"), help="default: ./runs")
    p_run.add_argument("--quiet", action="store_true", help="no progress on stderr")

    p_est = sub.add_parser("estimate", help="memory/disk estimate only")
    add_config_args(p_est)

    p_tw = sub.add_parser("testworld", help="write a synthetic input + config")
    p_tw.add_argument("name", choices=sorted(WORLDS))
    p_tw.add_argument("out_dir", type=Path)
    p_tw.add_argument("--width", type=int, default=512, help="levels image width")
    p_tw.add_argument("--output-width", type=int, help="config output_width (default: --width)")
    return parser


def _emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2))


def _fail(code: int, message: str) -> int:
    print(f"globerode: error: {message}", file=sys.stderr)
    _emit({"ok": False, "exit_code": code, "error": message})
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        if args.command == "testworld":
            try:
                cfg_path = write_world(args.name, args.out_dir, args.width, args.output_width)
            except ValueError as e:
                raise ConfigError(str(e)) from None
            _emit({"ok": True, "config": str(cfg_path), "levels": str(args.out_dir / "levels.png")})
            return EXIT_OK

        cfg = load_config(args.config, args.overrides)
        if args.preview:
            cfg = cfg.preview()
        if args.command == "estimate":
            _emit({"ok": True} | estimate(cfg))
            return EXIT_OK
        # run
        run_dir = make_run_dir(args.runs_dir, args.name or cfg.name)
        result = run(cfg, run_dir, quiet=args.quiet)
        _emit(result.summary)
        return EXIT_OK
    except ConfigError as e:
        return _fail(EXIT_CONFIG, str(e))
    except ResourceError as e:
        return _fail(EXIT_RESOURCES, str(e))
    except Exception as e:  # last resort: report, don't hide
        traceback.print_exc(file=sys.stderr)
        return _fail(EXIT_INTERNAL, f"internal error: {e!r}")


if __name__ == "__main__":
    raise SystemExit(main())
