"""Command-line interface.

Built on argparse (decision D7). Commands are added as the plan steps land.
"""

import argparse
from collections.abc import Sequence

from globerode import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="globerode",
        description="Spherical terrain erosion for world maps.",
    )
    parser.add_argument("--version", action="version", version=f"globerode {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
