#!/usr/bin/env python3
"""Print a compact status report for a pipeline status file."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_controller import load_json, print_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Print pipeline status summary.")
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    print_summary(load_json(Path(args.status)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
