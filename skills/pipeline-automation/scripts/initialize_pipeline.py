#!/usr/bin/env python3
"""Initialize a pipeline status file from a pipeline config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_controller import (
    PipelineError,
    initialize_status,
    load_json,
    save_status,
    validate_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize pipeline_status.json.")
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    config_path = Path(args.pipeline_config)
    config = load_json(config_path)
    validate_config(config, config_path, require_confirmed=False)
    status = initialize_status(config, config_path)
    save_status(Path(args.status), status)
    print(f"Initialized status: {args.status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
