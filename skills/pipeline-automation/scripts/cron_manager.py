#!/usr/bin/env python3
"""Render, install, or remove a managed cron loop for pipeline checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PLACEHOLDER_PATTERNS = (
    "/REPLACE/",
    "\\REPLACE\\",
    "YOUR_ACCOUNT",
    "REPLACE_WITH",
    "TODO",
)


class CronError(RuntimeError):
    """Raised for cron configuration errors."""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_strings(value: Any) -> List[str]:
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(iter_strings(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(iter_strings(item))
        return out
    if isinstance(value, str):
        return [value]
    return []


def has_placeholder(value: str) -> bool:
    return any(pattern in value for pattern in PLACEHOLDER_PATTERNS)


def reject_placeholders(config: Dict[str, Any], paths: List[str]) -> None:
    values = iter_strings(config) + paths
    hits = [value for value in values if has_placeholder(value)]
    if hits:
        joined = "\n".join(f"  - {hit}" for hit in hits)
        raise CronError(f"Replace placeholder values before installing cron:\n{joined}")


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path


def sanitize_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return value.strip("-") or "pipeline"


def cron_expression(interval_hours: Optional[float], explicit: Optional[str]) -> str:
    if explicit:
        fields = explicit.split()
        if len(fields) != 5:
            raise CronError("cron_expression must have exactly five fields.")
        return explicit

    hours = float(interval_hours or 6)
    if hours <= 0:
        raise CronError("interval_hours must be positive.")
    if hours.is_integer():
        hours_int = int(hours)
        if hours_int == 24:
            return "0 0 * * *"
        if 1 <= hours_int <= 23:
            return f"0 */{hours_int} * * *"
    minutes = int(round(hours * 60))
    if minutes < 1:
        raise CronError("interval_hours is too small.")
    return f"*/{minutes} * * * *"


def quote_path(path: Path) -> str:
    return shlex.quote(str(path))


def build_cron_entry(
    config: Dict[str, Any],
    config_path: Path,
    status_path: Path,
    project_root_arg: Optional[str],
    python_arg: Optional[str],
    interval_hours_arg: Optional[float],
    cron_expression_arg: Optional[str],
    log_path_arg: Optional[str],
) -> str:
    pipeline = config.get("pipeline", {})
    scheduler = config.get("scheduler", {})

    project_root = resolve_path(
        project_root_arg or pipeline.get("project_root"),
        config_path.parent,
    )
    if project_root is None:
        raise CronError("Provide project_root in config or --project-root.")

    interval_hours = interval_hours_arg
    if interval_hours is None:
        interval_hours = scheduler.get("interval_hours", pipeline.get("monitor_interval_hours", 6))
    expression = cron_expression(interval_hours, cron_expression_arg or scheduler.get("cron_expression"))

    python_cmd = python_arg or scheduler.get("python") or "python"
    log_path = resolve_path(log_path_arg or scheduler.get("log_path"), project_root)
    if log_path is None:
        log_path = project_root / "pipeline_controller_cron.log"

    controller = "skills/pipeline-automation/scripts/pipeline_controller.py"
    command = (
        f"cd {quote_path(project_root)} && "
        f"{shlex.quote(python_cmd)} {shlex.quote(controller)} "
        f"--pipeline-config {quote_path(config_path)} "
        f"--status {quote_path(status_path)} "
        f"--execute >> {quote_path(log_path)} 2>&1"
    )
    return f"{expression} {command}"


def managed_markers(config: Dict[str, Any]) -> tuple[str, str]:
    pipeline_name = sanitize_name(config.get("pipeline", {}).get("name", "pipeline"))
    return (
        f"# BEGIN pipeline-automation:{pipeline_name}",
        f"# END pipeline-automation:{pipeline_name}",
    )


def managed_block(config: Dict[str, Any], cron_entry: str) -> str:
    begin, end = managed_markers(config)
    return "\n".join(
        [
            begin,
            "# Managed by skills/pipeline-automation/scripts/cron_manager.py",
            cron_entry,
            end,
        ]
    )


def require_linux_crontab() -> None:
    if os.name == "nt":
        raise CronError("Cron install/remove is only available on Linux/HPC targets.")


def read_crontab() -> str:
    require_linux_crontab()
    result = subprocess.run(["crontab", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout
    if "no crontab" in result.stderr.lower():
        return ""
    return ""


def write_crontab(text: str) -> None:
    require_linux_crontab()
    result = subprocess.run(["crontab", "-"], input=text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise CronError(f"crontab install failed: {result.stderr.strip()}")


def remove_block(crontab_text: str, config: Dict[str, Any]) -> str:
    begin, end = managed_markers(config)
    pattern = re.compile(
        rf"^{re.escape(begin)}\n.*?^{re.escape(end)}\n?",
        flags=re.MULTILINE | re.DOTALL,
    )
    return pattern.sub("", crontab_text).rstrip() + "\n"


def install_block(config: Dict[str, Any], cron_entry: str) -> None:
    current = read_crontab()
    cleaned = remove_block(current, config).rstrip()
    block = managed_block(config, cron_entry)
    new_text = f"{cleaned}\n\n{block}\n" if cleaned else f"{block}\n"
    write_crontab(new_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render, install, or remove a managed cron pipeline loop.")
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--python")
    parser.add_argument("--interval-hours", type=float)
    parser.add_argument("--cron-expression")
    parser.add_argument("--log-path")
    parser.add_argument("--print", action="store_true", dest="print_entry")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_cron")
    parser.add_argument("--yes", action="store_true", help="Required with --install or --remove.")
    args = parser.parse_args()

    config_path = Path(args.pipeline_config).resolve()
    status_path = Path(args.status).resolve()
    config = load_json(config_path)

    if args.list_cron:
        print(read_crontab())
        return 0

    cron_entry = build_cron_entry(
        config=config,
        config_path=config_path,
        status_path=status_path,
        project_root_arg=args.project_root,
        python_arg=args.python,
        interval_hours_arg=args.interval_hours,
        cron_expression_arg=args.cron_expression,
        log_path_arg=args.log_path,
    )

    if args.print_entry or not (args.install or args.remove):
        print(managed_block(config, cron_entry))

    if args.install:
        if config.get("confirmed") is not True:
            raise CronError('Pipeline config must have "confirmed": true before cron install.')
        if not args.yes:
            raise CronError("Refusing to install cron without --yes.")
        reject_placeholders(
            config,
            [str(config_path), str(status_path), args.project_root or "", args.log_path or ""],
        )
        install_block(config, cron_entry)
        print("Installed managed cron block.")

    if args.remove:
        if not args.yes:
            raise CronError("Refusing to remove cron without --yes.")
        current = read_crontab()
        write_crontab(remove_block(current, config))
        print("Removed managed cron block.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CronError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
