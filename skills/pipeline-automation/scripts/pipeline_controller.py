#!/usr/bin/env python3
"""Deterministic controller for multi-stage skill pipelines."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "1.0"
TERMINAL_STATUSES = {"succeeded", "failed", "blocked", "skipped"}
PLACEHOLDER_PATTERNS = (
    "/REPLACE/",
    "\\REPLACE\\",
    "YOUR_ACCOUNT",
    "REPLACE_WITH",
    "TODO",
)

SLURM_ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
SLURM_SUCCESS_STATES = {"COMPLETED"}
SLURM_RESTARTABLE_STATES = {"TIMEOUT"}
SLURM_FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "confirmed": False,
    "pipeline": {
        "name": "sam_derivative_discovery",
        "project_root": "/REPLACE/with/perov-passivator",
        "run_dir": "/REPLACE/with/runs/sam_derivative_discovery",
        "journal_path": None,
        "monitor_interval_hours": 6,
        "stop_on_failure": True,
        "max_actions_per_check": 0,
    },
    "stages": [
        {
            "id": "ssl_neighbors",
            "name": "Find SSL embedding neighbors",
            "enabled": True,
            "kind": "command",
            "depends_on": [],
            "workdir": "/REPLACE/with/perov-passivator",
            "command": "python skills/ssl-neighbor-search/scripts/ssl_neighbor_search.py --config runs/sam_derivative_discovery/run_configs/ssl_neighbor_search_config.json && mkdir -p runs/sam_derivative_discovery/markers && touch runs/sam_derivative_discovery/markers/ssl_neighbors.success",
            "expected_outputs": [
                "runs/sam_derivative_discovery/outputs/ssl_neighbors_dedup.csv"
            ],
            "success_markers": [
                "runs/sam_derivative_discovery/markers/ssl_neighbors.success"
            ],
            "failure_markers": [],
            "skip_if_outputs_exist": True,
            "timeout_hours": None,
            "retry": {"max_attempts": 1, "retry_delay_minutes": 0},
        },
        {
            "id": "salt_vendor_lookup",
            "name": "Search halide salt vendors",
            "enabled": True,
            "kind": "command",
            "depends_on": ["ssl_neighbors"],
            "workdir": "/REPLACE/with/perov-passivator",
            "command": "python skills/mol-salt-vendor/scripts/mol_salt_vendor.py --config runs/sam_derivative_discovery/run_configs/mol_salt_vendor_config.json && mkdir -p runs/sam_derivative_discovery/markers && touch runs/sam_derivative_discovery/markers/salt_vendor_lookup.success",
            "expected_outputs": [
                "runs/sam_derivative_discovery/outputs/mol_salt_vendor_results.csv"
            ],
            "success_markers": [
                "runs/sam_derivative_discovery/markers/salt_vendor_lookup.success"
            ],
            "failure_markers": [],
            "skip_if_outputs_exist": True,
            "timeout_hours": None,
            "retry": {"max_attempts": 1, "retry_delay_minutes": 0},
        },
    ],
}


DEFAULT_STATUS: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "pipeline_name": "sam_derivative_discovery",
    "pipeline_config_path": "/REPLACE/with/runs/sam_derivative_discovery/pipeline_config.json",
    "status": "pending",
    "created_at": None,
    "updated_at": None,
    "last_checked": None,
    "next_check_after": None,
    "stages": {},
    "history": [],
}


class PipelineError(RuntimeError):
    """Raised for deterministic config or runtime failures."""


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(str(tmp), str(path))


def append_journal(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def tail_text(value: Optional[str], limit: int = 4000) -> str:
    if not value:
        return ""
    return value[-limit:]


def iter_strings(value: Any, prefix: str = "$") -> Iterable[Tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from iter_strings(item, f"{prefix}[{idx}]")
    elif isinstance(value, str):
        yield prefix, value


def find_placeholders(config: Dict[str, Any]) -> List[str]:
    hits = []
    for path, value in iter_strings(config):
        if any(pattern in value for pattern in PLACEHOLDER_PATTERNS):
            hits.append(f"{path}: {value}")
    return hits


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path


def has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def path_matches(path_value: str, base_dir: Path) -> List[Path]:
    resolved = resolve_path(path_value, base_dir)
    if resolved is None:
        return []
    if has_glob(str(resolved)):
        return [Path(match) for match in glob.glob(str(resolved))]
    return [resolved] if resolved.exists() else []


def all_paths_exist(paths: List[str], base_dir: Path) -> bool:
    if not paths:
        return False
    for path_value in paths:
        matches = path_matches(path_value, base_dir)
        if not matches:
            return False
    return True


def any_path_exists(paths: List[str], base_dir: Path) -> bool:
    return any(path_matches(path_value, base_dir) for path_value in paths or [])


def stage_base_dir(config: Dict[str, Any], stage: Dict[str, Any], config_path: Path) -> Path:
    pipeline = config.get("pipeline", {})
    project_root = resolve_path(pipeline.get("project_root"), config_path.parent)
    base = project_root or config_path.parent
    workdir = resolve_path(stage.get("workdir"), base)
    return workdir or base


def validate_config(config: Dict[str, Any], config_path: Path, require_confirmed: bool) -> None:
    if require_confirmed and config.get("confirmed") is not True:
        raise PipelineError('Pipeline config blocks execution until "confirmed": true.')

    if require_confirmed:
        placeholders = find_placeholders(config)
        if placeholders:
            joined = "\n".join(f"  - {hit}" for hit in placeholders)
            raise PipelineError(f"Replace placeholder values before execution:\n{joined}")

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        raise PipelineError('Missing required object: "pipeline".')

    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise PipelineError('Missing non-empty list: "stages".')

    ids: List[str] = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise PipelineError(f"Stage {idx} must be an object.")
        stage_id = stage.get("id")
        if not stage_id or not re.match(r"^[A-Za-z0-9_.-]+$", str(stage_id)):
            raise PipelineError(f"Stage {idx} has an invalid id: {stage_id!r}")
        if stage_id in ids:
            raise PipelineError(f"Duplicate stage id: {stage_id}")
        ids.append(stage_id)

        if stage.get("enabled", True) is False:
            continue

        kind = stage.get("kind")
        if kind not in {"command", "slurm"}:
            raise PipelineError(f"Stage {stage_id} has unsupported kind: {kind!r}")
        if kind == "command" and not stage.get("command"):
            raise PipelineError(f"Stage {stage_id} requires command.")
        if kind == "slurm" and not stage.get("submit_command"):
            raise PipelineError(f"Stage {stage_id} requires submit_command.")
        if kind == "slurm" and not stage.get("job_id_regex"):
            raise PipelineError(f"Stage {stage_id} requires job_id_regex.")
        if stage.get("skip_if_outputs_exist", True) and not (
            stage.get("success_markers") or []
        ):
            raise PipelineError(
                f"Stage {stage_id} enables skip_if_outputs_exist but has no "
                "success_markers. Configure a marker written only after successful "
                "completion, or set skip_if_outputs_exist to false."
            )

    id_set = set(ids)
    for stage in stages:
        for dep in stage.get("depends_on", []) or []:
            if dep not in id_set:
                raise PipelineError(f"Stage {stage.get('id')} depends on unknown stage: {dep}")

    project_root = resolve_path(pipeline.get("project_root"), config_path.parent)
    if require_confirmed and project_root and not project_root.exists():
        raise PipelineError(f"pipeline.project_root does not exist: {project_root}")


def make_stage_status(stage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": stage["id"],
        "name": stage.get("name", stage["id"]),
        "status": "skipped" if stage.get("enabled", True) is False else "pending",
        "attempts": 0,
        "job_id": None,
        "scheduler_state": None,
        "started_at": None,
        "finished_at": None,
        "next_retry_after": None,
        "last_return_code": None,
        "last_stdout_tail": "",
        "last_stderr_tail": "",
        "message": "",
    }


def initialize_status(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    now = iso_now()
    status = copy.deepcopy(DEFAULT_STATUS)
    status["pipeline_name"] = config.get("pipeline", {}).get("name", "pipeline")
    status["pipeline_config_path"] = str(config_path)
    status["created_at"] = now
    status["updated_at"] = now
    status["stages"] = {
        stage["id"]: make_stage_status(stage) for stage in config.get("stages", [])
    }
    status["history"] = []
    return status


def sync_status(config: Dict[str, Any], status: Dict[str, Any]) -> None:
    stages = status.setdefault("stages", {})
    for stage in config.get("stages", []):
        stage_id = stage["id"]
        if stage_id not in stages:
            stages[stage_id] = make_stage_status(stage)
        stages[stage_id]["name"] = stage.get("name", stage_id)
        if stage.get("enabled", True) is False and stages[stage_id]["status"] != "skipped":
            stages[stage_id]["status"] = "skipped"


def get_status_entry(status: Dict[str, Any], stage_id: str) -> Dict[str, Any]:
    return status.setdefault("stages", {}).setdefault(stage_id, {"status": "pending"})


def add_event(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    event_type: str,
    stage_id: Optional[str] = None,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    event = {
        "time": iso_now(),
        "event": event_type,
        "pipeline": config.get("pipeline", {}).get("name", "pipeline"),
        "stage_id": stage_id,
        "message": message,
    }
    if extra:
        event.update(extra)
    history = status.setdefault("history", [])
    history.append(event)
    if len(history) > 200:
        del history[:-200]

    pipeline = config.get("pipeline", {})
    base = resolve_path(pipeline.get("run_dir"), status_path.parent) or status_path.parent
    journal = resolve_path(pipeline.get("journal_path"), base)
    if journal is None:
        journal = status_path.parent / "pipeline_journal.jsonl"
    append_journal(journal, event)


def save_status(path: Path, status: Dict[str, Any]) -> None:
    status["updated_at"] = iso_now()
    write_json_atomic(path, status)


def run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def expected_success(config: Dict[str, Any], stage: Dict[str, Any], config_path: Path) -> bool:
    base = stage_base_dir(config, stage, config_path)
    expected = stage.get("expected_outputs", []) or []
    markers = stage.get("success_markers", []) or []
    if expected and not all_paths_exist(expected, base):
        return False
    if markers and not all_paths_exist(markers, base):
        return False
    return bool(expected or markers)


def restart_success_confirmed(
    config: Dict[str, Any], stage: Dict[str, Any], config_path: Path
) -> bool:
    """Require an explicit final marker before skipping work on restart."""
    if not (stage.get("success_markers") or []):
        return False
    return expected_success(config, stage, config_path)


def failure_marker_present(config: Dict[str, Any], stage: Dict[str, Any], config_path: Path) -> bool:
    base = stage_base_dir(config, stage, config_path)
    return any_path_exists(stage.get("failure_markers", []) or [], base)


def dependencies_satisfied(stage: Dict[str, Any], status: Dict[str, Any]) -> bool:
    for dep_id in stage.get("depends_on", []) or []:
        dep_status = get_status_entry(status, dep_id).get("status")
        if dep_status not in {"succeeded", "skipped"}:
            return False
    return True


def retry_limit(stage: Dict[str, Any]) -> int:
    retry = stage.get("retry") or {}
    return int(retry.get("max_attempts", 1))


def retry_delay_minutes(stage: Dict[str, Any]) -> float:
    retry = stage.get("retry") or {}
    return float(retry.get("retry_delay_minutes", 0) or 0)


def mark_retry_or_failed(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    stage: Dict[str, Any],
    message: str,
) -> None:
    entry = get_status_entry(status, stage["id"])
    entry["message"] = message
    entry["finished_at"] = iso_now()
    entry["job_id"] = None
    if int(entry.get("attempts", 0)) < retry_limit(stage):
        delay = retry_delay_minutes(stage)
        entry["status"] = "retrying"
        if delay > 0:
            next_time = datetime.now(timezone.utc) + timedelta(minutes=delay)
            entry["next_retry_after"] = next_time.replace(microsecond=0).isoformat()
        else:
            entry["next_retry_after"] = None
        add_event(config, status, status_path, "stage_retrying", stage["id"], message)
    else:
        entry["status"] = "failed"
        add_event(config, status, status_path, "stage_failed", stage["id"], message)


def mark_restart_or_blocked(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    stage: Dict[str, Any],
    message: str,
) -> None:
    """Schedule a restartable interruption or block when its budget is exhausted."""
    entry = get_status_entry(status, stage["id"])
    entry["message"] = message
    entry["finished_at"] = iso_now()
    entry["job_id"] = None
    if int(entry.get("attempts", 0)) < retry_limit(stage):
        delay = retry_delay_minutes(stage)
        entry["status"] = "retrying"
        if delay > 0:
            next_time = datetime.now(timezone.utc) + timedelta(minutes=delay)
            entry["next_retry_after"] = next_time.replace(microsecond=0).isoformat()
        else:
            entry["next_retry_after"] = None
        add_event(
            config,
            status,
            status_path,
            "stage_restart_scheduled",
            stage["id"],
            message,
        )
    else:
        entry["status"] = "blocked"
        add_event(
            config,
            status,
            status_path,
            "stage_restart_blocked",
            stage["id"],
            message,
        )


def retry_delay_elapsed(entry: Dict[str, Any]) -> bool:
    next_retry = parse_iso(entry.get("next_retry_after"))
    if next_retry is None:
        return True
    return datetime.now(timezone.utc) >= next_retry


def check_timeout(stage: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    timeout_hours = stage.get("timeout_hours")
    if timeout_hours is None:
        return False
    started = parse_iso(entry.get("started_at"))
    if started is None:
        return False
    return datetime.now(timezone.utc) - started > timedelta(hours=float(timeout_hours))


def parse_slurm_states(output: str) -> List[str]:
    """Parse unique Slurm states from newline- or pipe-delimited sacct output."""
    states: List[str] = []
    for line in output.splitlines():
        value = line.split("|", 1)[0].strip()
        if not value:
            continue
        state = value.split()[0].rstrip("+").upper()
        if state == "STATE" or state in states:
            continue
        states.append(state)
    return states


def run_command_stage(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    stage: Dict[str, Any],
    dry_run: bool,
) -> bool:
    stage_id = stage["id"]
    entry = get_status_entry(status, stage_id)
    cwd = stage_base_dir(config, stage, config_path)
    command = stage["command"]

    if dry_run:
        print(f"[dry-run] would run stage {stage_id}: {command}")
        return False

    entry["status"] = "running"
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["started_at"] = iso_now()
    entry["finished_at"] = None
    entry["message"] = f"Running command in {cwd}"
    add_event(config, status, status_path, "stage_started", stage_id, command)
    save_status(status_path, status)

    result = run_shell(command, cwd)
    entry["last_return_code"] = result.returncode
    entry["last_stdout_tail"] = tail_text(result.stdout)
    entry["last_stderr_tail"] = tail_text(result.stderr)
    entry["finished_at"] = iso_now()

    if result.returncode != 0:
        mark_retry_or_failed(
            config,
            status,
            status_path,
            stage,
            f"Command failed with return code {result.returncode}.",
        )
        return True

    if expected_success(config, stage, config_path) or not (
        stage.get("expected_outputs") or stage.get("success_markers")
    ):
        entry["status"] = "succeeded"
        entry["message"] = "Command completed successfully."
        add_event(config, status, status_path, "stage_succeeded", stage_id, entry["message"])
        return True

    mark_retry_or_failed(
        config,
        status,
        status_path,
        stage,
        "Command returned zero but expected outputs are missing.",
    )
    return True


def submit_slurm_stage(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    stage: Dict[str, Any],
    dry_run: bool,
) -> bool:
    stage_id = stage["id"]
    entry = get_status_entry(status, stage_id)
    cwd = stage_base_dir(config, stage, config_path)
    render_command = stage.get("render_command")
    submit_command = stage["submit_command"]

    if dry_run:
        if render_command:
            print(f"[dry-run] would render stage {stage_id}: {render_command}")
        print(f"[dry-run] would submit stage {stage_id}: {submit_command}")
        return False

    entry["status"] = "running"
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["started_at"] = iso_now()
    entry["finished_at"] = None
    entry["job_id"] = None
    entry["message"] = f"Submitting Slurm job in {cwd}"
    add_event(config, status, status_path, "stage_started", stage_id, submit_command)
    save_status(status_path, status)

    if render_command:
        render = run_shell(render_command, cwd)
        entry["last_return_code"] = render.returncode
        entry["last_stdout_tail"] = tail_text(render.stdout)
        entry["last_stderr_tail"] = tail_text(render.stderr)
        if render.returncode != 0:
            mark_retry_or_failed(
                config,
                status,
                status_path,
                stage,
                f"Render command failed with return code {render.returncode}.",
            )
            return True

    submit = run_shell(submit_command, cwd)
    entry["last_return_code"] = submit.returncode
    entry["last_stdout_tail"] = tail_text(submit.stdout)
    entry["last_stderr_tail"] = tail_text(submit.stderr)
    if submit.returncode != 0:
        mark_retry_or_failed(
            config,
            status,
            status_path,
            stage,
            f"Submit command failed with return code {submit.returncode}.",
        )
        return True

    combined = f"{submit.stdout}\n{submit.stderr}"
    match = re.search(stage["job_id_regex"], combined)
    if not match:
        mark_retry_or_failed(
            config,
            status,
            status_path,
            stage,
            "Submit command succeeded but no job id matched job_id_regex.",
        )
        return True

    entry["job_id"] = match.group(1)
    entry["scheduler_state"] = "SUBMITTED"
    entry["status"] = "waiting_external_job"
    entry["message"] = f"Submitted Slurm job {entry['job_id']}."
    add_event(
        config,
        status,
        status_path,
        "stage_submitted",
        stage_id,
        entry["message"],
        {"job_id": entry["job_id"]},
    )
    return True


def check_slurm_accounting_state(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    stage: Dict[str, Any],
    dry_run: bool,
) -> bool:
    """Interpret sacct state after a submitted job is no longer active."""
    stage_id = stage["id"]
    entry = get_status_entry(status, stage_id)
    job_id = entry.get("job_id")
    command = stage["accounting_command"].replace("{job_id}", str(job_id))
    cwd = stage_base_dir(config, stage, config_path)

    if dry_run:
        print(f"[dry-run] would query Slurm accounting for stage {stage_id}: {command}")
        return False

    result = run_shell(command, cwd)
    entry["last_return_code"] = result.returncode
    entry["last_stdout_tail"] = tail_text(result.stdout)
    entry["last_stderr_tail"] = tail_text(result.stderr)

    if result.returncode != 0:
        entry["message"] = (
            f"Slurm accounting query failed for job {job_id}; waiting for the next check."
        )
        return False

    states = parse_slurm_states(result.stdout)
    if not states:
        entry["scheduler_state"] = None
        entry["message"] = (
            f"Slurm job {job_id} is absent from the queue; accounting state is not "
            "available yet."
        )
        return False

    entry["scheduler_state"] = ",".join(states)
    if any(state in SLURM_ACTIVE_STATES for state in states):
        entry["message"] = (
            f"Slurm job {job_id} remains active with state(s): {entry['scheduler_state']}."
        )
        return False

    restartable_states = [
        state for state in states if state in SLURM_RESTARTABLE_STATES
    ]
    if restartable_states:
        mark_restart_or_blocked(
            config,
            status,
            status_path,
            stage,
            f"Slurm job {job_id} reached restartable state(s): "
            f"{','.join(restartable_states)}.",
        )
        return True

    failure_states = [state for state in states if state in SLURM_FAILURE_STATES]
    if failure_states:
        mark_retry_or_failed(
            config,
            status,
            status_path,
            stage,
            f"Slurm job {job_id} ended with state(s): {','.join(failure_states)}.",
        )
        return True

    if all(state in SLURM_SUCCESS_STATES for state in states):
        has_completion_requirements = bool(
            stage.get("expected_outputs") or stage.get("success_markers")
        )
        if not has_completion_requirements or expected_success(config, stage, config_path):
            entry["status"] = "succeeded"
            entry["finished_at"] = iso_now()
            entry["message"] = (
                f"Slurm job {job_id} completed and completion artifacts are valid."
            )
            add_event(
                config,
                status,
                status_path,
                "stage_succeeded",
                stage_id,
                entry["message"],
            )
        else:
            mark_retry_or_failed(
                config,
                status,
                status_path,
                stage,
                f"Slurm job {job_id} completed but expected outputs or success "
                "markers are missing.",
            )
        return True

    entry["message"] = (
        f"Slurm job {job_id} has unrecognized accounting state(s): "
        f"{entry['scheduler_state']}; waiting for the next check."
    )
    return False


def check_waiting_slurm_stage(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    stage: Dict[str, Any],
    dry_run: bool,
) -> bool:
    stage_id = stage["id"]
    entry = get_status_entry(status, stage_id)

    if failure_marker_present(config, stage, config_path):
        mark_retry_or_failed(config, status, status_path, stage, "Failure marker detected.")
        return True

    if check_timeout(stage, entry):
        mark_retry_or_failed(config, status, status_path, stage, "Stage timed out.")
        return True

    job_id = entry.get("job_id")
    if not job_id:
        entry["message"] = "Waiting for expected outputs."
        return False

    check_command = stage.get("check_command")
    accounting_command = stage.get("accounting_command")
    cwd = stage_base_dir(config, stage, config_path)
    if dry_run:
        if check_command:
            command = check_command.replace("{job_id}", str(job_id))
            print(f"[dry-run] would check active Slurm job for stage {stage_id}: {command}")
        if accounting_command:
            command = accounting_command.replace("{job_id}", str(job_id))
            print(
                f"[dry-run] would query Slurm accounting if the job is inactive: {command}"
            )
        if not check_command and not accounting_command:
            print(f"[dry-run] would keep waiting for outputs from stage {stage_id}.")
        return False

    if check_command:
        command = check_command.replace("{job_id}", str(job_id))
        result = run_shell(command, cwd)
        entry["last_return_code"] = result.returncode
        entry["last_stdout_tail"] = tail_text(result.stdout)
        entry["last_stderr_tail"] = tail_text(result.stderr)

        if result.returncode == 0 and result.stdout.strip():
            entry["scheduler_state"] = "ACTIVE"
            entry["message"] = f"Slurm job {job_id} is still visible in scheduler."
            return False

    if accounting_command:
        return check_slurm_accounting_state(
            config, status, status_path, config_path, stage, dry_run
        )

    if check_command:
        mark_retry_or_failed(
            config,
            status,
            status_path,
            stage,
            (
                f"Slurm job {job_id} is no longer visible and no accounting command "
                "is configured."
            ),
        )
        return True

    entry["message"] = "Waiting for a terminal Slurm state."
    return False


def stage_ready(stage: Dict[str, Any], entry: Dict[str, Any], status: Dict[str, Any]) -> bool:
    if stage.get("enabled", True) is False:
        return False
    if entry.get("status") not in {"pending", "retrying"}:
        return False
    if entry.get("status") == "retrying" and not retry_delay_elapsed(entry):
        return False
    return dependencies_satisfied(stage, status)


def refresh_stage_states(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    dry_run: bool,
) -> bool:
    """Refresh completion, failure, timeout, and external-job states once."""
    progress = False
    for stage in config.get("stages", []):
        entry = get_status_entry(status, stage["id"])

        if stage.get("enabled", True) is False or entry.get("status") in TERMINAL_STATUSES:
            continue

        if failure_marker_present(config, stage, config_path):
            mark_retry_or_failed(config, status, status_path, stage, "Failure marker detected.")
            progress = True
            continue

        if entry.get("status") == "waiting_external_job":
            changed = check_waiting_slurm_stage(
                config, status, status_path, config_path, stage, dry_run
            )
            progress = progress or changed
            continue

        if stage.get("skip_if_outputs_exist", True) and restart_success_confirmed(
            config, stage, config_path
        ):
            entry["status"] = "succeeded"
            entry["finished_at"] = iso_now()
            entry["message"] = "Expected outputs and final success markers already exist."
            add_event(config, status, status_path, "stage_succeeded", stage["id"], entry["message"])
            progress = True
            continue

        if check_timeout(stage, entry):
            mark_retry_or_failed(config, status, status_path, stage, "Stage timed out.")
            progress = True

    return progress


def pipeline_stops_on_failure(config: Dict[str, Any], status: Dict[str, Any]) -> bool:
    if not bool(config.get("pipeline", {}).get("stop_on_failure", True)):
        return False
    return any(
        entry.get("status") == "failed" for entry in status.get("stages", {}).values()
    )


def find_next_ready_stage(
    config: Dict[str, Any], status: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return the first stage that is ready to move the pipeline forward."""
    if pipeline_stops_on_failure(config, status):
        return None

    for stage in config.get("stages", []):
        entry = get_status_entry(status, stage["id"])
        if stage_ready(stage, entry, status):
            return stage

    return None


def execute_stage(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    stage: Dict[str, Any],
    dry_run: bool,
) -> bool:
    """Run a local stage or submit a Slurm stage."""
    kind = stage.get("kind")
    if kind == "command":
        return run_command_stage(config, status, status_path, config_path, stage, dry_run)
    if kind == "slurm":
        return submit_slurm_stage(config, status, status_path, config_path, stage, dry_run)
    raise PipelineError(f"Unsupported stage kind: {kind}")


def update_pipeline_status(config: Dict[str, Any], status: Dict[str, Any]) -> None:
    values = [entry.get("status") for entry in status.get("stages", {}).values()]
    if not values:
        status["status"] = "pending"
    elif any(value == "blocked" for value in values):
        status["status"] = "blocked"
    elif any(value == "failed" for value in values):
        status["status"] = "failed"
    elif all(value in {"succeeded", "skipped"} for value in values):
        status["status"] = "succeeded"
    elif any(value in {"waiting_external_job", "running"} for value in values):
        status["status"] = "waiting"
    else:
        status["status"] = "running"

    interval = float(config.get("pipeline", {}).get("monitor_interval_hours", 6) or 6)
    next_time = datetime.now(timezone.utc) + timedelta(hours=interval)
    status["next_check_after"] = next_time.replace(microsecond=0).isoformat()


def controller_pass(
    config: Dict[str, Any],
    status: Dict[str, Any],
    status_path: Path,
    config_path: Path,
    dry_run: bool = False,
) -> bool:
    sync_status(config, status)
    status["last_checked"] = iso_now()
    progress = refresh_stage_states(
        config, status, status_path, config_path, dry_run
    )
    actions = 0
    max_actions = int(config.get("pipeline", {}).get("max_actions_per_check", 0) or 0)

    while True:
        if max_actions and actions >= max_actions:
            break

        stage = find_next_ready_stage(config, status)
        if stage is None:
            break

        changed = execute_stage(
            config, status, status_path, config_path, stage, dry_run
        )
        progress = progress or changed
        actions += 1

        if dry_run or pipeline_stops_on_failure(config, status):
            break

    update_pipeline_status(config, status)
    return progress


def print_summary(status: Dict[str, Any]) -> None:
    print(f"Pipeline: {status.get('pipeline_name')} [{status.get('status')}]")
    print(f"Updated: {status.get('updated_at')}")
    print("")
    print(
        f"{'Stage':28} {'Status':20} {'Attempts':8} {'Job':12} "
        f"{'Slurm state':18} Message"
    )
    print("-" * 118)
    for stage_id, entry in status.get("stages", {}).items():
        print(
            f"{stage_id[:28]:28} "
            f"{str(entry.get('status', ''))[:20]:20} "
            f"{str(entry.get('attempts', 0))[:8]:8} "
            f"{str(entry.get('job_id') or '')[:12]:12} "
            f"{str(entry.get('scheduler_state') or '')[:18]:18} "
            f"{entry.get('message', '')}"
        )


def print_dry_run_preview(config: Dict[str, Any], status: Dict[str, Any], config_path: Path) -> None:
    preview = copy.deepcopy(status)
    sync_status(config, preview)

    print_summary(preview)
    print("")

    for stage in config.get("stages", []):
        entry = get_status_entry(preview, stage["id"])
        stage_id = stage["id"]

        if stage.get("enabled", True) is False:
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        if failure_marker_present(config, stage, config_path):
            print(f"[dry-run] would handle failure marker for stage {stage_id}.")
            return
        if entry.get("status") == "waiting_external_job":
            if entry.get("job_id") and (
                stage.get("check_command") or stage.get("accounting_command")
            ):
                if stage.get("check_command"):
                    command = stage["check_command"].replace(
                        "{job_id}", str(entry["job_id"])
                    )
                    print(f"[dry-run] would check active Slurm stage {stage_id}: {command}")
                if stage.get("accounting_command"):
                    command = stage["accounting_command"].replace(
                        "{job_id}", str(entry["job_id"])
                    )
                    print(
                        "[dry-run] would query Slurm accounting if inactive: "
                        f"{command}"
                    )
            else:
                print(f"[dry-run] would keep waiting for stage {stage_id}.")
            return
        if stage.get("skip_if_outputs_exist", True) and restart_success_confirmed(
            config, stage, config_path
        ):
            print(
                f"[dry-run] would mark stage {stage_id} succeeded because expected "
                "outputs and final success markers exist."
            )
            return
        if stage_ready(stage, entry, preview):
            if stage.get("kind") == "command":
                print(f"[dry-run] would run stage {stage_id}: {stage.get('command')}")
            else:
                if stage.get("render_command"):
                    print(f"[dry-run] would render stage {stage_id}: {stage.get('render_command')}")
                print(f"[dry-run] would submit stage {stage_id}: {stage.get('submit_command')}")
            return

    print("[dry-run] no ready action; pipeline is waiting, complete, or blocked by dependencies.")


def default_status_path(config_path: Path) -> Path:
    return config_path.parent / "pipeline_status.json"


def load_or_init_status(config: Dict[str, Any], config_path: Path, status_path: Path) -> Dict[str, Any]:
    if status_path.exists():
        status = load_json(status_path)
        sync_status(config, status)
        return status
    return initialize_status(config, config_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-stage skill pipeline.")
    parser.add_argument("--pipeline-config", help="Path to pipeline_config.json.")
    parser.add_argument("--status", help="Path to pipeline_status.json.")
    parser.add_argument("--write-config", help="Write a config template and exit.")
    parser.add_argument("--write-status", help="Write a blank status template and exit.")
    parser.add_argument("--init-status", action="store_true", help="Initialize status from config and exit.")
    parser.add_argument("--execute", action="store_true", help="Run the controller.")
    parser.add_argument("--dry-run", action="store_true", help="Show the next action without running commands.")
    parser.add_argument("--summary", action="store_true", help="Print current status summary.")
    parser.add_argument("--watch", action="store_true", help="Keep checking on an interval.")
    parser.add_argument("--interval-hours", type=float, help="Override watch interval in hours.")
    parser.add_argument("--max-iterations", type=int, default=0, help="Limit watch iterations; 0 means unlimited.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.write_config:
        write_json_atomic(Path(args.write_config), DEFAULT_CONFIG)
        print(f"Wrote config template: {args.write_config}")
        return 0

    if args.write_status:
        write_json_atomic(Path(args.write_status), DEFAULT_STATUS)
        print(f"Wrote status template: {args.write_status}")
        return 0

    if not args.pipeline_config:
        raise PipelineError("Provide --pipeline-config, --write-config, or --write-status.")

    config_path = Path(args.pipeline_config)
    config = load_json(config_path)
    validate_config(config, config_path, require_confirmed=args.execute or args.dry_run)

    status_path = Path(args.status) if args.status else default_status_path(config_path)

    if args.init_status:
        status = initialize_status(config, config_path)
        save_status(status_path, status)
        print(f"Initialized status: {status_path}")
        return 0

    status = load_or_init_status(config, config_path, status_path)

    if args.summary and not args.execute and not args.dry_run:
        print_summary(status)
        return 0

    if args.dry_run:
        print_dry_run_preview(config, status, config_path)
        return 0

    if not args.execute:
        print_summary(status)
        print("")
        print("Use --execute to advance the pipeline or --dry-run to preview the next action.")
        return 0

    iteration = 0
    while True:
        iteration += 1
        controller_pass(config, status, status_path, config_path, dry_run=args.dry_run)
        save_status(status_path, status)
        print_summary(status)

        if args.dry_run or not args.watch:
            return 0
        if status.get("status") in {"succeeded", "failed", "blocked"}:
            return 0
        if args.max_iterations and iteration >= args.max_iterations:
            return 0

        interval = args.interval_hours
        if interval is None:
            interval = float(config.get("pipeline", {}).get("monitor_interval_hours", 6) or 6)
        sleep_seconds = max(1, int(interval * 3600))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
