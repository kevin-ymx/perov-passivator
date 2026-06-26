"""
Render submit_gine_ssl_infer.slurm.template from slurm_config.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE = os.path.join(SCRIPT_DIR, "submit_gine_ssl_infer.slurm.template")
PLACEHOLDER_MARKERS = (
    "/REPLACE/",
    "/ABSOLUTE/PATH",
    "YOUR_ACCOUNT",
    "REPLACE_ME",
    "YOUR_PATH",
    "PLACEHOLDER",
)
PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass
class SlurmConfig:
    confirmed: bool = False
    job_name: str = "gine-ssl-infer"
    account: str = "YOUR_ACCOUNT"
    partition: str = "short"
    nodes: int = 4
    ntasks: int = 16
    gpus_per_node: int = 4
    cpus_per_task: int = 32
    mem: str = "0"
    time_limit: str = "24:00:00"
    qos: Optional[str] = None
    workdir: str = "/REPLACE/with/run_directory"
    bashrc: str = "/home/yeming/.bashrc"
    conda_env: str = "/scratch/yeming/conda_envs/ai4m"
    infer_script: str = "/REPLACE/with/gine-ssl-infer/scripts/gine_ssl_infer.py"
    run_config_path: str = "/REPLACE/with/run_config.json"
    shard_assignment_path: Optional[str] = None
    output_log: str = "/REPLACE/with/logs/gine-ssl-infer-%j.out"
    error_log: str = "/REPLACE/with/logs/gine-ssl-infer-%j.err"
    rendered_script_path: str = "/REPLACE/with/jobs/gine_ssl_infer.slurm"

    def validate(self) -> None:
        for label, value in asdict(self).items():
            if label in {"confirmed", "qos", "shard_assignment_path"}:
                continue
            if isinstance(value, str):
                if not value:
                    raise ValueError(f"{label} must not be empty.")
                if any(marker in value for marker in PLACEHOLDER_MARKERS):
                    raise ValueError(f"{label} contains a placeholder: {value}")
        if self.shard_assignment_path and has_placeholder(self.shard_assignment_path):
            raise ValueError(f"shard_assignment_path contains a placeholder: {self.shard_assignment_path}")
        if self.nodes < 1 or self.ntasks < 1 or self.gpus_per_node < 1 or self.cpus_per_task < 1:
            raise ValueError("nodes, ntasks, gpus_per_node, and cpus_per_task must be >= 1.")
        if self.ntasks != self.nodes * self.gpus_per_node:
            raise ValueError("ntasks must equal nodes * gpus_per_node so each GPU has exactly one worker task.")

    def placeholder_map(self, shard_assignment_path: str) -> Dict[str, str]:
        qos_directive = f"#SBATCH --qos={self.qos}" if self.qos else ""
        return {
            "JOB_NAME": self.job_name,
            "ACCOUNT": self.account,
            "PARTITION": self.partition,
            "NODES": str(self.nodes),
            "NTASKS": str(self.ntasks),
            "GPUS_PER_NODE": str(self.gpus_per_node),
            "TASKS_PER_NODE": str(self.gpus_per_node),
            "CPUS_PER_TASK": str(self.cpus_per_task),
            "MEM": self.mem,
            "TIME_LIMIT": self.time_limit,
            "QOS_DIRECTIVE": qos_directive,
            "OUTPUT_LOG": self.output_log,
            "ERROR_LOG": self.error_log,
            "BASHRC": self.bashrc,
            "CONDA_ENV": self.conda_env,
            "WORKDIR": self.workdir,
            "INFER_SCRIPT": self.infer_script,
            "RUN_CONFIG_PATH": self.run_config_path,
            "SHARD_ASSIGNMENT_PATH": shard_assignment_path,
        }


def has_placeholder(value: Any) -> bool:
    return isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS)


def default_slurm_config() -> SlurmConfig:
    return SlurmConfig()


def load_slurm_config(path: str) -> SlurmConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    fields = set(SlurmConfig.__dataclass_fields__.keys())
    return SlurmConfig(**{k: v for k, v in data.items() if k in fields})


def save_slurm_config(config: SlurmConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
        f.write("\n")


def load_run_config_for_assignment(path: str) -> Dict[str, Any]:
    if has_placeholder(path) or not os.path.isfile(path):
        raise ValueError(f"run_config_path is not a readable config: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        run = json.load(f)
    if not run.get("confirmed"):
        raise ValueError("run config confirmed is false. Approve the inference config before rendering.")
    return run


def list_shards_from_run_config(run: Dict[str, Any]) -> List[str]:
    io = run.get("io", {})
    mode = io.get("mode", "single")
    if mode == "single":
        path = io.get("input")
        if has_placeholder(path) or not os.path.isfile(path):
            raise ValueError(f"run config io.input is not a readable file: {path}")
        return [str(Path(path))]
    if mode != "shards":
        raise ValueError("run config io.mode must be 'single' or 'shards'.")
    input_dir = io.get("input_dir")
    shard_glob = io.get("shard_glob", "*.csv")
    if has_placeholder(input_dir) or not os.path.isdir(input_dir):
        raise ValueError(f"run config io.input_dir is not a readable directory: {input_dir}")
    paths = sorted(glob.glob(os.path.join(input_dir, shard_glob)))
    if not paths:
        raise ValueError(f"No shard files match {os.path.join(input_dir, shard_glob)}")
    return [str(Path(p)) for p in paths]


def count_csv_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = sum(1 for _ in f)
    return max(0, rows - 1)


def output_paths(input_path: str, output_dir: str) -> Dict[str, str]:
    stem = Path(input_path).stem
    return {
        "embeddings": str(Path(output_dir) / f"{stem}_embeddings.csv"),
        "done": str(Path(output_dir) / f"{stem}_done.json"),
    }


def count_output_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = sum(1 for _ in f)
    return max(0, rows - 1)


def shard_completed(input_path: str, output_dir: str) -> bool:
    paths = output_paths(input_path, output_dir)
    if not os.path.isfile(paths["done"]) or not os.path.isfile(paths["embeddings"]):
        return False
    try:
        with open(paths["done"], "r", encoding="utf-8") as f:
            done = json.load(f)
    except Exception:
        return False
    if not done.get("completed"):
        return False
    return count_output_rows(paths["embeddings"]) == count_csv_rows(input_path)


def default_assignment_path(config: SlurmConfig) -> str:
    if config.shard_assignment_path:
        return config.shard_assignment_path
    rendered = Path(config.rendered_script_path)
    return str(rendered.with_name(f"{rendered.stem}_worker_assignment.json"))


def write_shard_assignment(config: SlurmConfig, assignment_path: str) -> Dict[str, Any]:
    run = load_run_config_for_assignment(config.run_config_path)
    io = run.get("io", {})
    output_dir = io.get("output_dir")
    if has_placeholder(output_dir) or not output_dir:
        raise ValueError(f"run config io.output_dir contains a placeholder or is empty: {output_dir}")
    all_shards = list_shards_from_run_config(run)
    pending = []
    completed = []
    for shard in all_shards:
        if shard_completed(shard, output_dir):
            completed.append(shard)
        else:
            pending.append(shard)
    if not pending:
        raise ValueError("No unprocessed shards remain. All shards appear fully completed.")
    workers = [{"worker_index": i, "shards": []} for i in range(config.ntasks)]
    for idx, shard in enumerate(pending):
        workers[idx % config.ntasks]["shards"].append(shard)
    assignment = {
        "schema_version": "gine-ssl-infer-worker-assignment-v1",
        "run_config_path": config.run_config_path,
        "output_dir": output_dir,
        "worker_count": config.ntasks,
        "all_shards": all_shards,
        "pending_shards": pending,
        "completed_shards": completed,
        "workers": workers,
        "counts": {
            "all": len(all_shards),
            "pending": len(pending),
            "completed": len(completed),
            "workers": config.ntasks,
        },
    }
    out_dir = os.path.dirname(os.path.abspath(assignment_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2)
        f.write("\n")
    return assignment


def render_template(template_text: str, config: SlurmConfig, shard_assignment_path: str) -> str:
    mapping = config.placeholder_map(shard_assignment_path)

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in mapping:
            raise ValueError(f"Unknown template placeholder: {key}")
        return mapping[key]

    rendered = PLACEHOLDER_PATTERN.sub(replace, template_text)
    if PLACEHOLDER_PATTERN.search(rendered):
        raise ValueError("Rendered script still has unresolved placeholders.")
    return rendered


def render_slurm_script(config: SlurmConfig, template_path: str, output_path: Optional[str]) -> str:
    out_path = output_path or config.rendered_script_path
    if output_path:
        config.rendered_script_path = output_path
    assignment_path = default_assignment_path(config)
    write_shard_assignment(config, assignment_path)
    with open(template_path, "r", encoding="utf-8") as f:
        rendered = render_template(f.read(), config, assignment_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(rendered)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GIN-E SSL inference Slurm script.")
    parser.add_argument("--config", help="Slurm config JSON.")
    parser.add_argument("--write-config", metavar="PATH", help="Write Slurm config template and exit.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Slurm template path.")
    parser.add_argument("--output", help="Rendered Slurm output path.")
    parser.add_argument("--confirmed", action="store_true", help="Set confirmed=true after user approval.")
    parser.add_argument("--print", action="store_true", help="Print rendered script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_config:
        save_slurm_config(default_slurm_config(), args.write_config)
        print(f"Wrote Slurm config template: {args.write_config}")
        return
    if not args.config:
        raise SystemExit("Provide --config or --write-config.")
    try:
        config = load_slurm_config(args.config)
        if args.confirmed:
            config.confirmed = True
        if not config.confirmed:
            raise SystemExit("Render blocked: confirmed is false. Approve the full Slurm config first.")
        config.validate()
        out_path = render_slurm_script(config, args.template, args.output)
        print(f"Wrote Slurm script: {out_path}")
        print(f"Wrote worker shard assignment: {default_assignment_path(config)}")
        if args.print:
            with open(out_path, "r", encoding="utf-8") as f:
                print(f.read())
    except ValueError as exc:
        raise SystemExit(f"Invalid Slurm config: {exc}")


if __name__ == "__main__":
    main()
