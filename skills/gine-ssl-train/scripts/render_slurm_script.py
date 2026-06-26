"""
Render submit_gine_ssl_train.slurm.template from slurm_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE = os.path.join(SCRIPT_DIR, "submit_gine_ssl_train.slurm.template")
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
    job_name: str = "gine-ssl-train"
    account: str = "YOUR_ACCOUNT"
    partition: str = "short"
    nodes: int = 1
    ntasks: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 32
    mem: str = "0"
    time_limit: str = "24:00:00"
    qos: Optional[str] = None
    workdir: str = "/REPLACE/with/run_directory"
    bashrc: str = "/home/yeming/.bashrc"
    conda_env: str = "/scratch/yeming/conda_envs/ai4m"
    train_script: str = "/REPLACE/with/gine-ssl-train/scripts/gine_ssl_train.py"
    run_config_path: str = "/REPLACE/with/run_config.json"
    output_log: str = "/REPLACE/with/logs/gine-ssl-train-%j.out"
    error_log: str = "/REPLACE/with/logs/gine-ssl-train-%j.err"
    rendered_script_path: str = "/REPLACE/with/jobs/gine_ssl_train.slurm"

    def validate(self) -> None:
        for label, value in asdict(self).items():
            if label in {"confirmed", "qos"}:
                continue
            if isinstance(value, str):
                if not value:
                    raise ValueError(f"{label} must not be empty.")
                if any(marker in value for marker in PLACEHOLDER_MARKERS):
                    raise ValueError(f"{label} contains a placeholder: {value}")
        if self.nodes < 1 or self.ntasks < 1 or self.gpus_per_node < 1 or self.cpus_per_task < 1:
            raise ValueError("nodes, ntasks, gpus_per_node, and cpus_per_task must be >= 1.")

    def placeholder_map(self) -> Dict[str, str]:
        qos_directive = f"#SBATCH --qos={self.qos}" if self.qos else ""
        return {
            "JOB_NAME": self.job_name,
            "ACCOUNT": self.account,
            "PARTITION": self.partition,
            "NODES": str(self.nodes),
            "NTASKS": str(self.ntasks),
            "GPUS_PER_NODE": str(self.gpus_per_node),
            "CPUS_PER_TASK": str(self.cpus_per_task),
            "MEM": self.mem,
            "TIME_LIMIT": self.time_limit,
            "QOS_DIRECTIVE": qos_directive,
            "OUTPUT_LOG": self.output_log,
            "ERROR_LOG": self.error_log,
            "BASHRC": self.bashrc,
            "CONDA_ENV": self.conda_env,
            "WORKDIR": self.workdir,
            "TRAIN_SCRIPT": self.train_script,
            "RUN_CONFIG_PATH": self.run_config_path,
        }


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


def render_template(template_text: str, config: SlurmConfig) -> str:
    mapping = config.placeholder_map()

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
    with open(template_path, "r", encoding="utf-8") as f:
        rendered = render_template(f.read(), config)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(rendered)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GIN-E SSL training Slurm script.")
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
        if args.print:
            with open(out_path, "r", encoding="utf-8") as f:
                print(f.read())
    except ValueError as exc:
        raise SystemExit(f"Invalid Slurm config: {exc}")


if __name__ == "__main__":
    main()
