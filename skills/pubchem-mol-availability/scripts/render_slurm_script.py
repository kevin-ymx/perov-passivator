"""
Render submit_availability.slurm.template from slurm_config.json.

Mirrors the run-config confirmation gate: refuses to render unless confirmed is
true and no placeholder values remain.

Usage:
    python render_slurm_script.py --write-config slurm_config.json
    python render_slurm_script.py --config slurm_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TEMPLATE = os.path.join(_SCRIPT_DIR, "submit_availability.slurm.template")

_PLACEHOLDER_MARKERS = (
    "/ABSOLUTE/PATH",
    "YOUR_ACCOUNT",
    "PLACEHOLDER",
    "TODO",
    "CHANGEME",
    "YOUR_PATH",
)

_PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass
class SlurmJobConfig:
    confirmed: bool = False
    job_name: str = "pubchem-mol-availability"
    account: str = "YOUR_ACCOUNT"
    partition: str = "short"
    nodes: int = 1
    cpus_per_task: int = 64
    time_limit: str = "24:00:00"
    output_log: str = "/scratch/yeming/pubchem-mol-availability/logs/mol-availability-%j.out"
    error_log: str = "/scratch/yeming/pubchem-mol-availability/logs/mol-availability-%j.err"
    bashrc: str = "/home/yeming/.bashrc"
    # Private file sourced to export the LLM API key (e.g. OPENAI_API_KEY).
    # Set to null/empty to skip (e.g. when llm.enabled is false).
    secrets_file: Optional[str] = "/home/yeming/.secrets/openai"
    conda_env: str = "/scratch/yeming/conda_envs/ai4m"
    run_script: str = (
        "/home/yeming/skills/pubchem-mol-availability/scripts/run_availability.py"
    )
    run_config_path: str = "/scratch/yeming/pubchem-mol-availability/run_configs/run_config.json"
    rendered_script_path: Optional[str] = (
        "/scratch/yeming/pubchem-mol-availability/jobs/mol_availability.slurm"
    )

    def _check_value(self, value: str, field: str) -> None:
        lower = value.lower()
        for marker in _PLACEHOLDER_MARKERS:
            if marker.lower() in lower:
                raise ValueError(
                    f"slurm.{field} looks like an unconfirmed placeholder ({value!r}). "
                    "Set a real value after user confirmation."
                )

    def validate(self) -> None:
        if self.cpus_per_task < 1:
            raise ValueError("slurm.cpus_per_task must be >= 1")
        if self.nodes < 1:
            raise ValueError("slurm.nodes must be >= 1")
        if not self.job_name.strip():
            raise ValueError("slurm.job_name must not be empty")
        for field in (
            "account",
            "partition",
            "time_limit",
            "output_log",
            "error_log",
            "bashrc",
            "conda_env",
            "run_script",
            "run_config_path",
        ):
            self._check_value(getattr(self, field), field)
        if self.rendered_script_path:
            self._check_value(self.rendered_script_path, "rendered_script_path")
        if self.secrets_file:
            self._check_value(self.secrets_file, "secrets_file")

    def secrets_source_line(self) -> str:
        if self.secrets_file:
            return f"source {self.secrets_file}"
        return "# no secrets_file configured (LLM fallback expects no API key)"

    def placeholder_map(self) -> Dict[str, str]:
        return {
            "JOB_NAME": self.job_name,
            "ACCOUNT": self.account,
            "PARTITION": self.partition,
            "NODES": str(self.nodes),
            "CPUS_PER_TASK": str(self.cpus_per_task),
            "TIME_LIMIT": self.time_limit,
            "OUTPUT_LOG": self.output_log,
            "ERROR_LOG": self.error_log,
            "BASHRC": self.bashrc,
            "SECRETS_SOURCE": self.secrets_source_line(),
            "CONDA_ENV": self.conda_env,
            "RUN_SCRIPT": self.run_script,
            "RUN_CONFIG_PATH": self.run_config_path,
        }


def default_slurm_config() -> SlurmJobConfig:
    return SlurmJobConfig()


def slurm_config_from_dict(data: Dict) -> SlurmJobConfig:
    fields = {f.name for f in SlurmJobConfig.__dataclass_fields__.values()}
    return SlurmJobConfig(**{k: v for k, v in data.items() if k in fields})


def load_slurm_config_json(path: str) -> SlurmJobConfig:
    with open(path, "r", encoding="utf-8") as f:
        return slurm_config_from_dict(json.load(f))


def save_slurm_config_json(config: SlurmJobConfig, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
        f.write("\n")


def render_template(template_text: str, config: SlurmJobConfig) -> str:
    mapping = config.placeholder_map()

    def replace(match: "re.Match") -> str:
        key = match.group(1)
        if key not in mapping:
            raise ValueError(f"Unknown template placeholder: {{{{{key}}}}}")
        return mapping[key]

    rendered, count = _PLACEHOLDER_PATTERN.subn(replace, template_text)
    if _PLACEHOLDER_PATTERN.search(rendered):
        raise ValueError("Template still contains unresolved {{PLACEHOLDER}} tokens.")
    if count == 0 and "{{" in template_text:
        raise ValueError("Template uses placeholders but none were substituted.")
    return rendered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Slurm script from slurm_config.json")
    p.add_argument("--config", type=str, default=None, help="Slurm job config JSON.")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write slurm_config_template.json and exit.")
    p.add_argument("--template", type=str, default=_DEFAULT_TEMPLATE,
                   help="Path to submit_availability.slurm.template")
    p.add_argument("--output", type=str, default=None,
                   help="Output .slurm path (overrides rendered_script_path).")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark slurm config as user-confirmed (sets confirmed: true).")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass confirmed check (not for agent use).")
    p.add_argument("--print", action="store_true",
                   help="Print rendered script to stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_config:
        save_slurm_config_json(default_slurm_config(), args.write_config)
        print(f"Wrote slurm config template: {args.write_config}")
        return

    if not args.config:
        raise SystemExit("--config is required (or use --write-config).")

    config = load_slurm_config_json(args.config)
    if args.confirmed:
        config.confirmed = True

    if not config.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Render blocked: confirmed is false.\n"
            "Present the full slurm config to the user for approval, then set "
            '"confirmed": true in the JSON (or pass --confirmed). See SKILL.md.'
        )

    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(f"Invalid slurm config: {exc}")

    with open(args.template, "r", encoding="utf-8") as f:
        rendered = render_template(f.read(), config)

    out_path = args.output or config.rendered_script_path
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(rendered)
        print(f"Wrote Slurm script: {out_path}")

    if args.print or not out_path:
        print(rendered)


if __name__ == "__main__":
    main()
