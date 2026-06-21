"""
Config-driven Lewis base–Pb surface binding-energy (Eb) prediction for molecules.

Given query molecules (entered inline by the user, or loaded from a CSV), this
predicts the binding energy (eV) of each molecule, acting as a Lewis base,
coordinating to undercoordinated Pb on the FAPbI3 perovskite surface, using a trained
GIN-E downstream model. It is a thin wrapper around the existing
`inference_Eb.py` `BindingEnergyPredictor` (`predict_batch`), adding:

  - inline OR CSV query input (chosen from the run config),
  - a config + confirmation gate (same pattern as the other skills),
  - a fixed-schema output table with `cid` / `smiles` passthrough columns that
    feed directly into downstream skills (e.g. mol-salt-vendor), plus optional
    sorting by predicted Eb and a strong-binder threshold flag.

A single JSON run config fully specifies a run. Execution is blocked until
"confirmed": true (present the config to the user for approval first).

Usage:
    python eb_pbcoord_predict.py --write-config run_config.json
    python eb_pbcoord_predict.py --config run_config.json
    python eb_pbcoord_predict.py --config run_config.json --smiles "NCCc1ccccc1,OCCO"

Prerequisites: the trained downstream + finetuned GIN-E checkpoints, and the
backend Python env (torch, torch-geometric, RDKit). Runs where those live
(workstation / HPC); it is NOT a portable, dependency-free skill.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

PLACEHOLDER_MARKERS = ("/REPLACE/", "/ABSOLUTE/PATH", "REPLACE_ME")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class IOConfig:
    # Query input: provide EITHER inline `queries` OR a `query_csv` path.
    queries: List[Dict[str, Any]] = field(default_factory=list)
    query_csv: Optional[str] = None
    name_column: str = "molecule_name"
    cid_column: str = "cid"
    smiles_column: str = "smiles"

    # Backend (encoder + downstream model code). null = auto-detect from script.
    project_root: Optional[str] = None

    # Output.
    output_csv: str = "eb_pbcoord_predictions.csv"
    sort_by_energy: bool = True

    def validate(self) -> None:
        has_inline = bool(self.queries)
        has_csv = bool(self.query_csv)
        if has_inline and has_csv:
            raise ValueError("Provide either io.queries (inline) OR io.query_csv, not both.")
        if not has_inline and not has_csv:
            raise ValueError("No queries: set io.queries (inline) or io.query_csv.")
        if has_csv:
            if _has_placeholder(self.query_csv):
                raise ValueError(f"io.query_csv has a placeholder: {self.query_csv}")
            if not os.path.isfile(self.query_csv):
                raise ValueError(f"io.query_csv not found: {self.query_csv}")


@dataclass
class ModelConfig:
    # No portable defaults — checkpoints must be provided by the user.
    downstream_checkpoint: Optional[str] = None
    gin_e_checkpoint: Optional[str] = None
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    batch_size: int = 64
    # If set, rows with predicted Eb <= threshold (eV) are flagged as strong binders.
    energy_threshold: Optional[float] = None

    def validate(self) -> None:
        for label, val in (
            ("downstream_checkpoint", self.downstream_checkpoint),
            ("gin_e_checkpoint", self.gin_e_checkpoint),
        ):
            if not val:
                raise ValueError(f"model.{label} is required.")
            if _has_placeholder(val):
                raise ValueError(f"model.{label} still contains a placeholder: {val}")
            if not os.path.isfile(val):
                raise ValueError(f"model.{label} not found: {val}")
        if self.batch_size < 1:
            raise ValueError("model.batch_size must be >= 1.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("model.device must be one of: auto, cuda, cpu.")


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def default_run_config() -> RunConfig:
    return RunConfig(
        confirmed=False,
        io=IOConfig(
            queries=[{"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": None}],
            query_csv=None,
        ),
        model=ModelConfig(
            downstream_checkpoint="/REPLACE/with/downstream_best_model.pt",
            gin_e_checkpoint="/REPLACE/with/gin_e_finetuned.pt",
        ),
    )


def _has_placeholder(value: Optional[str]) -> bool:
    return bool(value) and any(m in value for m in PLACEHOLDER_MARKERS)


def load_run_config_json(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    io = IOConfig(**{**asdict(IOConfig()), **data.get("io", {})})
    model = ModelConfig(**{**asdict(ModelConfig()), **data.get("model", {})})
    return RunConfig(confirmed=bool(data.get("confirmed", False)), io=io, model=model)


def save_run_config_json(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)


# --------------------------------------------------------------------------- #
# Query loading -> normalized rows: {name, cid, smiles}
# --------------------------------------------------------------------------- #
def _norm_row(name: str, cid: Any, smiles: str) -> Dict[str, str]:
    return {
        "name": (name or "").strip(),
        "cid": ("" if cid is None else str(cid)).strip(),
        "smiles": (smiles or "").strip(),
    }


def load_inline_queries(queries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for q in queries:
        name = q.get("name") or q.get("molecule_name") or ""
        smiles = q.get("smiles") or q.get("SMILES") or ""
        rows.append(_norm_row(name, q.get("cid"), smiles))
    return rows


def load_csv_queries(path: str, name_col: str, cid_col: str, smiles_col: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            smiles = r.get(smiles_col) or r.get("SMILES") or r.get("smiles") or ""
            name = r.get(name_col) or r.get("molecule_name") or ""
            cid = r.get(cid_col) or r.get("cid") or ""
            rows.append(_norm_row(name, cid, smiles))
    return rows


def build_query_rows(io: IOConfig) -> List[Dict[str, str]]:
    if io.queries:
        return load_inline_queries(io.queries)
    return load_csv_queries(io.query_csv, io.name_column, io.cid_column, io.smiles_column)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
BASE_COLUMNS = ["name", "cid", "smiles", "predicted_binding_energy_eV", "prediction_status"]


def write_predictions_csv(
    output_path: str,
    rows: List[Dict[str, str]],
    energy_threshold: Optional[float],
) -> None:
    columns = list(BASE_COLUMNS)
    if energy_threshold is not None:
        columns.append("below_threshold")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


# --------------------------------------------------------------------------- #
# Backend import wiring
# --------------------------------------------------------------------------- #
def find_project_root(explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f"io.project_root not found: {explicit}")
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "inference_Eb.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit(
        "Could not locate the backend root (inference_Eb.py). "
        "Set io.project_root in the config."
    )


def import_predictor(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from inference_Eb import BindingEnergyPredictor  # type: ignore
    except Exception as exc:  # torch/RDKit/PyG missing, etc.
        raise SystemExit(
            f"Failed to import BindingEnergyPredictor from {project_root}: {exc}\n"
            "Ensure the backend env (torch, torch-geometric, rdkit) is installed."
        )
    return BindingEnergyPredictor


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lewis base–Pb binding energy (Eb) on FAPbI3 for user-specified molecules."
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON run config (confirmed/io/model). Blocked until confirmed: true.")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write a config template (confirmed: false) and exit.")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark run as user-confirmed. Use only after the full config was approved.")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass the confirmed gate (not for agent use).")

    p.add_argument("--smiles", type=str, default=None,
                   help="Comma-separated SMILES; overrides queries with these inline (no CSV).")
    p.add_argument("--query-csv", type=str, default=None)
    p.add_argument("--downstream-checkpoint", type=str, default=None)
    p.add_argument("--gin-e-checkpoint", type=str, default=None)
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--no-sort", action="store_true",
                   help="Do not sort output by predicted Eb (keep input order).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--energy-threshold", type=float, default=None,
                   help="Flag rows with predicted Eb <= this value (eV) as strong binders.")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    run = load_run_config_json(args.config) if args.config else default_run_config()

    if args.smiles is not None:
        smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
        run.io.queries = [{"name": "", "smiles": s, "cid": None} for s in smiles_list]
        run.io.query_csv = None
    if args.query_csv is not None:
        run.io.query_csv = args.query_csv
        run.io.queries = []
    if args.project_root is not None:
        run.io.project_root = args.project_root
    if args.output_csv is not None:
        run.io.output_csv = args.output_csv
    if args.no_sort:
        run.io.sort_by_energy = False

    if args.downstream_checkpoint is not None:
        run.model.downstream_checkpoint = args.downstream_checkpoint
    if args.gin_e_checkpoint is not None:
        run.model.gin_e_checkpoint = args.gin_e_checkpoint
    if args.batch_size is not None:
        run.model.batch_size = args.batch_size
    if args.energy_threshold is not None:
        run.model.energy_threshold = args.energy_threshold
    if args.device is not None:
        run.model.device = args.device

    if args.confirmed:
        run.confirmed = True
    return run


def main() -> None:
    args = parse_args()

    if args.write_config:
        save_run_config_json(default_run_config(), args.write_config)
        print(f"Wrote config template: {args.write_config}")
        return

    if not args.config and args.smiles is None and args.query_csv is None:
        raise SystemExit("Provide --config <run_config.json> (or --write-config to scaffold one).")

    run = build_run_config(args)
    try:
        run.io.validate()
        run.model.validate()
    except ValueError as exc:
        raise SystemExit(f"Invalid config: {exc}")

    if not run.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Run blocked: confirmed is false.\n"
            "Present the full run config to the user for approval, then set "
            '"confirmed": true (or pass --confirmed) before executing. See SKILL.md.'
        )

    io, model_cfg = run.io, run.model

    # Build queries.
    query_rows = build_query_rows(io)
    n_total = len(query_rows)
    query_rows = [r for r in query_rows if r.get("smiles")]
    n_with_smiles = len(query_rows)
    if n_with_smiles == 0:
        raise SystemExit(
            "No query molecules have a SMILES. This skill predicts Eb from SMILES; "
            "provide a 'smiles' for each inline query or a SMILES column in the CSV."
        )
    print(f"Queries: {n_with_smiles}/{n_total} with SMILES "
          f"({'inline' if io.queries else 'csv'} source).")

    # Import the heavy backend module only now (after the gate passes).
    project_root = find_project_root(io.project_root)
    BindingEnergyPredictor = import_predictor(project_root)

    device = None if model_cfg.device == "auto" else model_cfg.device

    print("Loading binding-energy model...")
    predictor = BindingEnergyPredictor(
        checkpoint_path=model_cfg.downstream_checkpoint,
        gin_e_checkpoint_path=model_cfg.gin_e_checkpoint,
        device=device,
    )

    smiles_list = [r["smiles"] for r in query_rows]
    print(f"Predicting Eb for {len(smiles_list)} molecule(s) "
          f"(batch_size={model_cfg.batch_size})...")

    out_rows: List[Dict[str, str]] = []
    n_ok = 0
    for start in range(0, len(smiles_list), model_cfg.batch_size):
        batch = smiles_list[start:start + model_cfg.batch_size]
        results = predictor.predict_batch(batch)
        for offset, (energy, status) in enumerate(results):
            qrow = query_rows[start + offset]
            row: Dict[str, Any] = {
                "name": qrow.get("name", ""),
                "cid": qrow.get("cid", ""),
                "smiles": qrow.get("smiles", ""),
                "predicted_binding_energy_eV": "" if energy is None else f"{energy:.6f}",
                "prediction_status": status,
            }
            if model_cfg.energy_threshold is not None:
                row["below_threshold"] = (
                    "" if energy is None else str(energy <= model_cfg.energy_threshold)
                )
            if energy is not None:
                n_ok += 1
            out_rows.append(row)
        print(f"  {min(start + model_cfg.batch_size, len(smiles_list))}/{len(smiles_list)} done.")

    if io.sort_by_energy:
        def _sort_key(r: Dict[str, str]):
            val = r.get("predicted_binding_energy_eV", "")
            # Valid energies first (ascending: most negative = strongest binder),
            # failed predictions (empty) pushed to the end.
            return (0, float(val)) if val != "" else (1, 0.0)
        out_rows.sort(key=_sort_key)

    write_predictions_csv(io.output_csv, out_rows, model_cfg.energy_threshold)
    print(f"\nWrote {len(out_rows)} rows ({n_ok} successful) -> {io.output_csv}")
    if model_cfg.energy_threshold is not None:
        n_below = sum(1 for r in out_rows if r.get("below_threshold") == "True")
        print(f"  Strong binders (Eb <= {model_cfg.energy_threshold} eV): {n_below}")
    print("  Output has 'cid'/'smiles' columns; feed it to the mol-salt-vendor skill.")
    print("\nDone.")


if __name__ == "__main__":
    main()
