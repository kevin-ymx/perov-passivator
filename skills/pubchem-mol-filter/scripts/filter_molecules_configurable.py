"""
Configurable batch filter for PubChem CSV data (single file or shard directory).

A single JSON run config fully specifies a run: the "io" block holds input/output
paths, mode, and workers; the "filters" block holds the RDKit criteria. This lets
any agent platform (Claude Code, Codex, Cursor, ...) execute the skill by editing
one config file. CLI flags are optional overrides on top of the config.

Usage:
    # 1. Write a full run-config template (io + filters), then edit it:
    python filter_molecules_configurable.py --write-config run_config.json

    # 2. Run entirely from the config (no other args needed):
    python filter_molecules_configurable.py --config run_config.json

    # Optional CLI overrides:
    python filter_molecules_configurable.py --config run_config.json \\
        --input_dir /path/to/shards --output_dir /path/to/out --workers 64
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molecule_filter import (
    FilterConfig,
    IOConfig,
    RunConfig,
    check_filters,
    default_run_config,
    load_run_config_json,
    save_run_config_json,
)

_WORKER_CONFIG: Optional[FilterConfig] = None


def _init_worker(config: FilterConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _filter_one(row: Tuple[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cid, smiles = row
    if not smiles:
        return None, None, "no_smiles"
    mol = Chem.MolFromSmiles(smiles)
    reason = check_filters(mol, _WORKER_CONFIG)
    if reason is not None:
        return None, None, reason
    return cid, smiles, None


def discover_csv_shards(input_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(input_dir, "*.csv")))


def load_csv_rows(filepath: str) -> List[Tuple[str, str]]:
    rows = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("PUBCHEM_COMPOUND_CID") or row.get("cid") or "").strip()
            smiles = (row.get("SMILES") or row.get("smiles") or "").strip()
            rows.append((cid, smiles))
    return rows


def save_csv(data: List[Tuple[str, str]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PUBCHEM_COMPOUND_CID", "SMILES"])
        for cid, smiles in data:
            writer.writerow([cid, smiles])


def print_rejection_counts(rej_counts: Dict[str, int], config: FilterConfig, indent: int = 2) -> None:
    prefix = " " * indent
    for reason in config.all_rejection_keys():
        cnt = rej_counts.get(reason, 0)
        if cnt > 0:
            print(f"{prefix}{reason}: {cnt:,}")


def filter_rows(
    rows: List[Tuple[str, str]],
    config: FilterConfig,
    workers: int,
    label: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    valid: List[Tuple[str, str]] = []
    rej_counts: Dict[str, int] = {}
    n_input = len(rows)

    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
            results_iter = pool.imap(_filter_one, rows, chunksize=500)
            for cid, smiles, reason in tqdm(results_iter, total=n_input, desc=label):
                if reason is None:
                    valid.append((cid, smiles))
                else:
                    rej_counts[reason] = rej_counts.get(reason, 0) + 1
    else:
        global _WORKER_CONFIG
        _WORKER_CONFIG = config
        for row in tqdm(rows, desc=label):
            cid, smiles, reason = _filter_one(row)
            if reason is None:
                valid.append((cid, smiles))
            else:
                rej_counts[reason] = rej_counts.get(reason, 0) + 1

    return valid, rej_counts


def process_file(
    input_path: str,
    output_path: str,
    config: FilterConfig,
    workers: int,
) -> Tuple[int, int, Dict[str, int]]:
    label = os.path.basename(input_path)
    print(f"\n{'=' * 60}\nProcessing: {label}\n{'=' * 60}")
    rows = load_csv_rows(input_path)
    n_input = len(rows)
    if n_input == 0:
        print("  Empty file, skipping.")
        return 0, 0, {}

    print(f"  Loaded {n_input:,} rows. Active filters: {', '.join(config.active_filter_names())}")
    valid, rej_counts = filter_rows(rows, config, workers, f"Filtering {label}")
    n_passed = len(valid)
    print(f"  Passed: {n_passed:,} / {n_input:,} ({100.0 * n_passed / max(n_input, 1):.2f}%)")
    print_rejection_counts(rej_counts, config, indent=4)
    save_csv(valid, output_path)
    print(f"  Saved to {output_path}")
    return n_input, n_passed, rej_counts


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value or not value.strip():
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def build_run_config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.config:
        run = load_run_config_json(args.config)
    else:
        run = default_run_config()

    apply_io_overrides(run.io, args)
    apply_filter_overrides(run.filters, args)
    if getattr(args, "confirmed", False):
        run.confirmed = True
    return run


def apply_io_overrides(io: IOConfig, args: argparse.Namespace) -> None:
    if args.mode is not None:
        io.mode = args.mode
    if args.input is not None:
        io.input = args.input
        if args.mode is None:
            io.mode = "single"
    if args.output is not None:
        io.output = args.output
    if args.input_dir is not None:
        io.input_dir = args.input_dir
        if args.mode is None:
            io.mode = "shards"
    if args.output_dir is not None:
        io.output_dir = args.output_dir
    if args.workers is not None:
        io.workers = args.workers
    if args.save_config_used is not None:
        io.save_config_used = args.save_config_used


def apply_filter_overrides(cfg: FilterConfig, args: argparse.Namespace) -> None:
    if args.no_sanitization:
        cfg.require_sanitization = False
    if args.no_single_component:
        cfg.require_single_component = False
    if args.allowed_elements is not None:
        cfg.allowed_elements = parse_csv_list(args.allowed_elements)
    if args.require_any_elements is not None:
        cfg.require_any_elements = parse_csv_list(args.require_any_elements)
    if args.require_all_elements is not None:
        cfg.require_all_elements = parse_csv_list(args.require_all_elements)
    if args.forbidden_elements is not None:
        cfg.forbidden_elements = parse_csv_list(args.forbidden_elements)
    if args.max_heavy_atoms is not None:
        cfg.max_heavy_atoms = args.max_heavy_atoms
    if args.no_max_heavy_atoms:
        cfg.max_heavy_atoms = None
    if args.max_ring_size is not None:
        cfg.max_ring_size = args.max_ring_size
    if args.no_max_ring_size:
        cfg.max_ring_size = None
    if args.max_mol_weight is not None:
        cfg.max_mol_weight = args.max_mol_weight
    if args.no_max_mol_weight:
        cfg.max_mol_weight = None
    if args.allow_zwitterion:
        cfg.reject_zwitterion = False
    if args.allow_isotopes:
        cfg.reject_isotopes = False
    if args.no_heteroatom_lone_pair:
        cfg.require_heteroatom_lone_pair = False
    if args.no_require_any_elements:
        cfg.require_any_elements = []
    if args.allow_radicals:
        cfg.reject_radicals = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter PubChem CSV(s) with configurable RDKit criteria."
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON run config with 'confirmed', 'io', and 'filters'. "
                        "Execution blocked until confirmed: true (see SKILL.md).")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write a full run-config JSON template (confirmed: false) and exit.")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark run as user-confirmed (set confirmed: true). "
                        "Use only after every config field was approved.")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass confirmed check (not for agent use).")

    io = p.add_argument_group("I/O overrides (merge on top of --config 'io')")
    io.add_argument("--mode", type=str, default=None, choices=["single", "shards"],
                    help="single (one CSV) or shards (directory of *.csv).")
    io.add_argument("--input", type=str, default=None, help="Single input CSV file (mode=single).")
    io.add_argument("--output", type=str, default=None, help="Single output CSV file (mode=single).")
    io.add_argument("--input_dir", type=str, default=None, help="Directory of CSV shards (mode=shards).")
    io.add_argument("--output_dir", type=str, default=None, help="Output directory (mode=shards).")
    io.add_argument("--workers", type=int, default=None, help="Parallel workers (default: 64).")
    io.add_argument("--save-config-used", type=str, default=None,
                    help="Path to archive the resolved run config before filtering.")

    g = p.add_argument_group("Filter overrides (merge on top of --config)")
    g.add_argument("--no-sanitization", action="store_true")
    g.add_argument("--no-single-component", action="store_true")
    g.add_argument("--allowed-elements", type=str, default=None,
                   help="Comma-separated symbols, e.g. H,C,N,O,S,P,F,Cl,Br,I")
    g.add_argument("--require-any-elements", type=str, default=None,
                   help="At least one required, e.g. N,O,S,P")
    g.add_argument("--require-all-elements", type=str, default=None)
    g.add_argument("--forbidden-elements", type=str, default=None)
    g.add_argument("--max-heavy-atoms", type=int, default=None)
    g.add_argument("--no-max-heavy-atoms", action="store_true")
    g.add_argument("--allow-radicals", action="store_true",
                   help="Do not reject molecules with radical electrons.")
    g.add_argument("--max-ring-size", type=int, default=None)
    g.add_argument("--no-max-ring-size", action="store_true")
    g.add_argument("--max-mol-weight", type=float, default=None)
    g.add_argument("--no-max-mol-weight", action="store_true")
    g.add_argument("--allow-zwitterion", action="store_true")
    g.add_argument("--allow-isotopes", action="store_true")
    g.add_argument("--no-heteroatom-lone-pair", action="store_true")
    g.add_argument("--no-require-any-elements", action="store_true")
    return p.parse_args()


def run_single(io: IOConfig, config: FilterConfig) -> None:
    n_in, n_pass, _ = process_file(io.input, io.output, config, io.workers)
    print(f"\nDone. {n_pass:,} / {n_in:,} passed. Output: {io.output}")


def run_shards(io: IOConfig, config: FilterConfig) -> None:
    paths = discover_csv_shards(io.input_dir)
    os.makedirs(io.output_dir, exist_ok=True)
    print(f"Shards: {len(paths)} in {io.input_dir}")

    total_in = total_pass = 0
    total_rej: Dict[str, int] = {}
    for idx, csv_path in enumerate(paths, 1):
        fname = os.path.basename(csv_path)
        out_path = os.path.join(io.output_dir, fname)
        if os.path.isfile(out_path):
            print(f"[{idx}/{len(paths)}] Skip {fname} (output exists)")
            continue
        print(f"[{idx}/{len(paths)}] {fname}")
        n_in, n_pass, rej = process_file(csv_path, out_path, config, io.workers)
        total_in += n_in
        total_pass += n_pass
        for k, v in rej.items():
            total_rej[k] = total_rej.get(k, 0) + v

    print(f"\n{'=' * 60}\nBATCH COMPLETE\n{'=' * 60}")
    print(f"Total passed: {total_pass:,} / {total_in:,}")
    print_rejection_counts(total_rej, config)
    print(f"Output: {io.output_dir}")


def main() -> None:
    args = parse_args()

    if args.write_config:
        save_run_config_json(default_run_config(), args.write_config)
        print(f"Wrote run-config template: {args.write_config}")
        return

    run = build_run_config_from_args(args)
    try:
        run.io.validate()
    except ValueError as exc:
        raise SystemExit(
            f"Invalid I/O config: {exc}\n"
            "Provide confirmed paths in the --config JSON 'io' block or via CLI overrides."
        )

    if not run.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Run blocked: confirmed is false.\n"
            "Present the full run config to the user for approval, then set "
            '"confirmed": true in the JSON (or pass --confirmed) before executing.\n'
            "See SKILL.md."
        )

    io, config = run.io, run.filters

    if io.save_config_used:
        save_run_config_json(run, io.save_config_used)
        print(f"Saved resolved run config: {io.save_config_used}")

    print(f"Mode: {io.mode} | Workers: {io.workers}")
    print("Active filters:", ", ".join(config.active_filter_names()))

    if io.mode == "single":
        run_single(io, config)
    else:
        run_shards(io, config)


if __name__ == "__main__":
    main()
