"""
pubchem-mol-availability runner.

For each neutral input molecule (cid, smiles) it:
  1. finds same-parent-connectivity related CIDs (PubChem),
  2. classifies which related CIDs are halide salts (RDKit),
  3. enriches the neutral molecule and each halide salt with physical form,
     melting point and vendor count (PubChem), resolving form via the cascade
     PubChem text -> melting-point heuristic -> LLM,
  4. emits one row per halide salt (plus a neutral-only row when no halide salt
     exists), carrying both salt and parent data,
  5. drops rows whose decision form is confirmed 'liquid' (kept rows -> output,
     dropped rows -> dropped_output).

A single JSON run config fully specifies a run. Execution is blocked until
`confirmed: true`. The whole pipeline runs in one process (one Slurm job) and is
resumable via the on-disk caches under io.cache_dir.

Usage:
    python run_availability.py --write-config run_config.json
    python run_availability.py --config run_config.json
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from availability_core import (
    RunConfig,
    classify_salt,
    default_run_config,
    load_run_config_json,
    needs_llm,
    parse_melting_point_c,
    resolve_form,
    run_config_to_dict,
    save_run_config_json,
)
from pubchem_client import JsonCache, LLMClient, PubChemClient

OUTPUT_COLUMNS = [
    "input_cid",
    "input_smiles",
    "has_halide_salt",
    "salt_cid",
    "salt_smiles",
    "salt_counterion",
    "salt_physical_form",
    "salt_n_vendors",
    "salt_purchasable",
    "salt_vendor_examples",
    "parent_physical_form",
    "parent_n_vendors",
    "parent_purchasable",
    "form_source",
    "form_confidence",
    "kept",
]


_CID_AUTO_COLUMNS = ("PUBCHEM_COMPOUND_CID", "cid", "CID")
_SMILES_AUTO_COLUMNS = ("SMILES", "smiles")


def _resolve_column(
    fieldnames: List[str], configured: Optional[str], auto_names: Tuple[str, ...], kind: str
) -> Optional[str]:
    """Pick the column name to read. Explicit config wins; else auto-detect."""
    if configured:
        if configured not in (fieldnames or []):
            raise SystemExit(
                f"io.{kind}_column = {configured!r} not found in input CSV header "
                f"{list(fieldnames or [])}. Set it to an existing column or null to auto-detect."
            )
        return configured
    for name in auto_names:
        if name in (fieldnames or []):
            return name
    return None


def read_input_rows(
    path: str, cid_column: Optional[str] = None, smiles_column: Optional[str] = None
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        cid_col = _resolve_column(fieldnames, cid_column, _CID_AUTO_COLUMNS, "cid")
        smiles_col = _resolve_column(fieldnames, smiles_column, _SMILES_AUTO_COLUMNS, "smiles")
        if cid_col is None and smiles_col is None:
            raise SystemExit(
                "Could not find CID or SMILES columns in input CSV "
                f"(header {fieldnames}). Set io.cid_column / io.smiles_column."
            )
        for row in reader:
            cid = (row.get(cid_col, "") if cid_col else "").strip()
            smiles = (row.get(smiles_col, "") if smiles_col else "").strip()
            if cid or smiles:
                rows.append((cid, smiles))
    return rows


class Enricher:
    """Resolve physical form + vendor info for a CID, using caches + cascade."""

    def __init__(self, run: RunConfig, pc: PubChemClient, llm: LLMClient):
        self.run = run
        self.pc = pc
        self.llm = llm

    def form_and_vendors(self, cid: int, smiles: str):
        desc = self.pc.physical_description(cid)
        mp_text = self.pc.melting_point_text(cid)
        mp_c = parse_melting_point_c(mp_text)

        llm_form = llm_conf = None
        if self.run.llm.enabled and needs_llm(desc, mp_c):
            llm_form, llm_conf = self.llm.classify(cid, smiles)

        verdict = resolve_form(
            description_text=desc,
            melting_point_c=mp_c,
            mp_solid_threshold_c=self.run.lookup.mp_solid_threshold_c,
            llm_form=llm_form,
            llm_confidence=llm_conf,
            llm_confidence_threshold=self.run.llm.confidence_threshold,
        )
        n_vendors, examples = self.pc.vendor_info(cid)
        return verdict, n_vendors, examples


def build_rows_for_molecule(
    input_cid: str,
    input_smiles: str,
    run: RunConfig,
    pc: PubChemClient,
    enricher: Enricher,
) -> List[Dict]:
    """Produce output row dict(s) for a single input molecule."""
    try:
        cid_int = int(input_cid)
    except (TypeError, ValueError):
        cid_int = None

    # Parent (neutral input) enrichment.
    parent_form = parent_vendors = parent_examples = None
    if cid_int is not None and run.lookup.report_neutral:
        pv, pn, pex = enricher.form_and_vendors(cid_int, input_smiles)
        parent_form, parent_vendors, parent_examples = pv, pn, pex

    # Discover halide-salt forms among same-parent CIDs.
    halide_salts: List[Tuple[int, str, str]] = []  # (cid, smiles, counterion)
    if cid_int is not None:
        related = pc.same_parent_cids(cid_int)
        if related:
            smap = pc.smiles_for_cids(related)
            for rcid in related:
                rsmiles = smap.get(rcid, "")
                info = classify_salt(rsmiles)
                use = info.is_halide_salt if run.lookup.halide_only else info.is_salt
                if use:
                    counterion = ",".join(
                        info.halide_counterions if run.lookup.halide_only else info.counterions
                    )
                    halide_salts.append((rcid, rsmiles, counterion))

    base = {
        "input_cid": input_cid,
        "input_smiles": input_smiles,
        "has_halide_salt": bool(halide_salts),
        "parent_physical_form": parent_form.form if parent_form else "unknown",
        "parent_n_vendors": parent_vendors if parent_vendors is not None else "",
        "parent_purchasable": (parent_vendors or 0) > 0 if parent_vendors is not None else "",
    }

    rows: List[Dict] = []
    if halide_salts:
        for scid, ssmiles, counterion in halide_salts:
            sv, sn, sex = enricher.form_and_vendors(scid, ssmiles)
            row = dict(base)
            row.update(
                {
                    "salt_cid": scid,
                    "salt_smiles": ssmiles,
                    "salt_counterion": counterion,
                    "salt_physical_form": sv.form,
                    "salt_n_vendors": sn,
                    "salt_purchasable": (sn or 0) > 0,
                    "salt_vendor_examples": "; ".join(sex),
                    "form_source": sv.source,
                    "form_confidence": "" if sv.confidence is None else round(sv.confidence, 3),
                    "_decision_form": sv.form,
                }
            )
            rows.append(row)
    else:
        row = dict(base)
        row.update(
            {
                "salt_cid": "",
                "salt_smiles": "",
                "salt_counterion": "",
                "salt_physical_form": "",
                "salt_n_vendors": "",
                "salt_purchasable": "",
                "salt_vendor_examples": "",
                "form_source": parent_form.source if parent_form else "none",
                "form_confidence": ""
                if not parent_form or parent_form.confidence is None
                else round(parent_form.confidence, 3),
                "_decision_form": parent_form.form if parent_form else "unknown",
            }
        )
        rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})


def run_pipeline(run: RunConfig) -> None:
    io, lookup, llm_cfg = run.io, run.lookup, run.llm
    cache_dir = io.cache_dir or os.path.join(os.path.dirname(io.output), "cache")

    pc_cache = JsonCache(os.path.join(cache_dir, "pubchem_cache.json"))
    llm_cache = JsonCache(os.path.join(cache_dir, "llm_cache.json"))
    pc = PubChemClient(
        pc_cache,
        request_rate_per_sec=lookup.request_rate_per_sec,
        max_retries=lookup.max_retries,
        timeout_sec=lookup.timeout_sec,
    )
    llm = LLMClient(llm_cfg, llm_cache)
    if llm_cfg.enabled and not llm.available:
        print(f"WARNING: LLM fallback disabled ({llm.unavailable_reason}); "
              "molecules with no PubChem form and no melting point stay 'unknown'.")
    enricher = Enricher(run, pc, llm)

    inputs = read_input_rows(io.input, io.cid_column, io.smiles_column)
    print(f"Loaded {len(inputs):,} input molecules from {io.input}")

    kept_rows: List[Dict] = []
    dropped_rows: List[Dict] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(inputs, desc="Molecules")
    except Exception:
        iterator = inputs

    for cid, smiles in iterator:
        rows = build_rows_for_molecule(cid, smiles, run, pc, enricher)
        for row in rows:
            decision_form = row.pop("_decision_form", "unknown")
            is_liquid = decision_form == "liquid"
            keep = not (run.filter.drop_liquids and is_liquid)
            row["kept"] = keep
            (kept_rows if keep else dropped_rows).append(row)

    pc_cache.flush()
    llm_cache.flush()

    write_csv(io.output, kept_rows)
    print(f"Wrote {len(kept_rows):,} kept rows -> {io.output}")
    if io.dropped_output:
        write_csv(io.dropped_output, dropped_rows)
        print(f"Wrote {len(dropped_rows):,} dropped (liquid) rows -> {io.dropped_output}")

    total = len(kept_rows) + len(dropped_rows)
    print(f"\nSummary: {total:,} rows | kept {len(kept_rows):,} | "
          f"dropped(liquid) {len(dropped_rows):,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PubChem salt-form / form / availability lookup.")
    p.add_argument("--config", type=str, default=None,
                   help="JSON run config. Execution blocked until confirmed: true.")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write a run-config template (confirmed: false) and exit.")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark run as user-confirmed. Use only after approval.")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass the confirmed gate (not for agent use).")
    p.add_argument("--input", type=str, default=None, help="Override io.input.")
    p.add_argument("--output", type=str, default=None, help="Override io.output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_config:
        save_run_config_json(default_run_config(), args.write_config)
        print(f"Wrote run-config template: {args.write_config}")
        return

    if not args.config:
        raise SystemExit("--config is required (or use --write-config).")

    run = load_run_config_json(args.config)
    if args.input is not None:
        run.io.input = args.input
    if args.output is not None:
        run.io.output = args.output
    if args.confirmed:
        run.confirmed = True

    try:
        run.validate()
    except ValueError as exc:
        raise SystemExit(f"Invalid run config: {exc}")

    if not run.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Run blocked: confirmed is false.\n"
            "Present the full run config to the user for approval, then set "
            '"confirmed": true in the JSON (or pass --confirmed). See SKILL.md.'
        )

    run_pipeline(run)


if __name__ == "__main__":
    main()
