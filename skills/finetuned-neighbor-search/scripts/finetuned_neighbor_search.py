"""
Config-driven finetuned GIN-E nearest-neighbor search for user molecules.

This runner mirrors the ssl-neighbor-search skill, but computes query embeddings
with a finetuned GIN-E encoder checkpoint and searches a reference embedding set
generated in that same finetuned embedding space.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PLACEHOLDER_MARKERS = ("/REPLACE/", "/ABSOLUTE/PATH", "REPLACE_ME")


@dataclass
class IOConfig:
    queries: List[Dict[str, Any]] = field(default_factory=list)
    query_csv: Optional[str] = None
    name_column: str = "molecule_name"
    cid_column: str = "cid"
    smiles_column: str = "smiles"
    embedding_dir: Optional[str] = None
    project_root: Optional[str] = None
    output_csv: str = "finetuned_neighbors.csv"
    output_dedup_csv: str = "finetuned_neighbors_dedup.csv"
    write_dedup: bool = True

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
        if not self.embedding_dir:
            raise ValueError("io.embedding_dir is required.")
        if _has_placeholder(self.embedding_dir):
            raise ValueError(f"io.embedding_dir still contains a placeholder: {self.embedding_dir}")
        if not os.path.isdir(self.embedding_dir):
            raise ValueError(f"io.embedding_dir not found: {self.embedding_dir}")


@dataclass
class ModelConfig:
    gin_e_checkpoint: Optional[str] = None
    device: str = "auto"

    def validate(self) -> None:
        if not self.gin_e_checkpoint:
            raise ValueError("model.gin_e_checkpoint is required.")
        if _has_placeholder(self.gin_e_checkpoint):
            raise ValueError(
                f"model.gin_e_checkpoint still contains a placeholder: {self.gin_e_checkpoint}"
            )
        if not os.path.isfile(self.gin_e_checkpoint):
            raise ValueError(f"model.gin_e_checkpoint not found: {self.gin_e_checkpoint}")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("model.device must be one of: auto, cuda, cpu.")


@dataclass
class KNNConfig:
    k: int = 21
    chunk_size: int = 100000
    drop_query_isotopes: bool = True

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("knn.k must be >= 1.")
        if self.chunk_size < 1:
            raise ValueError("knn.chunk_size must be >= 1.")


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)


def default_run_config() -> RunConfig:
    return RunConfig(
        confirmed=False,
        io=IOConfig(
            queries=[{"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": None}],
            query_csv=None,
            embedding_dir="/REPLACE/with/filtered_csv_latest_embeddings_finetuned",
        ),
        model=ModelConfig(gin_e_checkpoint="/REPLACE/with/gin_e_finetuned.pt"),
        knn=KNNConfig(),
    )


def _has_placeholder(value: Optional[str]) -> bool:
    return bool(value) and any(marker in value for marker in PLACEHOLDER_MARKERS)


def load_run_config_json(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    io = IOConfig(**{**asdict(IOConfig()), **data.get("io", {})})
    model = ModelConfig(**{**asdict(ModelConfig()), **data.get("model", {})})
    knn = KNNConfig(**{**asdict(KNNConfig()), **data.get("knn", {})})
    return RunConfig(confirmed=bool(data.get("confirmed", False)), io=io, model=model, knn=knn)


def save_run_config_json(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)


def _norm_row(name: str, cid: Any, smiles: str) -> Dict[str, str]:
    return {
        "molecule_name": (name or "").strip(),
        "cid": ("" if cid is None else str(cid)).strip(),
        "smiles": (smiles or "").strip(),
        "journal": "",
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


LONG_FORM_COLUMNS = [
    "query_name", "query_cid", "query_smiles",
    "rank", "ref_cid", "ref_smiles", "ref_status", "distance",
]


def write_long_form_csv(
    output_path: str,
    valid_rows: List[Dict[str, str]],
    knn_results: List[List[Tuple[float, str, str, str]]],
) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_FORM_COLUMNS)
        writer.writeheader()
        for row, neighbors in zip(valid_rows, knn_results):
            for rank, (dist, ref_cid, ref_smiles, ref_status) in enumerate(neighbors, start=1):
                writer.writerow({
                    "query_name": row.get("molecule_name", ""),
                    "query_cid": row.get("cid", ""),
                    "query_smiles": row.get("smiles", "") or row.get("SMILES", ""),
                    "rank": rank,
                    "ref_cid": ref_cid,
                    "ref_smiles": ref_smiles,
                    "ref_status": ref_status,
                    "distance": f"{dist:.6f}",
                })


DEDUP_COLUMNS = [
    "cid", "smiles", "ref_status",
    "matched_query_molecules", "n_query_matches", "best_rank", "best_distance",
]


def write_dedup_csv(
    output_path: str,
    valid_rows: List[Dict[str, str]],
    knn_results: List[List[Tuple[float, str, str, str]]],
) -> int:
    agg: Dict[str, Dict[str, Any]] = {}
    for qrow, neighbors in zip(valid_rows, knn_results):
        qlabel = (
            qrow.get("molecule_name")
            or qrow.get("cid")
            or qrow.get("smiles")
            or ""
        ).strip()
        for rank, (dist, ref_cid, ref_smiles, ref_status) in enumerate(neighbors, start=1):
            key = (ref_cid or ref_smiles or "").strip()
            if not key:
                continue
            if key not in agg:
                agg[key] = {
                    "cid": ref_cid,
                    "smiles": ref_smiles,
                    "ref_status": ref_status,
                    "queries": [],
                    "best_rank": rank,
                    "best_distance": dist,
                }
            entry = agg[key]
            if qlabel and qlabel not in entry["queries"]:
                entry["queries"].append(qlabel)
            if rank < entry["best_rank"]:
                entry["best_rank"] = rank
            if dist < entry["best_distance"]:
                entry["best_distance"] = dist

    keys = sorted(agg, key=lambda key: agg[key]["best_distance"])
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEDUP_COLUMNS)
        writer.writeheader()
        for key in keys:
            entry = agg[key]
            writer.writerow({
                "cid": entry["cid"],
                "smiles": entry["smiles"],
                "ref_status": entry["ref_status"],
                "matched_query_molecules": "; ".join(entry["queries"]),
                "n_query_matches": len(entry["queries"]),
                "best_rank": entry["best_rank"],
                "best_distance": f"{entry['best_distance']:.6f}",
            })
    return len(keys)


def find_project_root(explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f"io.project_root not found: {explicit}")
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "knn_finetunedembedding_search.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit(
        "Could not locate the backend root (knn_finetunedembedding_search.py). "
        "Set io.project_root in the config."
    )


def import_knn_module(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import knn_finetunedembedding_search as knn  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"Failed to import knn_finetunedembedding_search from {project_root}: {exc}\n"
            "Ensure the backend env (torch, torch-geometric, rdkit) is installed."
        )
    return knn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finetuned GIN-E nearest-neighbor search for user-specified molecules."
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON run config. Blocked until confirmed: true.")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write a config template and exit.")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark run as user-confirmed after the config was approved.")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass the confirmed gate (not for agent use).")
    p.add_argument("--smiles", type=str, default=None,
                   help="Comma-separated SMILES; overrides queries with inline entries.")
    p.add_argument("--query-csv", type=str, default=None)
    p.add_argument("--embedding-dir", type=str, default=None)
    p.add_argument("--gin-e-checkpoint", type=str, default=None)
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--output-dedup-csv", type=str, default=None)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
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
    if args.embedding_dir is not None:
        run.io.embedding_dir = args.embedding_dir
    if args.project_root is not None:
        run.io.project_root = args.project_root
    if args.output_csv is not None:
        run.io.output_csv = args.output_csv
    if args.output_dedup_csv is not None:
        run.io.output_dedup_csv = args.output_dedup_csv
    if args.no_dedup:
        run.io.write_dedup = False
    if args.gin_e_checkpoint is not None:
        run.model.gin_e_checkpoint = args.gin_e_checkpoint
    if args.device is not None:
        run.model.device = args.device
    if args.k is not None:
        run.knn.k = args.k
    if args.chunk_size is not None:
        run.knn.chunk_size = args.chunk_size
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
        run.knn.validate()
    except ValueError as exc:
        raise SystemExit(f"Invalid config: {exc}")

    if not run.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Run blocked: confirmed is false.\n"
            "Present the full run config to the user for approval, then set "
            '"confirmed": true (or pass --confirmed) before executing. See SKILL.md.'
        )

    io, model_cfg, knn_cfg = run.io, run.model, run.knn
    query_rows = build_query_rows(io)
    n_total = len(query_rows)
    query_rows = [r for r in query_rows if r.get("smiles")]
    n_with_smiles = len(query_rows)
    if n_with_smiles == 0:
        raise SystemExit(
            "No query molecules have a SMILES. This skill embeds molecules by SMILES; "
            "provide a 'smiles' for each inline query or a SMILES column in the CSV."
        )
    print(f"Queries: {n_with_smiles}/{n_total} with SMILES "
          f"({'inline' if io.queries else 'csv'} source).")

    project_root = find_project_root(io.project_root)
    knn = import_knn_module(project_root)

    if knn_cfg.drop_query_isotopes:
        before = len(query_rows)
        query_rows = [r for r in query_rows if not knn.has_isotope(r["smiles"])]
        dropped = before - len(query_rows)
        if dropped:
            print(f"  Dropped {dropped} query molecule(s) with isotope-labeled SMILES.")
        if not query_rows:
            raise SystemExit("All query molecules were dropped (isotopes). Nothing to do.")

    config = knn.Config()
    device_name = model_cfg.device if model_cfg.device != "auto" else (
        config.device if knn.torch.cuda.is_available() else "cpu"
    )
    device = knn.torch.device(device_name)
    print(f"Using device: {device}")

    print("Loading finetuned GIN-E encoder...")
    encoder = knn.load_finetuned_encoder(model_cfg.gin_e_checkpoint, config, device)

    print("Computing query embeddings with finetuned GIN-E...")
    query_embeddings, valid_rows, _ = knn.get_query_embeddings(query_rows, encoder, device)
    print(f"  Got embeddings for {len(valid_rows)} / {len(query_rows)} molecules.")
    if len(valid_rows) == 0:
        raise SystemExit("No valid query embeddings (check SMILES / checkpoint).")

    print(f"Running k-NN (k={knn_cfg.k}) over reference embeddings in {io.embedding_dir} ...")
    knn_results = knn.run_knn(
        query_embeddings,
        io.embedding_dir,
        k=knn_cfg.k,
        chunk_size=knn_cfg.chunk_size,
    )

    print("Writing results...")
    write_long_form_csv(io.output_csv, valid_rows, knn_results)
    print(f"  Long-form neighbors: {io.output_csv}")

    if io.write_dedup:
        n_unique = write_dedup_csv(io.output_dedup_csv, valid_rows, knn_results)
        print(f"  Deduplicated neighbors ({n_unique} unique): {io.output_dedup_csv}")
        print("  -> dedup CSV has 'cid'/'smiles' columns for downstream lookup.")

    print("\nDone.")


if __name__ == "__main__":
    main()
