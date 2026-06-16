"""
k-NN neighbor search for strong-binding and weak-binding molecules.

1. Read downstream_csv (from config.py), select rows with:
   - adsorption_energy < -1.7 eV  (strong binders)
   - adsorption_energy in [-0.5, 0] eV  (weak binders)
2. For each selected query molecule, look up its pre-computed embedding from the
   reference CSVs in DEFAULT_EMBEDDING_DIR by matching CID or SMILES.
3. Find k nearest neighbors (excluding the query itself) from the same reference
   embeddings using L2 distance (k=15 for strong binders, k=5 for weak binders).
4. Write results to a CSV file.

Usage:
    python knn_strong_binders.py
    python knn_strong_bindggGers.py --threshold -1.7 --k 15 --output AL_knn_results.csv
"""
import argparse
import csv
import glob
import heapq
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config

# Paths from knn_embedding_search.py
DEFAULT_EMBEDDING_DIR = "/kfs3/scratch/yeming/ai4m/prediction/filtered_csv_embeddings"
EMBEDDING_DIM = 256
CHUNK_SIZE = 100_000


# ---------------------------------------------------------------------------
# Step 1: Load downstream CSV and filter by binding energy
# ---------------------------------------------------------------------------

def load_selected_molecules(
    csv_path: str,
    threshold: float,
    weak_low: float = -0.5,
    weak_high: float = 0.0,
) -> List[dict]:
    """Load rows from downstream CSV with:
    - adsorption_energy < threshold  (strong binders)
    - weak_low <= adsorption_energy <= weak_high  (weak binders)

    Each returned row gets a 'query_category' field.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                energy = float(row["adsorption_energy"])
            except (ValueError, KeyError, TypeError):
                continue
            if energy < threshold:
                row["adsorption_energy"] = energy
                row["query_category"] = "strong_binder"
                rows.append(row)
            elif weak_low <= energy <= weak_high:
                row["adsorption_energy"] = energy
                row["query_category"] = "weak_binder"
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 2: Look up query embeddings from reference CSVs by CID / SMILES
# ---------------------------------------------------------------------------

def list_embedding_csvs(embedding_dir: str) -> List[str]:
    """List and sort embedding CSV files by leading number."""
    pattern = os.path.join(embedding_dir, "*.csv")
    files = glob.glob(pattern)

    def sort_key(p: str) -> int:
        base = os.path.basename(p)
        try:
            return int(base.split("_")[0])
        except ValueError:
            return 0

    return sorted(files, key=sort_key)


def get_emb_columns(fieldnames: List[str]) -> List[str]:
    """Return emb_0 .. emb_255 column names, sorted by index."""
    emb = [c for c in fieldnames if c.startswith("emb_") and len(c) > 4 and c[4:].isdigit()]
    return sorted(emb, key=lambda x: int(x.split("_")[1]))


def lookup_query_embeddings(
    query_rows: List[dict],
    embedding_dir: str,
) -> Tuple[np.ndarray, List[dict], List[int]]:
    """
    Scan all reference CSVs and look up embeddings for query molecules by CID
    (primary) or SMILES (fallback).

    Returns:
        embeddings: (n_matched, 256) float32 array
        matched_rows: list of downstream row dicts that were matched
        matched_indices: original indices into query_rows
    """
    # Build lookup sets for fast matching
    cid_to_idx: Dict[str, int] = {}
    smiles_to_idx: Dict[str, int] = {}
    for i, row in enumerate(query_rows):
        cid = str(row.get("cid", "")).strip()
        smiles = str(row.get("SMILES", "") or row.get("smiles", "")).strip()
        if cid and cid not in cid_to_idx:
            cid_to_idx[cid] = i
        if smiles and smiles not in smiles_to_idx:
            smiles_to_idx[smiles] = i

    found: Dict[int, np.ndarray] = {}  # query_idx -> embedding vector

    csv_files = list_embedding_csvs(embedding_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {embedding_dir}")

    for csv_path in tqdm(csv_files, desc="Looking up query embeddings"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            emb_cols = get_emb_columns(fieldnames)
            if len(emb_cols) != EMBEDDING_DIM:
                raise ValueError(f"Expected {EMBEDDING_DIM} emb columns, got {len(emb_cols)} in {csv_path}")

            cid_col = "PUBCHEM_COMPOUND_CID" if "PUBCHEM_COMPOUND_CID" in fieldnames else "cid"
            smiles_col = "SMILES" if "SMILES" in fieldnames else "smiles"

            for row in reader:
                ref_cid = str(row.get(cid_col, "")).strip()
                ref_smiles = str(row.get(smiles_col, "")).strip()

                idx: Optional[int] = None
                if ref_cid in cid_to_idx:
                    idx = cid_to_idx[ref_cid]
                elif ref_smiles in smiles_to_idx:
                    idx = smiles_to_idx[ref_smiles]

                if idx is not None and idx not in found:
                    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                    for j, col in enumerate(emb_cols):
                        val = row.get(col, "")
                        try:
                            vec[j] = float(val) if val != "" else 0.0
                        except (ValueError, TypeError):
                            vec[j] = 0.0
                    found[idx] = vec

        # Early exit if all queries matched
        if len(found) == len(query_rows):
            break

    matched_indices = sorted(found.keys())
    embeddings = np.stack([found[i] for i in matched_indices], axis=0)
    matched_rows = [query_rows[i] for i in matched_indices]
    print(f"  Matched embeddings for {len(matched_indices)} / {len(query_rows)} query molecules")
    return embeddings, matched_rows, matched_indices


# ---------------------------------------------------------------------------
# Step 3: k-NN search (exclude self by CID)
# ---------------------------------------------------------------------------

def run_knn(
    query_embeddings: np.ndarray,
    query_cids: List[str],
    embedding_dir: str,
    k_per_query: List[int],
    chunk_size: int = CHUNK_SIZE,
) -> List[List[Tuple[float, str, str, str]]]:
    """
    For each query embedding, find k nearest neighbors from all reference CSVs,
    excluding the query molecule itself (matched by CID).
    k_per_query allows a different k for each query.

    Returns list of length n_queries; each element is a sorted list of
    (distance, ref_cid, ref_smiles, ref_status).
    """
    n_queries = query_embeddings.shape[0]
    query_cid_set = [set() for _ in range(n_queries)]
    for q, cid in enumerate(query_cids):
        if cid:
            query_cid_set[q].add(cid)

    # Max-heap of (-distance, cid, smiles, status); size capped at k per query
    heaps: List[list] = [[] for _ in range(n_queries)]

    csv_files = list_embedding_csvs(embedding_dir)

    for csv_path in tqdm(csv_files, desc="k-NN search"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            emb_cols = get_emb_columns(fieldnames)

            cid_col = "PUBCHEM_COMPOUND_CID" if "PUBCHEM_COMPOUND_CID" in fieldnames else "cid"
            smiles_col = "SMILES" if "SMILES" in fieldnames else "smiles"
            status_col = "status"

            chunk_rows: List[dict] = []
            for row in reader:
                chunk_rows.append(row)
                if len(chunk_rows) >= chunk_size:
                    _update_heaps(
                        query_embeddings, query_cid_set, chunk_rows,
                        emb_cols, cid_col, smiles_col, status_col, heaps, k_per_query,
                    )
                    chunk_rows = []

            if chunk_rows:
                _update_heaps(
                    query_embeddings, query_cid_set, chunk_rows,
                    emb_cols, cid_col, smiles_col, status_col, heaps, k_per_query,
                )

    results = []
    for h in heaps:
        sorted_h = sorted(h, key=lambda x: -x[0])  # ascending distance
        results.append([(abs(x[0]), x[1], x[2], x[3]) for x in sorted_h])
    return results


def _update_heaps(
    query_emb: np.ndarray,
    query_cid_set: List[set],
    rows: List[dict],
    emb_cols: List[str],
    cid_col: str,
    smiles_col: str,
    status_col: str,
    heaps: List[list],
    k_per_query: List[int],
) -> None:
    """Vectorised distance computation + heap update, skipping self-matches."""
    n = len(rows)
    ref_emb = np.zeros((n, len(emb_cols)), dtype=np.float32)
    cids: List[str] = []
    smiles_list: List[str] = []
    statuses: List[str] = []
    for i, row in enumerate(rows):
        for j, col in enumerate(emb_cols):
            val = row.get(col, "")
            try:
                ref_emb[i, j] = float(val) if val != "" else 0.0
            except (ValueError, TypeError):
                ref_emb[i, j] = 0.0
        cids.append(str(row.get(cid_col, "")).strip())
        smiles_list.append(str(row.get(smiles_col, "")).strip())
        statuses.append(str(row.get(status_col, "")).strip())

    distances = cdist(query_emb, ref_emb, metric="euclidean")  # (n_q, n_ref)
    n_queries = distances.shape[0]

    for q in range(n_queries):
        k = k_per_query[q]
        for j in range(n):
            # Skip self
            if cids[j] in query_cid_set[q]:
                continue
            d = float(distances[q, j])
            entry = (-d, cids[j], smiles_list[j], statuses[j])
            if len(heaps[q]) < k:
                heapq.heappush(heaps[q], entry)
            elif d < -heaps[q][0][0]:
                heapq.heapreplace(heaps[q], entry)


# ---------------------------------------------------------------------------
# Step 4: Write output CSV
# ---------------------------------------------------------------------------

def write_results(
    output_path: str,
    matched_rows: List[dict],
    knn_results: List[List[Tuple[float, str, str, str]]],
) -> None:
    """Write one row per (query, neighbor) pair."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fieldnames = [
        "query_cid", "query_smiles", "query_energy",
        "query_category", "query_functional_group", "query_formula",
        "rank", "ref_cid", "ref_smiles", "ref_status", "distance",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, qrow in enumerate(matched_rows):
            q_cid = str(qrow.get("cid", "")).strip()
            q_smiles = str(qrow.get("SMILES", "") or qrow.get("smiles", "")).strip()
            q_energy = qrow.get("adsorption_energy", "")
            q_category = str(qrow.get("query_category", "")).strip()
            q_fg = str(qrow.get("functional_group", "")).strip()
            q_formula = str(qrow.get("formula", "")).strip()
            for rank, (dist, ref_cid, ref_smiles, ref_status) in enumerate(knn_results[i], start=1):
                writer.writerow({
                    "query_cid": q_cid,
                    "query_smiles": q_smiles,
                    "query_energy": q_energy,
                    "query_category": q_category,
                    "query_functional_group": q_fg,
                    "query_formula": q_formula,
                    "rank": rank,
                    "ref_cid": ref_cid,
                    "ref_smiles": ref_smiles,
                    "ref_status": ref_status,
                    "distance": f"{dist:.6f}",
                })
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="k-NN search for strong-binding downstream molecules using pre-computed embeddings."
    )
    parser.add_argument("--downstream_csv", default=None,
                        help="Path to downstream CSV (default: from config.py)")
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR,
                        help="Folder of reference embedding CSVs")
    parser.add_argument("--threshold", type=float, default=-1.7,
                        help="Binding energy threshold in eV (select < threshold)")
    parser.add_argument("--weak_low", type=float, default=-0.5,
                        help="Lower bound for weak binder range (default: -0.5)")
    parser.add_argument("--weak_high", type=float, default=0.0,
                        help="Upper bound for weak binder range (default: 0.0)")
    parser.add_argument("--k", type=int, default=15,
                        help="Number of neighbors for strong binders (excluding self)")
    parser.add_argument("--k_weak", type=int, default=5,
                        help="Number of neighbors for weak binders (excluding self)")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help="Rows per chunk when reading reference CSVs")
    parser.add_argument("--output", default="AL_knn_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    config = Config()
    downstream_csv = args.downstream_csv or config.downstream_csv

    # Step 1: filter strong + weak binders
    print(f"Loading downstream CSV: {downstream_csv}")
    print(f"Selecting molecules with adsorption_energy < {args.threshold} eV (strong binders)")
    print(f"  and adsorption_energy in [{args.weak_low}, {args.weak_high}] eV (weak binders) ...")
    query_rows = load_selected_molecules(
        downstream_csv, args.threshold,
        weak_low=args.weak_low, weak_high=args.weak_high,
    )
    n_strong = sum(1 for r in query_rows if r["query_category"] == "strong_binder")
    n_weak = sum(1 for r in query_rows if r["query_category"] == "weak_binder")
    print(f"  Selected {len(query_rows)} molecules ({n_strong} strong, {n_weak} weak)")
    if not query_rows:
        print("No molecules selected. Exiting.")
        return

    # Step 2: look up embeddings from reference CSVs
    print(f"\nLooking up query embeddings from {args.embedding_dir} ...")
    query_embeddings, matched_rows, _ = lookup_query_embeddings(query_rows, args.embedding_dir)
    if len(matched_rows) == 0:
        print("No query embeddings matched. Exiting.")
        return

    query_cids = [str(r.get("cid", "")).strip() for r in matched_rows]

    # Step 3: k-NN search (per-query k: strong -> args.k, weak -> args.k_weak)
    k_per_query = [
        args.k if r.get("query_category") == "strong_binder" else args.k_weak
        for r in matched_rows
    ]
    print(f"\nRunning k-NN search (k={args.k} strong, k={args.k_weak} weak, excluding self) ...")
    knn_results = run_knn(
        query_embeddings, query_cids, args.embedding_dir,
        k_per_query=k_per_query, chunk_size=args.chunk_size,
    )

    # Step 4: write output
    print(f"\nWriting results to {args.output} ...")
    write_results(args.output, matched_rows, knn_results)


if __name__ == "__main__":
    main()
