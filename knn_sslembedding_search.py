"""
k-NN search in GIN-E embedding space.

For each molecule in molecules_cid_smiles.csv, finds the 10 molecules from
filtered_csv_embeddings with smallest L2 distance. Query embeddings are computed
with the GIN-E encoder from the given checkpoint.

Usage:
    python knn_sslembedding_search.py
    python knn_sslembedding_search.py --query_csv path/to/molecules.csv --embedding_dir path/to/embeddings --checkpoint path/to/best_model.pt --output results.csv
    python knn_sslembedding_search.py --diamine_queries --output knn_diamine_eda_pda_cyda.csv
"""
import os
import sys
import csv
import argparse
import glob
import heapq
from typing import List, Tuple, Optional

import re

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Match SMILES isotope tokens like [13C], [2H], [15n], [235U], etc.
# An isotope is signaled by one or more digits immediately after '[' inside a bracket atom.
_ISOTOPE_RE = re.compile(r"\[\d+[A-Za-z]")


def has_isotope(smiles: str) -> bool:
    """Return True if the SMILES string contains an isotope-labeled atom."""
    if not smiles:
        return False
    return bool(_ISOTOPE_RE.search(smiles))

# Project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from inference_ssl import GINEEncoderInference


# Default paths
DEFAULT_QUERY_CSV = "/kfs3/scratch/yeming/ai4m/prediction/dataset/literature/molecule_images_by_journal/molecules_cid_smiles.csv"
DEFAULT_EMBEDDING_DIR = "/kfs3/scratch/yeming/ai4m/prediction/filtered_csv_embeddings"
DEFAULT_CHECKPOINT = "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/best_model.pt"
EMBEDDING_DIM = 256
K_NEIGHBORS = 21
DIAMINE_QUERY_NAMES = ("Ethylenediamine (EDA)", "1,3-Propanediamine (PDA)", "1,4-Diaminocyclohexane (CyDA)", "Piperazine")
DIAMINE_K_NEIGHBORS = 101
CHUNK_SIZE = 100000  # Rows per chunk when reading reference CSVs


def load_query_molecules(csv_path: str) -> List[dict]:
    """
    Load query CSV: molecule_name, cid, smiles, journal.
    Handles molecule_name containing commas by coalescing extra fields into the first column
    (expects 4 columns; if a row has more, the first column is joined with commas).
    Returns list of dicts.
    """
    rows = []
    num_cols = 4  # molecule_name, cid, smiles, journal
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows
        if len(header) > num_cols:
            header = [",".join(header[: len(header) - num_cols + 1]).strip()] + [h.strip() for h in header[-num_cols + 1 :]]
        else:
            header = [h.strip() if h else "" for h in header]
        for row in reader:
            if len(row) < num_cols:
                continue
            if len(row) > num_cols:
                row = [",".join(row[: len(row) - num_cols + 1]).strip()] + [c.strip() if c else "" for c in row[-num_cols + 1 :]]
            else:
                row = [c.strip() if isinstance(c, str) else ("" if c is None else c) for c in row]
            row_dict = {k: (v if isinstance(v, str) else ("" if v is None else str(v))) for k, v in zip(header, row)}
            rows.append(row_dict)
    return rows


def molecule_name_has_abbrev(molecule_name: str, abbrev: str) -> bool:
    """True if abbrev appears as a parenthesized abbreviation in molecule_name.

    Matches e.g. "ethylenediamine (EDA)", "propane-1,3-diammonium (PDA)",
    "1,3-propanediamine- (PDA-)". Avoids false positives like "BPDA" or "2,6-PDA".
    """
    name = (molecule_name or "").strip()
    abbrev = (abbrev or "").strip()
    if not name or not abbrev:
        return False
    if name == abbrev:
        return True
    if f"({abbrev})" in name or f"({abbrev}-" in name:
        return True
    # Standalone token (not embedded in a longer identifier).
    pattern = rf"(?:^|[\s(,]){re.escape(abbrev)}(?:[\s),.\-]|$)"
    return bool(re.search(pattern, name))


def filter_queries_by_abbreviations(
    query_rows: List[dict],
    abbrevs: Tuple[str, ...],
    name_col: str = "molecule_name",
) -> Tuple[List[dict], List[str]]:
    """Select one query row per abbreviation (first match in CSV order).

    Returns (selected_rows, missing_abbreviations).
    """
    selected: List[dict] = []
    missing: List[str] = []
    for abbrev in abbrevs:
        match = None
        for row in query_rows:
            if molecule_name_has_abbrev(row.get(name_col, ""), abbrev):
                match = row
                break
        if match is not None:
            selected.append(match)
        else:
            missing.append(abbrev)
    return selected, missing


def load_existing_query_cids(result_csv_path: str) -> set:
    """Load set of query_cid already present in an existing result CSV (to skip those queries)."""
    seen = set()
    if not result_csv_path or not os.path.isfile(result_csv_path):
        return seen
    with open(result_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "query_cid" not in (reader.fieldnames or []):
            return seen
        for row in reader:
            cid = (row.get("query_cid") or "").strip()
            if cid:
                seen.add(cid)
    return seen


def get_query_embeddings(
    query_rows: List[dict],
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[dict], List[int]]:
    """
    Get GIN-E embeddings for query molecules (by SMILES).
    Returns (embeddings array [n_valid, 256], list of valid row dicts, original indices).
    """
    smiles_col = "smiles" if "smiles" in (query_rows[0] or {}) else "SMILES"
    smiles_list = [r.get(smiles_col, "").strip() for r in query_rows]
    encoder = GINEEncoderInference(
        checkpoint_path=checkpoint_path,
        device=device,
        config=Config(),
    )
    embeddings, status_list, valid_indices = encoder.encode_batch(
        smiles_list, batch_size=64, show_progress=True
    )
    valid_rows = [query_rows[i] for i in valid_indices]
    return embeddings, valid_rows, valid_indices


def list_embedding_csvs(embedding_dir: str) -> List[str]:
    """List and sort CSV files like 000000001_000500000.csv by first number."""
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
    """Get column names that are emb_0, emb_1, ... emb_255, sorted by index."""
    emb = [c for c in fieldnames if c.startswith("emb_") and len(c) > 4 and c[4:].isdigit()]
    return sorted(emb, key=lambda x: int(x.split("_")[1]))


def run_knn(
    query_embeddings: np.ndarray,
    embedding_dir: str,
    k: int = 10,
    chunk_size: int = CHUNK_SIZE,
) -> List[List[Tuple[float, str, str, str]]]:
    """
    For each query embedding, find k nearest neighbors from all CSVs in embedding_dir.
    Returns list of length n_queries; each element is a list of k tuples (distance, cid, smiles, status).
    """
    n_queries = query_embeddings.shape[0]
    # For each query we keep a max-heap of size k (store -distance to get smallest distances)
    # Heap elements: (-distance, cid, smiles, status) so we pop the largest -distance = smallest distance
    heaps: List[List[Tuple[float, str, str, str]]] = [[] for _ in range(n_queries)]

    csv_files = list_embedding_csvs(embedding_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {embedding_dir}")

    for csv_path in tqdm(csv_files, desc="Reference CSVs"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            emb_cols = get_emb_columns(fieldnames)
            if len(emb_cols) != EMBEDDING_DIM:
                raise ValueError(f"Expected {EMBEDDING_DIM} embedding columns, got {len(emb_cols)} in {csv_path}")

            cid_col = "PUBCHEM_COMPOUND_CID" if "PUBCHEM_COMPOUND_CID" in fieldnames else "cid"
            smiles_col = "SMILES" if "SMILES" in fieldnames else "smiles"
            status_col = "status"

            chunk_rows = []
            for row in reader:
                chunk_rows.append(row)
                if len(chunk_rows) >= chunk_size:
                    ref_emb, cids, smiles_list, statuses = chunk_to_arrays(chunk_rows, emb_cols, cid_col, smiles_col, status_col)
                    update_heaps(query_embeddings, ref_emb, cids, smiles_list, statuses, heaps, k)
                    chunk_rows = []

            if chunk_rows:
                ref_emb, cids, smiles_list, statuses = chunk_to_arrays(chunk_rows, emb_cols, cid_col, smiles_col, status_col)
                update_heaps(query_embeddings, ref_emb, cids, smiles_list, statuses, heaps, k)

    # Convert each heap to sorted list (ascending distance). Heap stores (-dist, cid, smiles, status).
    result = []
    for h in heaps:
        sorted_h = sorted(h, key=lambda x: -x[0])  # ascending distance (x[0] is -dist)
        result.append([(abs(x[0]), x[1], x[2], x[3]) for x in sorted_h])
    return result


def chunk_to_arrays(rows: List[dict], emb_cols: List[str], cid_col: str, smiles_col: str, status_col: str):
    """Convert chunk of CSV rows to embedding matrix and metadata lists.
    Rows whose SMILES contains an isotope label (e.g. [13C], [2H]) are dropped."""
    # First filter out isotope-containing rows so we only allocate space for kept rows.
    kept_rows = []
    for row in rows:
        smi = str(row.get(smiles_col, "")).strip()
        if has_isotope(smi):
            continue
        kept_rows.append((row, smi))

    n = len(kept_rows)
    emb = np.zeros((n, len(emb_cols)), dtype=np.float32)
    cids = []
    smiles_list = []
    statuses = []
    for i, (row, smi) in enumerate(kept_rows):
        for j, col in enumerate(emb_cols):
            val = row.get(col, "")
            try:
                emb[i, j] = float(val) if val != "" else 0.0
            except (ValueError, TypeError):
                emb[i, j] = 0.0
        cids.append(str(row.get(cid_col, "")).strip())
        smiles_list.append(smi)
        statuses.append(str(row.get(status_col, "")).strip())
    return emb, cids, smiles_list, statuses


def update_heaps(
    query_emb: np.ndarray,
    ref_emb: np.ndarray,
    cids: List[str],
    smiles_list: List[str],
    statuses: List[str],
    heaps: List[list],
    k: int,
) -> None:
    """Compute L2 distances and update per-query heaps (keep k smallest)."""
    # query_emb (n_q, 256), ref_emb (n_ref, 256) -> distances (n_q, n_ref)
    distances = cdist(query_emb, ref_emb, metric="euclidean")
    n_queries = distances.shape[0]
    for q in range(n_queries):
        for j in range(distances.shape[1]):
            d = float(distances[q, j])
            cid = cids[j]
            smi = smiles_list[j]
            st = statuses[j]
            if len(heaps[q]) < k:
                heapq.heappush(heaps[q], (-d, cid, smi, st))
            else:
                # Replace worst if this is better (smaller distance)
                if d < -heaps[q][0][0]:
                    heapq.heapreplace(heaps[q], (-d, cid, smi, st))


def write_results(
    output_path: str,
    valid_query_rows: List[dict],
    knn_results: List[List[Tuple[float, str, str, str]]],
    name_col: str = "molecule_name",
    cid_col: str = "cid",
    smiles_col: str = "smiles",
    existing_rows: Optional[List[dict]] = None,
) -> None:
    """Write one row per (query, neighbor) with query info, rank, ref_cid, ref_smiles, distance.
    If existing_rows is provided, those rows are written first (e.g. to preserve skipped queries)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fieldnames = [
        "query_name", "query_cid", "query_smiles", "query_journal",
        "rank", "ref_cid", "ref_smiles", "ref_status", "distance",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if existing_rows:
            writer.writerows(existing_rows)
        for i, row in enumerate(valid_query_rows):
            query_name = row.get(name_col, "")
            query_cid = row.get(cid_col, "")
            query_smiles = row.get(smiles_col, "") or row.get("SMILES", "")
            query_journal = row.get("journal", "")
            for rank, (dist, ref_cid, ref_smiles, ref_status) in enumerate(knn_results[i], start=1):
                writer.writerow({
                    "query_name": query_name,
                    "query_cid": query_cid,
                    "query_smiles": query_smiles,
                    "query_journal": query_journal,
                    "rank": rank,
                    "ref_cid": ref_cid,
                    "ref_smiles": ref_smiles,
                    "ref_status": ref_status,
                    "distance": f"{dist:.6f}",
                })
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="k-NN search in GIN-E embedding space")
    parser.add_argument("--query_csv", default=DEFAULT_QUERY_CSV, help="Query CSV: molecule_name, cid, smiles, journal")
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR, help="Folder of embedding CSVs (emb_0..emb_255)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="GIN-E encoder checkpoint")
    parser.add_argument("--output", default="knn_embedding_results.csv", help="Output CSV path")
    parser.add_argument("--skip_if_in", default="", help="Path to existing result CSV; query molecules already in this file are skipped")
    parser.add_argument("--k", type=int, default=None, help="Number of neighbors per query")
    parser.add_argument(
        "--diamine_queries",
        action="store_true",
        help=(
            f"Run k-NN only for {', '.join(DIAMINE_QUERY_NAMES)} "
            f"with k={DIAMINE_K_NEIGHBORS} (from filtered_csv_embeddings)"
        ),
    )
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Rows per chunk when reading reference CSVs")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    if args.k is None:
        args.k = DIAMINE_K_NEIGHBORS if args.diamine_queries else K_NEIGHBORS

    print("Loading query molecules...")
    query_rows = load_query_molecules(args.query_csv)
    print(f"  Loaded {len(query_rows)} rows from {args.query_csv}")

    if args.diamine_queries:
        before = len(query_rows)
        query_rows, missing_abbrevs = filter_queries_by_abbreviations(
            query_rows, DIAMINE_QUERY_NAMES
        )
        print(
            f"  --diamine_queries: matched {len(query_rows)}/{before} rows "
            f"by abbreviation ({', '.join(DIAMINE_QUERY_NAMES)}), k={args.k}"
        )
        for row in query_rows:
            print(f"    -> {(row.get('molecule_name') or '').strip()}")
        if missing_abbrevs:
            print(
                f"  WARNING: no molecule_name containing abbreviation "
                f"(e.g. '(EDA)'): {', '.join(missing_abbrevs)}"
            )
        if not query_rows:
            print("No diamine query molecules found. Exiting.")
            return

    if args.skip_if_in:
        existing_cids = load_existing_query_cids(args.skip_if_in)
        if existing_cids:
            cid_col = "cid" if query_rows and "cid" in query_rows[0] else "cid"
            query_rows = [r for r in query_rows if (r.get(cid_col) or "").strip() not in existing_cids]
            print(f"  Skipping {len(existing_cids)} query molecule(s) already in {args.skip_if_in}; {len(query_rows)} remaining")
        else:
            print(f"  No existing query_cid found in {args.skip_if_in} (or file missing); processing all")

    if not query_rows:
        print("No query molecules to process. Exiting.")
        return

    print("Computing query embeddings with GIN-E...")
    query_embeddings, valid_rows, valid_indices = get_query_embeddings(
        query_rows, args.checkpoint, args.device
    )
    print(f"  Got embeddings for {len(valid_rows)} / {len(query_rows)} molecules")
    if len(valid_rows) == 0:
        print("No valid query embeddings. Exiting.")
        return

    print("Running k-NN over reference embeddings...")
    knn_results = run_knn(
        query_embeddings,
        args.embedding_dir,
        k=args.k,
        chunk_size=args.chunk_size,
    )

    print("Writing results...")
    name_col = "molecule_name" if "molecule_name" in valid_rows[0] else "molecule_name"
    cid_col = "cid" if "cid" in valid_rows[0] else "cid"
    smiles_col = "smiles" if "smiles" in valid_rows[0] else "SMILES"
    existing_rows = None
    if args.skip_if_in and os.path.isfile(args.skip_if_in) and os.path.normpath(args.skip_if_in) == os.path.normpath(args.output):
        with open(args.skip_if_in, "r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        print(f"  Preserving {len(existing_rows)} existing row(s) from {args.output}")
    write_results(
        args.output,
        valid_rows,
        knn_results,
        name_col=name_col,
        cid_col=cid_col,
        smiles_col=smiles_col,
        existing_rows=existing_rows,
    )


if __name__ == "__main__":
    main()
