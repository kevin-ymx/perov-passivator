"""
k-NN search in finetuned GIN-E embedding space.

For each molecule in molecules_cid_smiles_finetuned.csv, finds the 10 molecules from
filtered_csv_latest_embeddings_finetuned with smallest L2 distance. Query embeddings
are computed with the finetuned GIN-E encoder.

Usage:
    python knn_finetunedembedding_search.py
    python knn_finetunedembedding_search.py --query_csv path/to/molecules.csv --embedding_dir path/to/embeddings --checkpoint path/to/gin_e_finetuned.pt --output results.csv
"""
import os
import sys
import csv
import argparse
import glob
import heapq
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# Project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from models.gin_e import GINEEncoder
from inference_Eb import mol_to_graph, smiles_to_mol


# Default paths
DEFAULT_QUERY_CSV = "/kfs3/scratch/yeming/ai4m/prediction/dataset/literature/molecule_images_by_journal/molecules_cid_smiles_finetuned.csv"
DEFAULT_EMBEDDING_DIR = "/kfs3/scratch/yeming/ai4m/prediction/filtered_csv_latest_embeddings_finetuned"
DEFAULT_CHECKPOINT = "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/downstream_notag_03222026/downstream/gin_e_finetuned.pt"
EMBEDDING_DIM = 256
K_NEIGHBORS = 21
CHUNK_SIZE = 100000


def load_finetuned_encoder(checkpoint_path: str, config: Config, device: torch.device) -> GINEEncoder:
    """Load finetuned GIN-E encoder from checkpoint."""
    model = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout,
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GIN-E checkpoint not found: {checkpoint_path}")
    print(f"Loading finetuned GIN-E encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'encoder_state_dict' in ckpt:
        model.load_state_dict(ckpt['encoder_state_dict'])
        print(f"  Loaded encoder from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
        print("  Loaded encoder weights")
    model = model.to(device)
    model.eval()
    return model


def load_query_molecules(csv_path: str) -> List[dict]:
    """
    Load query CSV: molecule_name, cid, smiles, journal.
    Handles molecule_name containing commas by coalescing extra fields.
    """
    rows = []
    num_cols = 4
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


def load_existing_query_cids(result_csv_path: str) -> set:
    """Load set of query_cid already present in an existing result CSV."""
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
    encoder: GINEEncoder,
    device: torch.device,
) -> Tuple[np.ndarray, List[dict], List[int]]:
    """
    Compute finetuned GIN-E embeddings for query molecules (by SMILES).
    Returns (embeddings array [n_valid, 256], list of valid row dicts, original indices).
    """
    smiles_col = "smiles" if "smiles" in (query_rows[0] or {}) else "SMILES"
    valid_graphs = []
    valid_rows = []
    valid_indices = []

    for i, row in enumerate(query_rows):
        smiles = (row.get(smiles_col) or "").strip()
        if not smiles:
            continue
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        try:
            graph = mol_to_graph(mol)
        except Exception:
            continue
        valid_graphs.append(graph)
        valid_rows.append(row)
        valid_indices.append(i)

    if not valid_graphs:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), [], []

    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(valid_graphs), 64):
            batch_graphs = valid_graphs[start:start + 64]
            batch = Batch.from_data_list(batch_graphs).to(device)
            emb = encoder(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, valid_rows, valid_indices


def list_embedding_csvs(embedding_dir: str) -> List[str]:
    """List and sort CSV files by leading number."""
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
    """Get emb_0 .. emb_255 column names, sorted by index."""
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
    heaps: List[list] = [[] for _ in range(n_queries)]

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

    result = []
    for h in heaps:
        sorted_h = sorted(h, key=lambda x: -x[0])
        result.append([(abs(x[0]), x[1], x[2], x[3]) for x in sorted_h])
    return result


# Match SMILES isotope tokens like [13C], [2H], [15n], [235U], etc.
_ISOTOPE_RE = re.compile(r"\[\d+[A-Za-z]")


def has_isotope(smiles: str) -> bool:
    """Return True if the SMILES string contains an isotope-labeled atom."""
    if not smiles:
        return False
    return bool(_ISOTOPE_RE.search(smiles))


def chunk_to_arrays(rows: List[dict], emb_cols: List[str], cid_col: str, smiles_col: str, status_col: str):
    """Convert chunk of CSV rows to embedding matrix and metadata lists.
    Rows whose SMILES contains an isotope label (e.g. [13C], [2H]) are dropped."""
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
    distances = cdist(query_emb, ref_emb, metric="euclidean")
    n_queries = distances.shape[0]
    for q in range(n_queries):
        for j in range(distances.shape[1]):
            d = float(distances[q, j])
            if len(heaps[q]) < k:
                heapq.heappush(heaps[q], (-d, cids[j], smiles_list[j], statuses[j]))
            elif d < -heaps[q][0][0]:
                heapq.heapreplace(heaps[q], (-d, cids[j], smiles_list[j], statuses[j]))


def write_results(
    output_path: str,
    valid_query_rows: List[dict],
    knn_results: List[List[Tuple[float, str, str, str]]],
    name_col: str = "molecule_name",
    cid_col: str = "cid",
    smiles_col: str = "smiles",
    existing_rows: Optional[List[dict]] = None,
) -> None:
    """Write one row per (query, neighbor) pair."""
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
    parser = argparse.ArgumentParser(description="k-NN search in finetuned GIN-E embedding space")
    parser.add_argument("--query_csv", default=DEFAULT_QUERY_CSV, help="Query CSV: molecule_name, cid, smiles, journal")
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR, help="Folder of finetuned embedding CSVs (emb_0..emb_255)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Finetuned GIN-E encoder checkpoint")
    parser.add_argument("--output", default="knn_finetunedembedding_results.csv", help="Output CSV path")
    parser.add_argument("--skip_if_in", default="", help="Path to existing result CSV; query molecules already in this file are skipped")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help="Number of neighbors per query")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Rows per chunk when reading reference CSVs")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    config = Config()
    device = torch.device(args.device if args.device else (config.device if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading query molecules...")
    query_rows = load_query_molecules(args.query_csv)
    print(f"  Loaded {len(query_rows)} rows from {args.query_csv}")

    if args.skip_if_in:
        existing_cids = load_existing_query_cids(args.skip_if_in)
        if existing_cids:
            cid_col = "cid" if query_rows and "cid" in query_rows[0] else "cid"
            query_rows = [r for r in query_rows if (r.get(cid_col) or "").strip() not in existing_cids]
            print(f"  Skipping {len(existing_cids)} query molecule(s) already in {args.skip_if_in}; {len(query_rows)} remaining")

    if not query_rows:
        print("No query molecules to process. Exiting.")
        return

    print("Loading finetuned GIN-E encoder...")
    encoder = load_finetuned_encoder(args.checkpoint, config, device)

    print("Computing query embeddings with finetuned GIN-E...")
    query_embeddings, valid_rows, valid_indices = get_query_embeddings(query_rows, encoder, device)
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
