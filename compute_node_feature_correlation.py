"""
Compute Pearson correlation among GNN node features on a random sample of molecules.

Streams a large SMILES CSV (e.g. combine.csv) with reservoir sampling so the full
~7M-molecule file never has to fit in memory. Each atom is one row; features match
the SSL MolToGraphConverter (7 physicochemical / categorical fields, no binding tag).

Usage:
    python compute_node_feature_correlation.py
    python compute_node_feature_correlation.py \\
        --input /kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/combine.csv \\
        --output-dir ./analysis/node_feature_corr \\
        --num-molecules 100000 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset.ssl.molecular_graph import MolToGraphConverter

# Column order for outputs (user-facing names; order differs from internal tensor layout).
FEATURE_NAMES = [
    "atomic_number",
    "partial_charge",
    "hybridization",
    "coordination_number",
    "valence_electrons",
    "electronegativity",
    "tetrahedral_chirality",
]

DEFAULT_INPUT = (
    "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/combine.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pearson correlation matrix for atom-level GNN node features (random sample)."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="CSV with PUBCHEM_COMPOUND_CID and SMILES columns (default: combine.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "analysis", "node_feature_correlation"),
        help="Directory for correlation_matrix.csv and heatmap PNG.",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=100_000,
        help="Number of molecules to sample via reservoir sampling (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling (default: 42).",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Optional cap on total atom rows kept (subsample after extraction).",
    )
    return parser.parse_args()


def iter_smiles_rows(csv_path: str) -> Iterator[str]:
    """Yield non-empty SMILES strings from CSV, one row at a time."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        smiles_key = fields.get("smiles")
        if smiles_key is None:
            raise ValueError(
                "CSV must contain a SMILES column. Found: {}".format(reader.fieldnames)
            )
        for row in reader:
            smiles = (row.get(smiles_key) or "").strip()
            if smiles:
                yield smiles


def reservoir_sample_smiles(
    csv_path: str, k: int, seed: int
) -> Tuple[List[str], int]:
    """
    Reservoir sample k SMILES from a stream without storing all molecules.

    Returns:
        (sampled_smiles, total_rows_seen)
    """
    if k <= 0:
        raise ValueError("--num-molecules must be positive.")

    rng = np.random.default_rng(seed)
    reservoir: List[str] = []
    total = 0

    for smiles in iter_smiles_rows(csv_path):
        if len(reservoir) < k:
            reservoir.append(smiles)
        else:
            j = int(rng.integers(0, total + 1))
            if j < k:
                reservoir[j] = smiles
        total += 1

    if total == 0:
        raise ValueError("No SMILES rows found in {}".format(csv_path))
    if len(reservoir) < k:
        print(
            "Warning: only {} molecules in file (requested sample {}).".format(
                len(reservoir), k
            )
        )
    return reservoir, total


def atoms_to_feature_matrix(converter: MolToGraphConverter, mol: Chem.Mol) -> np.ndarray:
    """
    Extract (n_atoms, 7) float matrix aligned with FEATURE_NAMES.

    Uses MolToGraphConverter so features match training graphs:
      x[:,0] atomic_num, x[:,1] chiral tag, x[:,2] partial_charge, x[:,3] hybridization,
      x[:,4] coordination (degree), x[:,5] valence_electrons, x[:,6] electronegativity.
    """
    graph = converter.convert(mol)
    x = graph.x.detach().cpu().numpy()
    if x.shape[1] < 7:
        raise ValueError("Expected at least 7 node features, got shape {}".format(x.shape))

    # Reorder to FEATURE_NAMES column order.
    return np.column_stack(
        [
            x[:, 0],  # atomic_number
            x[:, 2],  # partial_charge
            x[:, 3],  # hybridization (RDKit enum as float)
            x[:, 4],  # coordination_number
            x[:, 5],  # valence_electrons
            x[:, 6],  # electronegativity
            x[:, 1],  # tetrahedral_chirality (RDKit ChiralType)
        ]
    ).astype(np.float64, copy=False)


def build_atom_feature_table(
    smiles_list: List[str],
    max_atoms: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Parse SMILES and stack atom rows into a single 2D array.

    Returns:
        features: (n_atoms, 7)
        stats: processing counters
    """
    converter = MolToGraphConverter()
    chunks: List[np.ndarray] = []
    stats = {
        "molecules_requested": len(smiles_list),
        "molecules_parsed": 0,
        "molecules_failed": 0,
        "atoms_extracted": 0,
    }

    for smiles in tqdm(smiles_list, desc="Extracting atom features"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats["molecules_failed"] += 1
            continue
        try:
            mat = atoms_to_feature_matrix(converter, mol)
        except Exception:
            stats["molecules_failed"] += 1
            continue
        if mat.size == 0:
            stats["molecules_failed"] += 1
            continue
        chunks.append(mat)
        stats["molecules_parsed"] += 1
        stats["atoms_extracted"] += mat.shape[0]

    if not chunks:
        raise RuntimeError("No atom features extracted from sampled molecules.")

    features = np.vstack(chunks)
    if max_atoms is not None and features.shape[0] > max_atoms:
        rng = np.random.default_rng(0)
        idx = rng.choice(features.shape[0], size=max_atoms, replace=False)
        features = features[idx]
        stats["atoms_after_cap"] = max_atoms
    return features, stats


def clean_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, List[str]]:
    """
    Drop rows with NaN/inf and remove constant columns.

    Returns:
        cleaned, col_mask, kept_names, n_bad_rows, dropped_constant_names
    """
    finite_mask = np.isfinite(features).all(axis=1)
    cleaned = features[finite_mask]
    n_bad = int((~finite_mask).sum())

    std = np.nanstd(cleaned, axis=0)
    const_mask = std > 0.0
    dropped = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if not const_mask[i]]
    cleaned = cleaned[:, const_mask]
    kept_names = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if const_mask[i]]
    return cleaned, const_mask, kept_names, n_bad, dropped


def compute_correlation(features: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix via numpy (no pandas/pytz dependency)."""
    # np.corrcoef expects variables in rows; shape (n_features, n_atoms).
    return np.corrcoef(features, rowvar=False)


def save_correlation_csv(corr: np.ndarray, feature_names: List[str], path: str) -> None:
    """Write labeled correlation matrix to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + feature_names)
        for name, row in zip(feature_names, corr):
            writer.writerow([name] + ["{:.6f}".format(v) for v in row])


def save_heatmap(
    corr: np.ndarray, feature_names: List[str], path: str, title: str
) -> None:
    """Save correlation matrix as a labeled heatmap PNG."""
    n = len(feature_names)
    fig_w = max(6.0, 0.55 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, "{:.2f}".format(val),
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.5 else "black",
                )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit("Input CSV not found: {}".format(args.input))

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "node_feature_correlation_matrix.csv")
    out_png = os.path.join(args.output_dir, "node_feature_correlation_heatmap.png")
    out_stats = os.path.join(args.output_dir, "sampling_stats.txt")

    print("Reservoir sampling {} molecules from:\n  {}".format(args.num_molecules, args.input))
    sampled, total_rows = reservoir_sample_smiles(
        args.input, k=args.num_molecules, seed=args.seed
    )
    print("  Total SMILES rows scanned: {:,}".format(total_rows))
    print("  Reservoir size: {:,}".format(len(sampled)))

    print("\nBuilding atom-level feature matrix...")
    features, proc_stats = build_atom_feature_table(sampled, max_atoms=args.max_atoms)
    print("  Raw atom rows: {:,}".format(features.shape[0]))

    cleaned, col_mask, kept_names, n_bad_rows, dropped_cols = clean_features(features)
    print("  Rows removed (NaN/inf): {:,}".format(n_bad_rows))
    if dropped_cols:
        print("  Constant columns removed: {}".format(", ".join(dropped_cols)))
    print("  Atom rows used: {:,}".format(cleaned.shape[0]))
    print("  Features used: {}".format(", ".join(kept_names)))

    if cleaned.shape[1] < 2:
        raise SystemExit("Need at least 2 non-constant features for correlation.")

    corr = compute_correlation(cleaned)
    save_correlation_csv(corr, kept_names, out_csv)
    print("\nSaved correlation matrix: {}".format(out_csv))

    save_heatmap(
        corr,
        kept_names,
        out_png,
        title="Node feature Pearson correlation (n={:,} atoms)".format(cleaned.shape[0]),
    )
    print("Saved heatmap: {}".format(out_png))

    with open(out_stats, "w", encoding="utf-8") as f:
        f.write("input={}\n".format(args.input))
        f.write("total_smiles_rows={}\n".format(total_rows))
        f.write("num_molecules_sampled={}\n".format(len(sampled)))
        f.write("seed={}\n".format(args.seed))
        for key, val in proc_stats.items():
            f.write("{}={}\n".format(key, val))
        f.write("atom_rows_raw={}\n".format(features.shape[0]))
        f.write("atom_rows_clean={}\n".format(cleaned.shape[0]))
        f.write("rows_dropped_nan_inf={}\n".format(n_bad_rows))
        f.write("features_kept={}\n".format(",".join(kept_names)))
        if dropped_cols:
            f.write("features_dropped_constant={}\n".format(",".join(dropped_cols)))
    print("Saved stats: {}".format(out_stats))


if __name__ == "__main__":
    main()
