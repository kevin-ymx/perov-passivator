"""
Inference script for predicting binding energy from SMILES.

Usage:
    python inference_Eb.py --smiles "CCO"
    python inference_Eb.py --csv input.csv --output literature_predictions.csv
    python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb
    python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb --embeddings_only

After batch prediction (not --embeddings_only), a parity scatter (predicted on x-axis,
reference / MLIP on y-axis) is written next to the output CSV as ``*_parity.png`` when the CSV
contains a suitable reference column (see --reference_energy_col). Use --no_parity_plot to skip.
"""
import os
import sys
import argparse
import csv
import glob
import math
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from config import Config
from models.gin_e import GINEEncoder
from models.downstream_model import DownstreamModel


# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    1: 2.20,    # H
    3: 0.98,    # Li
    5: 2.04,    # B
    6: 2.55,    # C
    7: 3.04,    # N
    8: 3.44,    # O
    9: 3.98,    # F
    11: 0.93,   # Na
    12: 1.31,   # Mg
    13: 1.61,   # Al
    14: 1.90,   # Si
    15: 2.19,   # P
    16: 2.58,   # S
    17: 3.16,   # Cl
    19: 0.82,   # K
    20: 1.00,   # Ca
    34: 2.55,   # Se
    35: 2.96,   # Br
    53: 2.66,   # I
}


# Valence electrons map
VALENCE_ELECTRONS = {
    1: 1, 3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
    19: 1, 20: 2, 34: 6, 35: 7, 53: 7,
}


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to RDKit molecule with 3D coordinates.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        RDKit molecule object or None if failed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            # Try to embed 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                # If embedding fails, try with more random seeds
                AllChem.EmbedMolecule(mol, maxAttempts=100, randomSeed=42)
        return mol
    except Exception as e:
        print(f"Error creating molecule from SMILES '{smiles}': {e}")
        return None


def get_partial_charges(mol: Chem.Mol) -> List[float]:
    """Extract or compute Gasteiger partial charges."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        # Replace NaN values with 0.0
        charges = [0.0 if (c != c) else c for c in charges]
        return charges
    except:
        return [0.0] * mol.GetNumAtoms()


def mol_to_graph(mol: Chem.Mol) -> Data:
    """
    Convert a molecule to a PyTorch Geometric graph (binding_tag=0 for all atoms).

    Returns:
        PyTorch Geometric Data object with node and edge features.
    """
    partial_charges = get_partial_charges(mol)

    # Node features: [atomic_num, chirality, partial_charge, hybridization,
    #                 coordination_num, valence_electrons, electronegativity, binding_tag]
    node_features = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        node_features.append([
            float(atomic_num),
            float(int(atom.GetChiralTag())),
            float(partial_charges[atom_idx]),
            float(int(atom.GetHybridization())),
            float(len(atom.GetNeighbors())),
            float(VALENCE_ELECTRONS.get(atomic_num, 4)),
            float(ELECTRONEGATIVITY.get(atomic_num, 2.0)),
            0.0,  # binding_tag always 0
        ])
    
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Edge features: [bond_type, bond_direction]
    edge_index = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        bond_type = int(bond.GetBondType())
        bond_direction = int(bond.GetBondDir())
        
        edge_feat = [float(bond_type), float(bond_direction)]
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        num_nodes=mol.GetNumAtoms()
    )


class BindingEnergyPredictor:
    """
    Predictor class for binding energy inference.
    """
    
    # Default checkpoint paths
    DEFAULT_DOWNSTREAM_CHECKPOINT = "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/downstream_notag_03222026/downstream/downstream_best_model.pt"
    DEFAULT_GINE_CHECKPOINT = "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/downstream_notag_03222026/downstream/gin_e_finetuned.pt"

    def __init__(
        self,
        checkpoint_path: str = None,
        gin_e_checkpoint_path: str = None,
        device: str = None,
        config: Config = None,
        embeddings_only: bool = False,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to the downstream model checkpoint.
            gin_e_checkpoint_path: Path to the finetuned GIN-E encoder checkpoint.
            device: Device to run inference on ('cuda' or 'cpu').
            config: Config object (if None, uses default Config).
            embeddings_only: If True, only load the GIN-E encoder (skip downstream head).
        """
        self.config = config if config is not None else Config()
        self.embeddings_only = embeddings_only

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        if checkpoint_path is None:
            checkpoint_path = self.DEFAULT_DOWNSTREAM_CHECKPOINT
        if gin_e_checkpoint_path is None:
            gin_e_checkpoint_path = self.DEFAULT_GINE_CHECKPOINT

        if embeddings_only:
            self.model = self._load_encoder_only(gin_e_checkpoint_path)
        else:
            self.model = self._load_model(checkpoint_path, gin_e_checkpoint_path)
        self.model.eval()
    
    def _load_encoder_only(self, gin_e_checkpoint_path: str) -> GINEEncoder:
        """Load only the finetuned GIN-E encoder for embeddings-only mode."""
        model = GINEEncoder(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            node_embedding_dim=self.config.node_embedding_dim,
            edge_embedding_dim=self.config.edge_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_gin_layers,
            dropout=self.config.dropout,
        )
        if not os.path.exists(gin_e_checkpoint_path):
            raise FileNotFoundError(f"GIN-E checkpoint not found: {gin_e_checkpoint_path}")
        print(f"Loading finetuned GIN-E encoder from {gin_e_checkpoint_path}...")
        ckpt = torch.load(gin_e_checkpoint_path, map_location=self.device)
        if 'encoder_state_dict' in ckpt:
            model.load_state_dict(ckpt['encoder_state_dict'])
            print(f"  Loaded encoder from epoch {ckpt.get('epoch', '?')}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded encoder weights")
        model = model.to(self.device)
        return model

    def _load_model(self, checkpoint_path: str, gin_e_checkpoint_path: str) -> DownstreamModel:
        """Load the downstream model from checkpoint."""
        # Create GIN-E encoder
        gin_e_encoder = GINEEncoder(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            node_embedding_dim=self.config.node_embedding_dim,
            edge_embedding_dim=self.config.edge_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_gin_layers,
            dropout=self.config.dropout
        )
        
        # Create downstream model (without loading GIN-E checkpoint here, 
        # since we'll load the full model weights)
        model = DownstreamModel(
            gin_e_encoder=gin_e_encoder,
            gin_e_checkpoint_path=None,  # Don't load separately
            freeze_gin_e=True,  # Freeze for inference
            mlp_hidden_dim=self.config.downstream_mlp_hidden_dim,
            mlp_dropout=self.config.downstream_mlp_dropout,
            num_tasks=1,  # Single task: binding energy
            task_hidden_dim=self.config.downstream_task_hidden_dim,
            task_dropout=self.config.downstream_task_dropout
        )
        
        # Load downstream checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Downstream model checkpoint not found: {checkpoint_path}\n"
                f"Please train the model first using train_downstream.py"
            )
        
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"  Loaded model from epoch {epoch}, validation loss: {loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("  Loaded model weights")
        
        model = model.to(self.device)
        return model
    
    def predict_single(self, smiles: str) -> Tuple[Optional[float], str]:
        """
        Predict binding energy for a single molecule.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Tuple of (predicted_energy, status_message).
            predicted_energy is None if prediction failed.
        """
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None, f"Failed to parse SMILES: {smiles}"

        try:
            graph = mol_to_graph(mol)
        except Exception as e:
            return None, f"Failed to convert molecule to graph: {e}"

        graph = graph.to(self.device)

        with torch.no_grad():
            prediction = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=None
            )

        if prediction.dim() > 0:
            predicted_energy = prediction.squeeze().item()
        else:
            predicted_energy = prediction.item()

        return predicted_energy, "OK"
    
    def predict_batch(
        self,
        smiles_list: List[str],
    ) -> List[Tuple[Optional[float], str]]:
        """
        Predict binding energy for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of (predicted_energy, status_message) tuples.
        """
        results = []
        valid_graphs = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            mol = smiles_to_mol(smiles)
            if mol is None:
                results.append((None, f"Failed to parse SMILES: {smiles}"))
                continue

            try:
                graph = mol_to_graph(mol)
                valid_graphs.append(graph)
                valid_indices.append((i, "OK"))
            except Exception as e:
                results.append((None, f"Failed to convert molecule to graph: {e}"))
        
        # Batch inference for valid graphs
        if len(valid_graphs) > 0:
            batched_graph = Batch.from_data_list(valid_graphs).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(
                    x=batched_graph.x,
                    edge_index=batched_graph.edge_index,
                    edge_attr=batched_graph.edge_attr,
                    batch=batched_graph.batch
                )
            
            predictions = predictions.cpu().numpy().flatten()
            
            # Merge results
            pred_idx = 0
            final_results = [None] * len(smiles_list)
            
            for i, status in valid_indices:
                final_results[i] = (float(predictions[pred_idx]), status)
                pred_idx += 1
            
            # Fill in failed results
            result_idx = 0
            for i in range(len(smiles_list)):
                if final_results[i] is None:
                    final_results[i] = results[result_idx]
                    result_idx += 1
            
            return final_results
        
        return results

    def embed_batch(
        self,
        smiles_list: List[str],
    ) -> List[Tuple[Optional[np.ndarray], str]]:
        """
        Extract finetuned GIN-E embeddings for a batch of molecules.

        Returns:
            List of (embedding_array_or_None, status_message) tuples.
        """
        results = []
        valid_graphs = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            mol = smiles_to_mol(smiles)
            if mol is None:
                results.append((None, f"Failed to parse SMILES: {smiles}"))
                continue
            try:
                graph = mol_to_graph(mol)
                valid_graphs.append(graph)
                valid_indices.append((i, "OK"))
            except Exception as e:
                results.append((None, f"Failed to convert molecule to graph: {e}"))

        if len(valid_graphs) > 0:
            batched_graph = Batch.from_data_list(valid_graphs).to(self.device)

            encoder = self.model if self.embeddings_only else self.model.gin_e_encoder
            with torch.no_grad():
                embeddings = encoder(
                    x=batched_graph.x,
                    edge_index=batched_graph.edge_index,
                    edge_attr=batched_graph.edge_attr,
                    batch=batched_graph.batch,
                )
            embeddings = embeddings.cpu().numpy()

            pred_idx = 0
            final_results = [None] * len(smiles_list)
            for i, status in valid_indices:
                final_results[i] = (embeddings[pred_idx], status)
                pred_idx += 1

            result_idx = 0
            for i in range(len(smiles_list)):
                if final_results[i] is None:
                    final_results[i] = results[result_idx]
                    result_idx += 1

            return final_results

        return results


def _count_output_rows(output_csv: str) -> int:
    """Count the number of data rows (excluding header) in an existing output CSV."""
    if not os.path.exists(output_csv):
        return 0
    count = 0
    try:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                count += 1
    except Exception:
        return 0
    return count


def process_csv_file(
    input_csv: str,
    output_csv: str,
    predictor: BindingEnergyPredictor,
    smiles_col: str = 'SMILES',
    batch_size: int = 512,
    embeddings_only: bool = False,
) -> Tuple[int, int]:
    """
    Process a single CSV file: read rows, run inference in batches, write output.
    Supports resuming from partial output: if the output CSV already exists with
    fewer rows than the input, appends remaining rows instead of restarting.

    Returns:
        (n_ok, n_fail) counts (only for newly processed rows).
    """
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return 0, 0

    smiles_col_actual = smiles_col
    if smiles_col not in fieldnames:
        for alt in ['SMILES', 'smiles', 'canonical_smiles']:
            if alt in fieldnames:
                smiles_col_actual = alt
                break

    emb_dim = predictor.config.hidden_dim  # GIN-E output dim
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]

    out_fieldnames = list(fieldnames)
    if embeddings_only:
        out_fieldnames += emb_cols + ['status']
    else:
        out_fieldnames += emb_cols + ['predicted_binding_energy', 'status']

    # Check for partial output and determine resume point
    already_done = _count_output_rows(output_csv)
    if already_done >= len(rows):
        # Fully complete — nothing to do
        return 0, 0
    if already_done > 0:
        # Partial output exists — append remaining rows
        print(f"  Resuming from row {already_done}/{len(rows)} ({len(rows) - already_done} remaining)")
        rows = rows[already_done:]
        open_mode = 'a'
        write_header = False
    else:
        open_mode = 'w'
        write_header = True

    n_ok, n_fail = 0, 0
    with open(output_csv, open_mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        if write_header:
            writer.writeheader()

        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            smiles_list = [(r.get(smiles_col_actual) or "").strip() for r in batch_rows]

            # Get embeddings
            emb_results = predictor.embed_batch(smiles_list)

            # Optionally get predictions
            pred_results = None
            if not embeddings_only:
                pred_results = predictor.predict_batch(smiles_list)

            for j, row in enumerate(batch_rows):
                emb, emb_status = emb_results[j]
                if emb is not None:
                    for k, col in enumerate(emb_cols):
                        row[col] = f"{emb[k]:.6f}"
                    row['status'] = 'Success'
                    if not embeddings_only and pred_results is not None:
                        energy, _ = pred_results[j]
                        row['predicted_binding_energy'] = f"{energy:.6f}" if energy is not None else 'N/A'
                    n_ok += 1
                else:
                    for col in emb_cols:
                        row[col] = ''
                    row['status'] = emb_status
                    if not embeddings_only:
                        row['predicted_binding_energy'] = 'N/A'
                    n_fail += 1
                writer.writerow(row)

    return n_ok, n_fail


def predict_from_csv(
    input_csv: str,
    output_csv: str,
    predictor: BindingEnergyPredictor,
    smiles_col: str = 'SMILES',
):
    """
    Predict binding energies from a CSV file.
    
    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to output CSV file.
        predictor: BindingEnergyPredictor instance.
        smiles_col: Column name for SMILES.
    """
    print(f"Reading input from {input_csv}...")

    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"Found {len(rows)} samples")

    smiles_list = [row[smiles_col] for row in rows]

    print("Running predictions...")
    results = predictor.predict_batch(smiles_list)
    
    # Write output
    print(f"Writing results to {output_csv}...")
    output_fieldnames = list(fieldnames) + ['predicted_binding_energy', 'prediction_status']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row, (energy, status) in zip(rows, results):
            row['predicted_binding_energy'] = energy if energy is not None else 'N/A'
            row['prediction_status'] = status
            writer.writerow(row)
    
    # Summary
    successful = sum(1 for energy, _ in results if energy is not None)
    print(f"\nPrediction complete!")
    print(f"  Successful: {successful}/{len(rows)}")
    print(f"  Results saved to: {output_csv}")


def _norm_header(h: str) -> str:
    return (h or "").strip().lstrip("\ufeff").lower()


def _header_lookup(fieldnames: Optional[List[str]]) -> Dict[str, str]:
    return {_norm_header(h): h for h in (fieldnames or []) if h}


def _first_col(lookup: Dict[str, str], candidates: List[str], skip_lower: Optional[set] = None) -> Optional[str]:
    skip = skip_lower or set()
    for c in candidates:
        k = _norm_header(c)
        if k in lookup and k not in skip:
            return lookup[k]
    return None


# Column names for parity plot (predicted vs MLIP / reference)
_PREDICTED_ENERGY_COLS = [
    "predicted_binding_energy",
    "predicted binding energy",
    "predicted_eb",
    "pred_binding_energy",
]
_REFERENCE_ENERGY_COLS = [
    "mlip_binding_energy",
    "mlip adsorption energy",
    "mlip_adsorption_energy",
    "eb_mlip",
    "binding_energy_mlip",
    "mlip_energy",
    "mlip",
    "reference_binding_energy",
    "dft_binding_energy",
    "adsorption_energy",
    "binding_energy",
]


def _parse_energy_cell(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def plot_predicted_vs_reference_parity(
    csv_path: str,
    *,
    reference_col: Optional[str] = None,
    predicted_col: Optional[str] = None,
    parity_output: Optional[str] = None,
) -> Optional[str]:
    """Scatter reference (e.g. MLIP) vs predicted binding energy with y=x line.

    Horizontal axis: predicted; vertical axis: reference (MLIP / DFT).
    Reads ``csv_path`` (e.g. ``literature_predictions.csv``). Resolves columns
    case-insensitively; reference defaults to the first match in
    ``_REFERENCE_ENERGY_COLS`` that is not the predicted column.

    Saves PNG using non-interactive Agg backend. Returns output path or None if skipped.
    """
    if not os.path.isfile(csv_path):
        print(f"Parity plot skipped: file not found: {csv_path}")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Parity plot skipped: matplotlib not available ({e})")
        return None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print("Parity plot skipped: no data rows in CSV.")
        return None

    lookup = _header_lookup(fieldnames)

    pred_key = None
    if predicted_col:
        nk = _norm_header(predicted_col)
        pred_key = lookup.get(nk)
    if pred_key is None:
        pred_key = _first_col(lookup, _PREDICTED_ENERGY_COLS)
    if pred_key is None:
        print(
            "Parity plot skipped: no predicted energy column "
            f"(tried {', '.join(_PREDICTED_ENERGY_COLS[:3])}...)."
        )
        return None

    ref_key = None
    if reference_col:
        nk = _norm_header(reference_col)
        ref_key = lookup.get(nk)
        if ref_key is None:
            print(f"Parity plot skipped: --reference_energy_col {reference_col!r} not in CSV header.")
            return None
    else:
        skip = {_norm_header(pred_key)}
        ref_key = _first_col(lookup, _REFERENCE_ENERGY_COLS, skip_lower=skip)
    if ref_key is None:
        print(
            "Parity plot skipped: no reference / MLIP energy column. "
            "Add a column (e.g. mlip_binding_energy) to the CSV or pass --reference_energy_col."
        )
        return None

    if _norm_header(ref_key) == _norm_header(pred_key):
        print("Parity plot skipped: reference and predicted columns are the same.")
        return None

    xs: List[float] = []
    ys: List[float] = []

    for row in rows:
        x = _parse_energy_cell(row.get(ref_key))
        y = _parse_energy_cell(row.get(pred_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    if len(xs) < 1:
        print(f"Parity plot skipped: no valid ({ref_key} vs {pred_key}) pairs.")
        return None

    arr_ref = np.asarray(xs, dtype=np.float64)
    arr_pred = np.asarray(ys, dtype=np.float64)
    mae = float(np.mean(np.abs(arr_pred - arr_ref)))
    rmse = float(np.sqrt(np.mean((arr_pred - arr_ref) ** 2)))
    ss_res = float(np.sum((arr_ref - arr_pred) ** 2))
    ss_tot = float(np.sum((arr_ref - np.mean(arr_ref)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    out_png = parity_output
    if not out_png:
        root, _ = os.path.splitext(os.path.abspath(csv_path))
        out_png = root + "_parity.png"

    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor="white")
    ax.scatter(
        arr_pred,
        arr_ref,
        s=52,
        alpha=0.88,
        color="#3D5A80",
        edgecolors="#2A3238",
        linewidths=0.6,
        zorder=2,
    )

    lo, hi = -3.0, 0.5
    ax.plot([lo, hi], [lo, hi], color="#9A9A9A", linestyle="--", linewidth=1.0, zorder=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"{pred_key} (eV)", fontsize=12)
    ax.set_ylabel(f"{ref_key} (eV)", fontsize=12)
    ax.set_title("Reference vs predicted binding energy", fontsize=13)
    ax.tick_params(axis="both", direction="in", labelsize=10)
    ax.grid(True, linestyle="-", alpha=0.22, linewidth=0.6)

    stat = f"N = {len(xs)}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV"
    if math.isfinite(r2):
        stat += f"\nR² = {r2:.3f}"
    ax.text(
        0.04,
        0.96,
        stat,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Parity plot saved: {out_png}")
    return out_png


def main():
    parser = argparse.ArgumentParser(
        description='Predict binding energy from SMILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python inference_Eb.py --smiles "CCO"

  # Batch prediction; writes literature_predictions.csv and literature_predictions_parity.png
  python inference_Eb.py --csv input.csv --output literature_predictions.csv

  # Process all CSVs in a folder (embeddings + predictions)
  python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb

  # Embeddings only (no binding energy prediction)
  python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb --embeddings_only
        """
    )

    # Input options
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    parser.add_argument('--csv', type=str, help='Path to input CSV file for batch prediction')
    parser.add_argument('--output', type=str, default='literature_predictions.csv',
                        help='Path to output CSV file (default: literature_predictions.csv)')
    parser.add_argument('--filtered_csv', type=str, default=None,
                        help='Folder of CSV files to process (like inference_ssl.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for folder mode (default: filtered_csv_Eb next to input)')

    # Mode
    parser.add_argument('--embeddings_only', action='store_true',
                        help='Only store finetuned GIN-E embeddings, skip binding energy prediction')

    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to downstream model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu, default: auto-detect)')

    # CSV column options
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Column name for SMILES in CSV (default: SMILES)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for folder-mode inference (default: 512)')

    parser.add_argument(
        '--no_parity_plot',
        action='store_true',
        help='Skip predicted vs reference (e.g. MLIP) parity scatter after writing prediction CSVs.',
    )
    parser.add_argument(
        '--reference_energy_col',
        type=str,
        default=None,
        help='CSV column for reference binding energy (e.g. MLIP). Auto-detected if omitted.',
    )
    parser.add_argument(
        '--predicted_energy_col',
        type=str,
        default=None,
        help='Override predicted column (default: predicted_binding_energy).',
    )
    parser.add_argument(
        '--parity_plot_output',
        type=str,
        default=None,
        help='Output PNG path for parity plot (default: <output_csv>_parity.png next to the CSV).',
    )

    # Multi-GPU / multi-node
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of this process (0 to world_size-1) for splitting CSV files across workers')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of workers (default: 1)')

    args = parser.parse_args()

    if args.filtered_csv is None and args.csv is None and args.smiles is None:
        parser.error("One of --filtered_csv, --csv, or --smiles is required")

    print("="*60)
    print("Binding Energy Prediction")
    print("="*60)

    predictor = BindingEnergyPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        embeddings_only=args.embeddings_only,
    )

    if args.filtered_csv:
        # Folder mode: process all CSVs
        folder = os.path.abspath(args.filtered_csv)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(folder), "filtered_csv_Eb")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        all_csv = sorted(glob.glob(os.path.join(folder, "*.csv")))

        # Determine which files need processing: not started or partially complete
        pending = []
        complete = 0
        for f in all_csv:
            out_path = os.path.join(output_dir, os.path.basename(f))
            if not os.path.exists(out_path):
                pending.append(f)
            else:
                # Count input rows to check if output is complete
                with open(f, 'r', encoding='utf-8') as fh:
                    n_input = sum(1 for _ in fh) - 1  # subtract header
                n_output = _count_output_rows(out_path)
                if n_output < n_input:
                    pending.append(f)  # partial — needs resume
                else:
                    complete += 1

        if args.world_size > 1:
            csv_files = [f for i, f in enumerate(pending) if i % args.world_size == args.rank]
            print(f"Rank {args.rank}/{args.world_size}: processing {len(csv_files)} of {len(pending)} pending CSV file(s)")
        else:
            csv_files = pending
            if complete > 0:
                print(f"Skipping {complete} already-complete file(s)")

        mode_str = "embeddings only" if args.embeddings_only else "embeddings + binding energy"
        if not csv_files:
            print(f"No CSV files to process" + (f" for rank {args.rank}" if args.world_size > 1 else ""))
        else:
            print(f"\nProcessing {len(csv_files)} CSV file(s) from {folder} ({mode_str})")
            print(f"Output directory: {output_dir}\n")
            total_ok, total_fail = 0, 0
            pbar = tqdm(csv_files, desc="CSV files", unit="file")
            for csv_path in pbar:
                basename = os.path.basename(csv_path)
                out_path = os.path.join(output_dir, basename)
                pbar.set_postfix_str(basename)
                n_ok, n_fail = process_csv_file(
                    csv_path, out_path, predictor,
                    smiles_col=args.smiles_col,
                    batch_size=args.batch_size,
                    embeddings_only=args.embeddings_only,
                )
                total_ok += n_ok
                total_fail += n_fail
                if not args.embeddings_only and not args.no_parity_plot:
                    plot_predicted_vs_reference_parity(
                        out_path,
                        reference_col=args.reference_energy_col,
                        predicted_col=args.predicted_energy_col,
                        parity_output=None,
                    )
            print(f"\nSummary: {total_ok} processed, {total_fail} failed across {len(csv_files)} file(s)"
                  + (f", {complete} already complete" if complete else ""))

    elif args.csv:
        predict_from_csv(
            input_csv=args.csv,
            output_csv=args.output,
            predictor=predictor,
            smiles_col=args.smiles_col,
        )
        if not args.embeddings_only and not args.no_parity_plot:
            plot_predicted_vs_reference_parity(
                args.output,
                reference_col=args.reference_energy_col,
                predicted_col=args.predicted_energy_col,
                parity_output=args.parity_plot_output,
            )
    else:
        print(f"\nInput:")
        print(f"  SMILES: {args.smiles}")

        energy, status = predictor.predict_single(args.smiles)

        print(f"\nResult:")
        print(f"  Status: {status}")
        if energy is not None:
            print(f"  Predicted Binding Energy: {energy:.4f} eV")
        else:
            print(f"  Prediction failed")


if __name__ == "__main__":
    main()

