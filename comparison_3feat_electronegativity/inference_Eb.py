"""
Inference script for predicting binding energy from SMILES.

Usage:
    python inference_Eb.py --smiles "CCO"
    python inference_Eb.py --csv input.csv --output predictions.csv
    python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb
    python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb --embeddings_only
"""
import os
import sys
import argparse
import csv
import glob
import torch
import numpy as np
from typing import Optional, List, Tuple
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
    Convert a molecule to a PyTorch Geometric graph (3-feature: electronegativity).
    Node features: [atomic_num, chirality, electronegativity].
    """
    ELECTRONEGATIVITY = {
        1: 2.20, 3: 0.98, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
        19: 0.82, 20: 1.00, 34: 2.55, 35: 2.96, 53: 2.66,
    }
    node_features = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        node_features.append([
            float(an),
            float(int(atom.GetChiralTag())),
            float(ELECTRONEGATIVITY.get(an, 2.0)),
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
    DEFAULT_DOWNSTREAM_CHECKPOINT = "./checkpoints/comparison_3feat_electronegativity/downstream/downstream_best_model.pt"
    DEFAULT_GINE_CHECKPOINT = "./checkpoints/comparison_3feat_electronegativity/downstream/gin_e_finetuned.pt"

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

    Args:
        embeddings_only: If True, only store GIN-E embeddings (emb_0..emb_N).
                         If False, also predict binding energy.

    Returns:
        (n_ok, n_fail) counts.
    """
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return 0, 0

    smiles_col_actual = smiles_col
    if smiles_col not in fieldnames:
        # Try common alternatives
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

    n_ok, n_fail = 0, 0
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
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


def main():
    parser = argparse.ArgumentParser(
        description='Predict binding energy from SMILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python inference_Eb.py --smiles "CCO"

  # Batch prediction from a single CSV
  python inference_Eb.py --csv input.csv --output predictions.csv

  # Process all CSVs in a folder (embeddings + predictions)
  python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb

  # Embeddings only (no binding energy prediction)
  python inference_Eb.py --filtered_csv ./filtered_csv_latest --output_dir ./filtered_csv_Eb --embeddings_only
        """
    )

    # Input options
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    parser.add_argument('--csv', type=str, help='Path to input CSV file for batch prediction')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output CSV file (default: predictions.csv)')
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
        pending = [f for f in all_csv if not os.path.exists(os.path.join(output_dir, os.path.basename(f)))]

        if args.world_size > 1:
            csv_files = [f for i, f in enumerate(pending) if i % args.world_size == args.rank]
            print(f"Rank {args.rank}/{args.world_size}: processing {len(csv_files)} of {len(pending)} pending CSV file(s)")
        else:
            csv_files = pending
            if len(pending) < len(all_csv):
                print(f"Skipping {len(all_csv) - len(pending)} already-processed file(s)")

        mode_str = "embeddings only" if args.embeddings_only else "embeddings + binding energy"
        if not csv_files:
            print(f"No CSV files to process" + (f" for rank {args.rank}" if args.world_size > 1 else ""))
        else:
            print(f"\nProcessing {len(csv_files)} CSV file(s) from {folder} ({mode_str})")
            print(f"Output directory: {output_dir}\n")
            total_ok, total_fail, skipped = 0, 0, 0
            pbar = tqdm(csv_files, desc="CSV files", unit="file")
            for csv_path in pbar:
                basename = os.path.basename(csv_path)
                out_path = os.path.join(output_dir, basename)
                if os.path.exists(out_path):
                    pbar.set_postfix_str(f"[SKIP] {basename}")
                    skipped += 1
                    continue
                pbar.set_postfix_str(basename)
                n_ok, n_fail = process_csv_file(
                    csv_path, out_path, predictor,
                    smiles_col=args.smiles_col,
                    batch_size=args.batch_size,
                    embeddings_only=args.embeddings_only,
                )
                total_ok += n_ok
                total_fail += n_fail
            processed = len(csv_files) - skipped
            print(f"\nSummary: {total_ok} processed, {total_fail} failed across {processed} file(s)"
                  + (f", {skipped} skipped (already exist)" if skipped else ""))

    elif args.csv:
        predict_from_csv(
            input_csv=args.csv,
            output_csv=args.output,
            predictor=predictor,
            smiles_col=args.smiles_col,
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

