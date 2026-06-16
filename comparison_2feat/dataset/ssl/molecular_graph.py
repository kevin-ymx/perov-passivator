"""
Molecular graph dataset — 2-feature comparison (atomic_num + chirality only).
Supports SDF files (.sdf and .sdf.gz) and CSV files with SMILES column.
"""
import csv
import gzip
import os
from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm


def is_valid_graph(graph: Data) -> bool:
    """Check if a graph is valid for training."""
    if graph.num_nodes == 0 or graph.num_nodes < 2:
        return False
    if graph.edge_index.size(1) == 0:
        return False
    if graph.x.size(0) != graph.num_nodes:
        return False
    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        return False
    if graph.edge_attr is not None:
        if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
            return False
    return True


class MolToGraphConverter:
    """
    Converts RDKit molecules to PyTorch Geometric Data graphs.
    Node features: [atomic_num, chirality] (2D).
    Edge features: [bond_type, bond_direction] (2D).
    """

    def convert(self, mol: Chem.Mol) -> Data:
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append([
                float(atom.GetAtomicNum()),
                float(atom.GetChiralTag()),
            ])
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            fe = [float(bond.GetBondType()), float(bond.GetBondDir())]
            edge_features.extend([fe, fe])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=mol.GetNumAtoms())


class MolecularGraphDataset:
    """Dataset for constructing molecular graphs from SDF files (2-feature)."""

    def __init__(self, sdf_file: str, max_molecules: Optional[int] = None):
        self.sdf_file = sdf_file
        self.max_molecules = max_molecules
        self.molecules = []
        self._converter = MolToGraphConverter()
        self._load_molecules(max_molecules)

    def _load_molecules(self, max_molecules: Optional[int] = None) -> None:
        if not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        print(f"Loading molecules from {self.sdf_file}...")
        if max_molecules:
            print(f"  (Limited to {max_molecules:,} molecules)")
        count = 0
        if self.sdf_file.endswith('.gz'):
            with gzip.open(self.sdf_file, 'rb') as gz_file:
                supplier = Chem.ForwardSDMolSupplier(gz_file, removeHs=False)
                for mol in supplier:
                    if mol is not None:
                        self.molecules.append(mol)
                        count += 1
                        if count % 100000 == 0:
                            print(f"  Loaded {count:,} molecules...")
                        if max_molecules and count >= max_molecules:
                            print(f"  Reached limit of {max_molecules:,} molecules")
                            break
        else:
            supplier = Chem.SDMolSupplier(self.sdf_file, removeHs=False)
            for mol in tqdm(supplier, desc="Loading molecules"):
                if mol is not None:
                    self.molecules.append(mol)
                    count += 1
                    if max_molecules and count >= max_molecules:
                        break
        print(f"Loaded {len(self.molecules):,} molecules")

    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        return self._converter.convert(mol)

    def get_all_graphs(self) -> List[Data]:
        graphs = []
        for mol in tqdm(self.molecules, desc="Converting to graphs"):
            try:
                graph = self.mol_to_graph(mol)
                graphs.append(graph)
            except Exception:
                continue
        return graphs

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Data:
        return self.mol_to_graph(self.molecules[idx])


class MolecularGraphDatasetCSV:
    """Dataset for constructing molecular graphs from CSV files with SMILES (2-feature)."""

    def __init__(self, csv_file: str, max_molecules: Optional[int] = None):
        self.csv_file = csv_file
        self.max_molecules = max_molecules
        self.molecules = []
        self._converter = MolToGraphConverter()
        self._load_molecules(max_molecules)

    def _load_molecules(self, max_molecules: Optional[int] = None) -> None:
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        print(f"Loading molecules from {self.csv_file}...")
        if max_molecules:
            print(f"  (Limited to {max_molecules:,} molecules)")
        count = 0
        invalid_smiles = 0
        with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading molecules"):
                smiles = row.get('SMILES', '').strip()
                if not smiles:
                    invalid_smiles += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self.molecules.append(mol)
                    count += 1
                    if max_molecules and count >= max_molecules:
                        print(f"  Reached limit of {max_molecules:,} molecules")
                        break
                else:
                    invalid_smiles += 1
        print(f"Loaded {len(self.molecules):,} molecules")
        if invalid_smiles > 0:
            print(f"  Skipped {invalid_smiles:,} invalid/empty SMILES")

    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        return self._converter.convert(mol)

    def get_all_graphs(self) -> List[Data]:
        graphs = []
        for mol in tqdm(self.molecules, desc="Converting to graphs"):
            try:
                graph = self.mol_to_graph(mol)
                graphs.append(graph)
            except Exception:
                continue
        return graphs

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Data:
        return self.mol_to_graph(self.molecules[idx])
