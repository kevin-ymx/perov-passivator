"""
Config-driven molecule CSV -> raw PyTorch Geometric graph cache.

Usage:
    python mol_graph_cache.py --list-features
    python mol_graph_cache.py --write-config run_config.json
    python mol_graph_cache.py --config run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

torch = None
Chem = None
AllChem = None
Data = None

SCHEMA_VERSION = "mol-graph-cache-v1"
PLACEHOLDER_MARKERS = (
    "/REPLACE/",
    "/ABSOLUTE/PATH",
    "REPLACE_ME",
    "YOUR_PATH",
    "PLACEHOLDER",
)

ELEMENT_VOCAB = [
    "H", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "K", "Ca", "Br", "I", "Li", "Se", "other",
]
HYBRIDIZATION_VOCAB = [
    "UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP2D", "SP3D", "SP3D2", "OTHER",
]
CHIRALITY_VOCAB = [
    "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER",
]
BOND_TYPE_VOCAB = [
    "UNSPECIFIED", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "other",
]
BOND_DIRECTION_VOCAB = [
    "NONE", "BEGINWEDGE", "BEGINDASH", "ENDDOWNRIGHT", "ENDUPRIGHT",
    "EITHERDOUBLE", "UNKNOWN",
]
BOND_STEREO_VOCAB = [
    "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS",
]

ELECTRONEGATIVITY = {
    1: 2.20, 3: 0.98, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58,
    17: 3.16, 19: 0.82, 20: 1.00, 34: 2.55, 35: 2.96, 53: 2.66,
}
VALENCE_ELECTRONS = {
    1: 1, 3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 11: 1, 12: 2,
    13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 19: 1, 20: 2, 34: 6,
    35: 7, 53: 7,
}


def ensure_runtime_deps() -> None:
    """Import graph-building dependencies only when conversion actually runs."""
    global torch, Chem, AllChem, Data
    if torch is not None and Chem is not None and AllChem is not None and Data is not None:
        return
    try:
        import torch as torch_module
        from rdkit import Chem as chem_module
        from rdkit import RDLogger
        from rdkit.Chem import AllChem as all_chem_module
        from torch_geometric.data import Data as data_class
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Install the packages in requirements.txt "
            "(torch, torch-geometric, rdkit) in the Python environment used to run "
            "graph conversion."
        ) from exc
    RDLogger.DisableLog("rdApp.warning")
    torch = torch_module
    Chem = chem_module
    AllChem = all_chem_module
    Data = data_class


@dataclass
class IOConfig:
    mode: str = "single"
    input: Optional[str] = "/REPLACE/with/input.csv"
    input_dir: Optional[str] = None
    output_dir: str = "/REPLACE/with/output_graph_cache"
    cid_column: str = "CID"
    smiles_column: str = "SMILES"


@dataclass
class GraphConfig:
    include_hydrogens: bool = False
    node_features: List[str] = field(default_factory=lambda: [
        "element_onehot",
        "partial_charge",
        "hybridization_onehot",
        "degree",
        "valence_electrons",
        "electronegativity",
    ])
    edge_features: List[str] = field(default_factory=lambda: [
        "bond_type_onehot",
        "bond_direction_onehot",
    ])


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    scope: str
    dim: int
    description: str
    extractor: Callable[..., List[float]]
    vocab: Optional[List[str]] = None


def enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def clean_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def onehot(value: str, vocab: Sequence[str]) -> List[float]:
    key = value if value in vocab else ("other" if "other" in vocab else value)
    return [1.0 if item == key else 0.0 for item in vocab]


def atom_min_ring_size(atom: Chem.Atom, mol: Chem.Mol) -> float:
    idx = atom.GetIdx()
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings() if idx in ring]
    return float(min(sizes) if sizes else 0)


def get_partial_charges(mol: Chem.Mol) -> List[float]:
    try:
        AllChem.ComputeGasteigerCharges(mol)
        return [
            clean_float(atom.GetDoubleProp("_GasteigerCharge"))
            if atom.HasProp("_GasteigerCharge") else 0.0
            for atom in mol.GetAtoms()
        ]
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def node_element_onehot(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return onehot(atom.GetSymbol(), ELEMENT_VOCAB)


def node_atomic_mass(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetMass())]


def node_formal_charge(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetFormalCharge())]


def node_partial_charge(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    idx = atom.GetIdx()
    return [charges[idx] if idx < len(charges) else 0.0]


def node_degree(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetDegree())]


def node_total_degree(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalDegree())]


def node_coordination_num(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(len(atom.GetNeighbors()))]


def node_explicit_valence(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetExplicitValence())]


def node_implicit_valence(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetImplicitValence())]


def node_total_valence(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalValence())]


def node_total_num_hs(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalNumHs())]


def node_num_radical_electrons(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(atom.GetNumRadicalElectrons())]


def node_hybridization_onehot(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return onehot(enum_name(atom.GetHybridization()), HYBRIDIZATION_VOCAB)


def node_chirality_onehot(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return onehot(enum_name(atom.GetChiralTag()), CHIRALITY_VOCAB)


def node_is_aromatic(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [1.0 if atom.GetIsAromatic() else 0.0]


def node_is_in_ring(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [1.0 if atom.IsInRing() else 0.0]


def node_min_ring_size(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [atom_min_ring_size(atom, mol)]


def node_valence_electrons(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(VALENCE_ELECTRONS.get(atom.GetAtomicNum(), 4))]


def node_electronegativity(atom: Chem.Atom, mol: Chem.Mol, charges: List[float]) -> List[float]:
    return [float(ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 2.0))]


def edge_bond_type_onehot(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return onehot(enum_name(bond.GetBondType()), BOND_TYPE_VOCAB)


def edge_bond_order(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return [float(bond.GetBondTypeAsDouble())]


def edge_bond_direction_onehot(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return onehot(enum_name(bond.GetBondDir()), BOND_DIRECTION_VOCAB)


def edge_bond_stereo_onehot(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return onehot(enum_name(bond.GetStereo()), BOND_STEREO_VOCAB)


def edge_is_conjugated(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return [1.0 if bond.GetIsConjugated() else 0.0]


def edge_is_aromatic(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return [1.0 if bond.GetIsAromatic() else 0.0]


def edge_is_in_ring(bond: Chem.Bond, mol: Chem.Mol) -> List[float]:
    return [1.0 if bond.IsInRing() else 0.0]


NODE_FEATURES: Dict[str, FeatureSpec] = {
    "element_onehot": FeatureSpec("element_onehot", "node", len(ELEMENT_VOCAB), "Atom element one-hot.", node_element_onehot, ELEMENT_VOCAB),
    "atomic_mass": FeatureSpec("atomic_mass", "node", 1, "Atomic mass.", node_atomic_mass),
    "formal_charge": FeatureSpec("formal_charge", "node", 1, "Formal charge.", node_formal_charge),
    "partial_charge": FeatureSpec("partial_charge", "node", 1, "Gasteiger partial charge, fallback 0.", node_partial_charge),
    "degree": FeatureSpec("degree", "node", 1, "Explicit neighbor count.", node_degree),
    "total_degree": FeatureSpec("total_degree", "node", 1, "Total degree including implicit hydrogens.", node_total_degree),
    "coordination_num": FeatureSpec("coordination_num", "node", 1, "Number of explicit neighboring atoms.", node_coordination_num),
    "explicit_valence": FeatureSpec("explicit_valence", "node", 1, "Explicit valence.", node_explicit_valence),
    "implicit_valence": FeatureSpec("implicit_valence", "node", 1, "Implicit valence.", node_implicit_valence),
    "total_valence": FeatureSpec("total_valence", "node", 1, "Total valence.", node_total_valence),
    "total_num_hs": FeatureSpec("total_num_hs", "node", 1, "Total attached hydrogens.", node_total_num_hs),
    "num_radical_electrons": FeatureSpec("num_radical_electrons", "node", 1, "Number of radical electrons.", node_num_radical_electrons),
    "hybridization_onehot": FeatureSpec("hybridization_onehot", "node", len(HYBRIDIZATION_VOCAB), "RDKit hybridization one-hot.", node_hybridization_onehot, HYBRIDIZATION_VOCAB),
    "chirality_onehot": FeatureSpec("chirality_onehot", "node", len(CHIRALITY_VOCAB), "RDKit chiral tag one-hot.", node_chirality_onehot, CHIRALITY_VOCAB),
    "is_aromatic": FeatureSpec("is_aromatic", "node", 1, "1 if atom is aromatic.", node_is_aromatic),
    "is_in_ring": FeatureSpec("is_in_ring", "node", 1, "1 if atom is in any ring.", node_is_in_ring),
    "min_ring_size": FeatureSpec("min_ring_size", "node", 1, "Smallest containing ring size, or 0.", node_min_ring_size),
    "valence_electrons": FeatureSpec("valence_electrons", "node", 1, "Lookup valence electron count, fallback 4.", node_valence_electrons),
    "electronegativity": FeatureSpec("electronegativity", "node", 1, "Pauling electronegativity lookup, fallback 2.0.", node_electronegativity),
}

EDGE_FEATURES: Dict[str, FeatureSpec] = {
    "bond_type_onehot": FeatureSpec("bond_type_onehot", "edge", len(BOND_TYPE_VOCAB), "RDKit bond type one-hot.", edge_bond_type_onehot, BOND_TYPE_VOCAB),
    "bond_order": FeatureSpec("bond_order", "edge", 1, "Numeric bond order.", edge_bond_order),
    "bond_direction_onehot": FeatureSpec("bond_direction_onehot", "edge", len(BOND_DIRECTION_VOCAB), "RDKit bond direction one-hot.", edge_bond_direction_onehot, BOND_DIRECTION_VOCAB),
    "bond_stereo_onehot": FeatureSpec("bond_stereo_onehot", "edge", len(BOND_STEREO_VOCAB), "RDKit bond stereo one-hot.", edge_bond_stereo_onehot, BOND_STEREO_VOCAB),
    "is_conjugated": FeatureSpec("is_conjugated", "edge", 1, "1 if bond is conjugated.", edge_is_conjugated),
    "is_aromatic": FeatureSpec("is_aromatic", "edge", 1, "1 if bond is aromatic.", edge_is_aromatic),
    "is_in_ring": FeatureSpec("is_in_ring", "edge", 1, "1 if bond is in any ring.", edge_is_in_ring),
}


def default_run_config() -> RunConfig:
    return RunConfig()


def has_placeholder(value: Optional[str]) -> bool:
    return bool(value) and any(marker in value for marker in PLACEHOLDER_MARKERS)


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if "preset" in data.get("graph", {}):
        raise ValueError("graph.preset is not supported. List node_features and edge_features explicitly.")
    io = IOConfig(**{**asdict(IOConfig()), **data.get("io", {})})
    graph = GraphConfig(**{**asdict(GraphConfig()), **data.get("graph", {})})
    return RunConfig(confirmed=bool(data.get("confirmed", False)), io=io, graph=graph)


def save_run_config(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)
        f.write("\n")


def feature_dims(node_features: List[str], edge_features: List[str]) -> Dict[str, Any]:
    node = {name: NODE_FEATURES[name].dim for name in node_features}
    edge = {name: EDGE_FEATURES[name].dim for name in edge_features}
    return {
        "node": node,
        "edge": edge,
        "node_total_dim": sum(node.values()),
        "edge_total_dim": sum(edge.values()),
    }


def onehot_vocabs(node_features: List[str], edge_features: List[str]) -> Dict[str, List[str]]:
    vocabs: Dict[str, List[str]] = {}
    for name in node_features:
        vocab = NODE_FEATURES[name].vocab
        if vocab is not None:
            vocabs[name] = list(vocab)
    for name in edge_features:
        vocab = EDGE_FEATURES[name].vocab
        if vocab is not None:
            vocabs[name] = list(vocab)
    return vocabs


def validate_run_config(run: RunConfig) -> None:
    if run.io.mode not in {"single", "shards"}:
        raise ValueError("io.mode must be 'single' or 'shards'.")
    if has_placeholder(run.io.output_dir):
        raise ValueError(f"io.output_dir contains a placeholder: {run.io.output_dir}")
    if run.io.mode == "single":
        if not run.io.input:
            raise ValueError("io.input is required for mode='single'.")
        if has_placeholder(run.io.input):
            raise ValueError(f"io.input contains a placeholder: {run.io.input}")
        if not os.path.isfile(run.io.input):
            raise ValueError(f"io.input not found: {run.io.input}")
    else:
        if not run.io.input_dir:
            raise ValueError("io.input_dir is required for mode='shards'.")
        if has_placeholder(run.io.input_dir):
            raise ValueError(f"io.input_dir contains a placeholder: {run.io.input_dir}")
        if not os.path.isdir(run.io.input_dir):
            raise ValueError(f"io.input_dir not found: {run.io.input_dir}")
    if not run.graph.node_features:
        raise ValueError("graph.node_features must not be empty.")
    unknown_node = [name for name in run.graph.node_features if name not in NODE_FEATURES]
    unknown_edge = [name for name in run.graph.edge_features if name not in EDGE_FEATURES]
    if unknown_node:
        raise ValueError(f"Unknown node feature(s): {unknown_node}")
    if unknown_edge:
        raise ValueError(f"Unknown edge feature(s): {unknown_edge}")


def list_input_files(io: IOConfig) -> List[str]:
    if io.mode == "single":
        return [str(Path(io.input))]
    paths = sorted(str(p) for p in Path(io.input_dir).glob("*.csv"))
    if not paths:
        raise ValueError(f"No *.csv files found in {io.input_dir}")
    return paths


def resolve_column(fieldnames: Sequence[str], preferred: str, fallbacks: Sequence[str]) -> Optional[str]:
    exact = [preferred, *fallbacks]
    for candidate in exact:
        if candidate in fieldnames:
            return candidate
    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in exact:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def prepare_mol(smiles: str, include_hydrogens: bool) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol) if include_hydrogens else Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def build_graph(mol: Chem.Mol, graph_config: GraphConfig) -> Data:
    charges = get_partial_charges(mol)
    node_rows: List[List[float]] = []
    for atom in mol.GetAtoms():
        row: List[float] = []
        for name in graph_config.node_features:
            row.extend(NODE_FEATURES[name].extractor(atom, mol, charges))
        node_rows.append([clean_float(v) for v in row])

    node_dim = sum(NODE_FEATURES[name].dim for name in graph_config.node_features)
    edge_dim = sum(EDGE_FEATURES[name].dim for name in graph_config.edge_features)
    x = torch.tensor(node_rows, dtype=torch.float) if node_rows else torch.empty((0, node_dim), dtype=torch.float)

    edge_index_rows: List[List[int]] = []
    edge_attr_rows: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat: List[float] = []
        for name in graph_config.edge_features:
            feat.extend(EDGE_FEATURES[name].extractor(bond, mol))
        feat = [clean_float(v) for v in feat]
        edge_index_rows.extend([[i, j], [j, i]])
        edge_attr_rows.extend([feat, feat])

    if edge_index_rows:
        edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=mol.GetNumAtoms())


def is_valid_graph(graph: Data, node_dim: int, edge_dim: int) -> Tuple[bool, str]:
    if graph.num_nodes < 1:
        return False, "empty_graph"
    if graph.x is None or tuple(graph.x.shape) != (graph.num_nodes, node_dim):
        return False, "invalid_node_shape"
    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        return False, "invalid_node_value"
    if graph.edge_index is None or graph.edge_index.size(0) != 2:
        return False, "invalid_edge_index_shape"
    if graph.edge_attr is None or graph.edge_attr.size(1) != edge_dim:
        return False, "invalid_edge_attr_shape"
    if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
        return False, "invalid_edge_value"
    return True, "ok"


def failure_row(source_file: str, source_row: int, cid: str, smiles: str, reason: str) -> Dict[str, str]:
    return {
        "source_file": source_file,
        "source_row": str(source_row),
        "cid": cid,
        "smiles": smiles,
        "reason": reason,
    }


def process_csv(path: str, run: RunConfig, feature_info: Dict[str, Any]) -> Dict[str, Any]:
    input_path = Path(path)
    output_dir = Path(run.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{input_path.stem}_graphs.pt"

    graphs: List[Data] = []
    cids: List[str] = []
    smiles_values: List[str] = []
    source_rows: List[int] = []
    failures: List[Dict[str, str]] = []
    counts: Dict[str, int] = {"rows": 0, "kept": 0, "failed": 0}
    node_dim = feature_info["feature_dims"]["node_total_dim"]
    edge_dim = feature_info["feature_dims"]["edge_total_dim"]

    with open(input_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        smiles_col = resolve_column(fieldnames, run.io.smiles_column, ["SMILES", "smiles", "canonical_smiles"])
        cid_col = resolve_column(fieldnames, run.io.cid_column, ["CID", "cid", "PUBCHEM_COMPOUND_CID"])
        if smiles_col is None:
            raise ValueError(f"{path} has no SMILES column. Columns: {fieldnames}")

        for row_idx, row in enumerate(reader, start=2):
            counts["rows"] += 1
            smiles = (row.get(smiles_col) or "").strip()
            cid = (row.get(cid_col) or "").strip() if cid_col else ""
            if not smiles:
                counts["failed"] += 1
                failures.append(failure_row(path, row_idx, cid, smiles, "missing_smiles"))
                continue
            mol = prepare_mol(smiles, run.graph.include_hydrogens)
            if mol is None:
                counts["failed"] += 1
                failures.append(failure_row(path, row_idx, cid, smiles, "invalid_smiles_or_sanitize_failed"))
                continue
            try:
                graph = build_graph(mol, run.graph)
                ok, reason = is_valid_graph(graph, node_dim, edge_dim)
                if not ok:
                    counts["failed"] += 1
                    failures.append(failure_row(path, row_idx, cid, smiles, reason))
                    continue
            except Exception as exc:
                counts["failed"] += 1
                failures.append(failure_row(path, row_idx, cid, smiles, f"graph_build_failed:{type(exc).__name__}"))
                continue
            graphs.append(graph)
            cids.append(cid)
            smiles_values.append(smiles)
            source_rows.append(row_idx)
            counts["kept"] += 1

    cache = {
        "schema_version": SCHEMA_VERSION,
        "node_features": list(run.graph.node_features),
        "edge_features": list(run.graph.edge_features),
        "feature_dims": feature_info["feature_dims"],
        "onehot_vocabs": feature_info["onehot_vocabs"],
        "include_hydrogens": run.graph.include_hydrogens,
        "graphs": graphs,
        "cids": cids,
        "smiles": smiles_values,
        "source_file": str(input_path),
        "source_row_indices": source_rows,
    }
    torch.save(cache, cache_path)

    return {
        "source_file": str(input_path),
        "cache_file": str(cache_path),
        "counts": counts,
        "cids": cids,
        "smiles": smiles_values,
        "source_row_indices": source_rows,
        "failures": failures,
    }


def write_index(path: Path, results: List[Dict[str, Any]]) -> int:
    fieldnames = ["cache_file", "cache_index", "cid", "smiles", "source_file", "source_row"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            zipped = zip(result["cids"], result["smiles"], result["source_row_indices"])
            for idx, (cid, smiles, source_row) in enumerate(zipped):
                writer.writerow({
                    "cache_file": result["cache_file"],
                    "cache_index": idx,
                    "cid": cid,
                    "smiles": smiles,
                    "source_file": result["source_file"],
                    "source_row": source_row,
                })
                n += 1
    return n


def write_failures(path: Path, results: List[Dict[str, Any]]) -> int:
    fieldnames = ["source_file", "source_row", "cid", "smiles", "reason"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["failures"]:
                writer.writerow(row)
                n += 1
    return n


def write_manifest(path: Path, run: RunConfig, feature_info: Dict[str, Any], results: List[Dict[str, Any]], index_rows: int, failure_rows: int) -> None:
    files = []
    total_rows = total_kept = total_failed = 0
    for result in results:
        counts = dict(result["counts"])
        total_rows += counts["rows"]
        total_kept += counts["kept"]
        total_failed += counts["failed"]
        files.append({
            "source_file": result["source_file"],
            "cache_file": result["cache_file"],
            "counts": counts,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_config": asdict(run),
        "node_features": list(run.graph.node_features),
        "edge_features": list(run.graph.edge_features),
        "feature_dims": feature_info["feature_dims"],
        "onehot_vocabs": feature_info["onehot_vocabs"],
        "include_hydrogens": run.graph.include_hydrogens,
        "totals": {
            "files": len(results),
            "rows": total_rows,
            "kept": total_kept,
            "failed": total_failed,
            "index_rows": index_rows,
            "failure_rows": failure_rows,
        },
        "files": files,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def run_cache(run: RunConfig) -> None:
    validate_run_config(run)
    if not run.confirmed:
        raise SystemExit(
            "Run blocked: confirmed is false. Present the full config to the user, "
            "then set confirmed to true before execution."
        )
    ensure_runtime_deps()
    files = list_input_files(run.io)
    info = {
        "feature_dims": feature_dims(run.graph.node_features, run.graph.edge_features),
        "onehot_vocabs": onehot_vocabs(run.graph.node_features, run.graph.edge_features),
    }
    output_dir = Path(run.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input files: {len(files)}")
    print(f"Output dir: {output_dir}")
    print(f"Node dim: {info['feature_dims']['node_total_dim']}")
    print(f"Edge dim: {info['feature_dims']['edge_total_dim']}")
    print(f"Include hydrogens: {run.graph.include_hydrogens}")

    results = []
    for path in files:
        print(f"Processing {path} ...")
        result = process_csv(path, run, info)
        counts = result["counts"]
        print(f"  kept {counts['kept']} / {counts['rows']} rows; failed {counts['failed']}; cache {result['cache_file']}")
        results.append(result)

    index_rows = write_index(output_dir / "index.csv", results)
    failure_rows = write_failures(output_dir / "failures.csv", results)
    write_manifest(output_dir / "manifest.json", run, info, results, index_rows, failure_rows)

    print("Done.")
    print(f"  index.csv rows: {index_rows}")
    print(f"  failures.csv rows: {failure_rows}")
    print(f"  manifest: {output_dir / 'manifest.json'}")


def print_feature_catalog() -> None:
    print("Node features:")
    for name in sorted(NODE_FEATURES):
        spec = NODE_FEATURES[name]
        vocab = f" vocab={','.join(spec.vocab)}" if spec.vocab else ""
        print(f"  {name} dim={spec.dim} - {spec.description}{vocab}")
    print("\nEdge features:")
    for name in sorted(EDGE_FEATURES):
        spec = EDGE_FEATURES[name]
        vocab = f" vocab={','.join(spec.vocab)}" if spec.vocab else ""
        print(f"  {name} dim={spec.dim} - {spec.description}{vocab}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw molecule graph caches from CSV SMILES.")
    parser.add_argument("--config", help="JSON run config.")
    parser.add_argument("--write-config", metavar="PATH", help="Write default config and exit.")
    parser.add_argument("--list-features", action="store_true", help="Print selectable node and edge features.")
    parser.add_argument("--confirmed", action="store_true", help="Set confirmed=true after user approval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_features:
        print_feature_catalog()
        return
    if args.write_config:
        save_run_config(default_run_config(), args.write_config)
        print(f"Wrote config template: {args.write_config}")
        return
    if not args.config:
        raise SystemExit("Provide --config, --write-config, or --list-features.")
    try:
        run = load_run_config(args.config)
        if args.confirmed:
            run.confirmed = True
        run_cache(run)
    except ValueError as exc:
        raise SystemExit(f"Invalid config: {exc}")
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
