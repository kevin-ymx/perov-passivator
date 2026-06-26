"""
Standalone GIN-E-style encoder inference from CSV SMILES.

Usage:
    python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --write-config run_config.json
    python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --config run_config.json
    python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --config run_config.json --shard-index 0
    python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --config run_config.json --shard-assignment worker_shards.json --worker-index 0
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

PLACEHOLDER_MARKERS = (
    "/REPLACE/",
    "/ABSOLUTE/PATH",
    "REPLACE_ME",
    "YOUR_PATH",
    "PLACEHOLDER",
)

torch = None
np = None
nn = None
F = None
Chem = None
AllChem = None
Data = None
Batch = None
GINEConv = None
global_mean_pool = None
global_add_pool = None
tqdm = None
GINEEncoder = None

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


@dataclass
class IOConfig:
    mode: str = "shards"
    input: Optional[str] = "/REPLACE/with/input.csv"
    input_dir: Optional[str] = "/REPLACE/with/input_shards"
    output_dir: str = "/REPLACE/with/gine_ssl_inference"
    shard_glob: str = "*.csv"
    cid_column: str = "CID"
    smiles_column: str = "SMILES"


@dataclass
class CheckpointConfig:
    path: str = "/REPLACE/with/best_model.pt"


@dataclass
class GraphConfig:
    include_hydrogens: Any = "auto"
    node_features: Any = "auto"
    edge_features: Any = "auto"


@dataclass
class ModelConfig:
    node_feature_dim: Any = "auto"
    edge_feature_dim: Any = "auto"
    node_embedding_dim: Any = "auto"
    edge_embedding_dim: Any = "auto"
    hidden_dim: Any = "auto"
    num_gin_layers: Any = "auto"
    dropout: Any = "auto"
    use_batch_norm: Any = "auto"
    pooling: Any = "auto"


@dataclass
class InferenceConfig:
    batch_size: int = 512
    device: str = "cuda"
    num_workers: int = 0
    normalize_embeddings: bool = False
    skip_completed: bool = True


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    scope: str
    dim: int
    description: str
    extractor: Callable[..., List[float]]
    vocab: Optional[List[str]] = None


@dataclass
class ResolvedGraphSpec:
    include_hydrogens: bool
    node_features: List[str]
    edge_features: List[str]
    feature_dims: Dict[str, Any]
    onehot_vocabs: Dict[str, List[str]]
    source: str


@dataclass
class ResolvedModelSpec:
    node_feature_dim: int
    edge_feature_dim: int
    node_embedding_dim: int
    edge_embedding_dim: int
    hidden_dim: int
    num_gin_layers: int
    dropout: float
    use_batch_norm: bool
    pooling: str


def has_placeholder(value: Any) -> bool:
    return isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS)


def ensure_runtime_deps() -> None:
    global torch, np, nn, F, Chem, AllChem, Data, Batch
    global GINEConv, global_mean_pool, global_add_pool, tqdm
    if torch is not None:
        return
    try:
        import numpy as numpy_module
        import torch as torch_module
        import torch.nn as nn_module
        import torch.nn.functional as functional_module
        from rdkit import Chem as chem_module
        from rdkit import RDLogger
        from rdkit.Chem import AllChem as all_chem_module
        from torch_geometric.data import Batch as batch_class
        from torch_geometric.data import Data as data_class
        from torch_geometric.nn import GINEConv as gine_conv_class
        from torch_geometric.nn import global_add_pool as global_add_pool_function
        from torch_geometric.nn import global_mean_pool as global_mean_pool_function
        from tqdm import tqdm as tqdm_function
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependency. Install requirements.txt in the runtime "
            "environment: torch, torch-geometric, rdkit, numpy, and tqdm."
        ) from exc
    RDLogger.DisableLog("rdApp.warning")
    np = numpy_module
    torch = torch_module
    nn = nn_module
    F = functional_module
    Chem = chem_module
    AllChem = all_chem_module
    Data = data_class
    Batch = batch_class
    GINEConv = gine_conv_class
    global_mean_pool = global_mean_pool_function
    global_add_pool = global_add_pool_function
    tqdm = tqdm_function
    install_standalone_encoder()


def install_standalone_encoder() -> None:
    global GINEEncoder

    class NodeFeatureEncoder(nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )

        def forward(self, x):
            return self.encoder(x)

    class EdgeFeatureEncoder(nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )

        def forward(self, edge_attr):
            return self.encoder(edge_attr)

    class LocalGINEEncoder(nn.Module):
        def __init__(
            self,
            node_feature_dim: int,
            edge_feature_dim: int,
            node_embedding_dim: int = 128,
            edge_embedding_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 6,
            dropout: float = 0.1,
            use_batch_norm: bool = True,
            pooling: str = "mean",
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.pooling = pooling
            self.node_encoder = NodeFeatureEncoder(node_feature_dim, node_embedding_dim)
            self.edge_encoder = EdgeFeatureEncoder(edge_feature_dim, edge_embedding_dim)
            self.gin_layers = nn.ModuleList()
            self.gin_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(node_embedding_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                        nn.ReLU(),
                    ),
                    edge_dim=edge_embedding_dim,
                    train_eps=True,
                )
            )
            for _ in range(num_layers - 2):
                self.gin_layers.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                            nn.ReLU(),
                        ),
                        edge_dim=edge_embedding_dim,
                        train_eps=True,
                    )
                )
            if num_layers > 1:
                self.gin_layers.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                            nn.ReLU(),
                        ),
                        edge_dim=edge_embedding_dim,
                        train_eps=True,
                    )
                )
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, x, edge_index, edge_attr, batch=None):
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            for i, gin_layer in enumerate(self.gin_layers):
                x = gin_layer(x, edge_index, edge_attr)
                if i < len(self.gin_layers) - 1:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.final_projection(x)
            if batch is not None:
                if self.pooling == "mean":
                    return global_mean_pool(x, batch)
                return global_add_pool(x, batch)
            if self.pooling == "mean":
                return x.mean(dim=0, keepdim=True)
            return x.sum(dim=0, keepdim=True)

    GINEEncoder = LocalGINEEncoder


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


def atom_min_ring_size(atom: Any, mol: Any) -> float:
    idx = atom.GetIdx()
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings() if idx in ring]
    return float(min(sizes) if sizes else 0)


def get_partial_charges(mol: Any) -> List[float]:
    try:
        AllChem.ComputeGasteigerCharges(mol)
        return [
            clean_float(atom.GetDoubleProp("_GasteigerCharge"))
            if atom.HasProp("_GasteigerCharge") else 0.0
            for atom in mol.GetAtoms()
        ]
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def node_element_onehot(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return onehot(atom.GetSymbol(), ELEMENT_VOCAB)


def node_atomic_mass(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetMass())]


def node_formal_charge(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetFormalCharge())]


def node_partial_charge(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    idx = atom.GetIdx()
    return [charges[idx] if idx < len(charges) else 0.0]


def node_degree(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetDegree())]


def node_total_degree(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalDegree())]


def node_coordination_num(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(len(atom.GetNeighbors()))]


def node_explicit_valence(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetExplicitValence())]


def node_implicit_valence(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetImplicitValence())]


def node_total_valence(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalValence())]


def node_total_num_hs(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetTotalNumHs())]


def node_num_radical_electrons(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(atom.GetNumRadicalElectrons())]


def node_hybridization_onehot(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return onehot(enum_name(atom.GetHybridization()), HYBRIDIZATION_VOCAB)


def node_chirality_onehot(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return onehot(enum_name(atom.GetChiralTag()), CHIRALITY_VOCAB)


def node_is_aromatic(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [1.0 if atom.GetIsAromatic() else 0.0]


def node_is_in_ring(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [1.0 if atom.IsInRing() else 0.0]


def node_min_ring_size(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [atom_min_ring_size(atom, mol)]


def node_valence_electrons(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(VALENCE_ELECTRONS.get(atom.GetAtomicNum(), 4))]


def node_electronegativity(atom: Any, mol: Any, charges: List[float]) -> List[float]:
    return [float(ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 2.0))]


def edge_bond_type_onehot(bond: Any, mol: Any) -> List[float]:
    return onehot(enum_name(bond.GetBondType()), BOND_TYPE_VOCAB)


def edge_bond_order(bond: Any, mol: Any) -> List[float]:
    return [float(bond.GetBondTypeAsDouble())]


def edge_bond_direction_onehot(bond: Any, mol: Any) -> List[float]:
    return onehot(enum_name(bond.GetBondDir()), BOND_DIRECTION_VOCAB)


def edge_bond_stereo_onehot(bond: Any, mol: Any) -> List[float]:
    return onehot(enum_name(bond.GetStereo()), BOND_STEREO_VOCAB)


def edge_is_conjugated(bond: Any, mol: Any) -> List[float]:
    return [1.0 if bond.GetIsConjugated() else 0.0]


def edge_is_aromatic(bond: Any, mol: Any) -> List[float]:
    return [1.0 if bond.GetIsAromatic() else 0.0]


def edge_is_in_ring(bond: Any, mol: Any) -> List[float]:
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


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return RunConfig(
        confirmed=bool(data.get("confirmed", False)),
        io=IOConfig(**{**asdict(IOConfig()), **data.get("io", {})}),
        checkpoint=CheckpointConfig(**{**asdict(CheckpointConfig()), **data.get("checkpoint", {})}),
        graph=GraphConfig(**{**asdict(GraphConfig()), **data.get("graph", {})}),
        model=ModelConfig(**{**asdict(ModelConfig()), **data.get("model", {})}),
        inference=InferenceConfig(**{**asdict(InferenceConfig()), **data.get("inference", {})}),
    )


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


def validate_feature_names(node_features: List[str], edge_features: List[str]) -> None:
    unknown_node = [name for name in node_features if name not in NODE_FEATURES]
    unknown_edge = [name for name in edge_features if name not in EDGE_FEATURES]
    if unknown_node:
        raise ValueError(f"Unknown node feature(s): {unknown_node}")
    if unknown_edge:
        raise ValueError(f"Unknown edge feature(s): {unknown_edge}")


def validate_config(run: RunConfig) -> None:
    if run.io.mode not in {"single", "shards"}:
        raise ValueError("io.mode must be 'single' or 'shards'.")
    for label, value in (
        ("io.output_dir", run.io.output_dir),
        ("checkpoint.path", run.checkpoint.path),
    ):
        if not value:
            raise ValueError(f"{label} is required.")
        if has_placeholder(value):
            raise ValueError(f"{label} contains a placeholder: {value}")
    if not os.path.isfile(run.checkpoint.path):
        raise ValueError(f"checkpoint.path not found: {run.checkpoint.path}")
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
    if run.inference.batch_size < 1:
        raise ValueError("inference.batch_size must be >= 1.")
    if run.inference.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("inference.device must be 'auto', 'cuda', or 'cpu'.")
    if run.graph.node_features != "auto" or run.graph.edge_features != "auto":
        if not isinstance(run.graph.node_features, list) or not isinstance(run.graph.edge_features, list):
            raise ValueError("graph.node_features and graph.edge_features must both be 'auto' or lists.")
        validate_feature_names(run.graph.node_features, run.graph.edge_features)


def list_input_files(io: IOConfig) -> List[str]:
    if io.mode == "single":
        return [str(Path(io.input))]
    paths = sorted(str(p) for p in Path(io.input_dir).glob(io.shard_glob))
    if not paths:
        raise ValueError(f"No input shards match {Path(io.input_dir) / io.shard_glob}")
    return paths


def load_assigned_shards(path: str) -> List[str]:
    if has_placeholder(path) or not os.path.isfile(path):
        raise ValueError(f"shard_assignment not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        assignment = json.load(f)
    shards = assignment.get("pending_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"shard_assignment has no pending_shards: {path}")
    missing = [shard for shard in shards if not os.path.isfile(shard)]
    if missing:
        raise ValueError(f"shard_assignment references missing shard file(s): {missing[:5]}")
    return [str(Path(shard)) for shard in shards]


def load_worker_shards(path: str, worker_index: int) -> List[str]:
    if has_placeholder(path) or not os.path.isfile(path):
        raise ValueError(f"shard_assignment not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        assignment = json.load(f)
    workers = assignment.get("workers")
    if not isinstance(workers, list):
        raise ValueError(f"shard_assignment has no workers list: {path}")
    for worker in workers:
        if int(worker.get("worker_index", -1)) == worker_index:
            shards = worker.get("shards", [])
            if not isinstance(shards, list):
                raise ValueError(f"worker {worker_index} shards is not a list.")
            missing = [shard for shard in shards if not os.path.isfile(shard)]
            if missing:
                raise ValueError(f"worker {worker_index} references missing shard file(s): {missing[:5]}")
            return [str(Path(shard)) for shard in shards]
    raise ValueError(f"worker_index {worker_index} not found in shard_assignment.")


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


def prepare_mol(smiles: str, include_hydrogens: bool) -> Optional[Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol) if include_hydrogens else Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def build_graph(mol: Any, graph_spec: ResolvedGraphSpec) -> Any:
    charges = get_partial_charges(mol)
    node_rows: List[List[float]] = []
    for atom in mol.GetAtoms():
        row: List[float] = []
        for name in graph_spec.node_features:
            row.extend(NODE_FEATURES[name].extractor(atom, mol, charges))
        node_rows.append([clean_float(v) for v in row])

    node_dim = graph_spec.feature_dims["node_total_dim"]
    edge_dim = graph_spec.feature_dims["edge_total_dim"]
    x = torch.tensor(node_rows, dtype=torch.float) if node_rows else torch.empty((0, node_dim), dtype=torch.float)

    edge_index_rows: List[List[int]] = []
    edge_attr_rows: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat: List[float] = []
        for name in graph_spec.edge_features:
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


def is_valid_graph(graph: Any, node_dim: int, edge_dim: int) -> Tuple[bool, str]:
    if graph.num_nodes < 1:
        return False, "empty_graph"
    if graph.x is None or graph.x.size(0) != graph.num_nodes or graph.x.size(1) != node_dim:
        return False, "bad_node_feature_shape"
    if graph.edge_index is None or graph.edge_index.size(0) != 2:
        return False, "bad_edge_index_shape"
    if graph.edge_attr is None or graph.edge_attr.size(1) != edge_dim:
        return False, "bad_edge_feature_shape"
    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        return False, "nonfinite_node_features"
    if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
        return False, "nonfinite_edge_features"
    return True, "ok"


def choose_device(name: str) -> Any:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("inference.device is 'cuda' but CUDA is not available.")
    return torch.device(name)


def torch_load(path: str, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "encoder_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def checkpoint_feature_metadata(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("feature_metadata"), dict):
        return checkpoint["feature_metadata"]
    return {}


def checkpoint_model_config(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    run_config = checkpoint.get("run_config")
    if isinstance(run_config, dict) and isinstance(run_config.get("model"), dict):
        return run_config["model"]
    if isinstance(checkpoint.get("model_config"), dict):
        return checkpoint["model_config"]
    return {}


def value_or_auto(config_value: Any, fallback: Any, label: str) -> Any:
    if config_value != "auto":
        return config_value
    if fallback is not None:
        return fallback
    raise ValueError(f"{label} is 'auto' but could not be resolved from checkpoint.")


def infer_dims_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    inferred: Dict[str, Any] = {}
    node_w = state_dict.get("node_encoder.encoder.0.weight")
    edge_w = state_dict.get("edge_encoder.encoder.0.weight")
    if node_w is not None:
        inferred["node_embedding_dim"] = int(node_w.size(0))
        inferred["node_feature_dim"] = int(node_w.size(1))
    if edge_w is not None:
        inferred["edge_embedding_dim"] = int(edge_w.size(0))
        inferred["edge_feature_dim"] = int(edge_w.size(1))
    final_w = state_dict.get("final_projection.0.weight")
    if final_w is not None:
        inferred["hidden_dim"] = int(final_w.size(0))
    layer_indices = set()
    for key in state_dict:
        if key.startswith("gin_layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if layer_indices:
        inferred["num_gin_layers"] = max(layer_indices) + 1
    return inferred


def resolve_graph_spec(run: RunConfig, feature_meta: Dict[str, Any]) -> ResolvedGraphSpec:
    meta_node = feature_meta.get("node_features")
    meta_edge = feature_meta.get("edge_features")
    meta_h = feature_meta.get("include_hydrogens")

    if run.graph.node_features == "auto":
        if not meta_node:
            raise ValueError("graph.node_features is 'auto' but checkpoint has no node feature metadata.")
        node_features = list(meta_node)
        source = "checkpoint"
    else:
        node_features = list(run.graph.node_features)
        source = "config"
        if meta_node and node_features != list(meta_node):
            raise ValueError("Configured node_features do not match checkpoint feature metadata.")

    if run.graph.edge_features == "auto":
        if meta_edge is None:
            raise ValueError("graph.edge_features is 'auto' but checkpoint has no edge feature metadata.")
        edge_features = list(meta_edge)
        source = "checkpoint"
    else:
        edge_features = list(run.graph.edge_features)
        if meta_edge is not None and edge_features != list(meta_edge):
            raise ValueError("Configured edge_features do not match checkpoint feature metadata.")

    if run.graph.include_hydrogens == "auto":
        if meta_h is None:
            raise ValueError("graph.include_hydrogens is 'auto' but checkpoint has no include_hydrogens metadata.")
        include_hydrogens = bool(meta_h)
    else:
        include_hydrogens = bool(run.graph.include_hydrogens)
        if meta_h is not None and include_hydrogens != bool(meta_h):
            raise ValueError("Configured include_hydrogens does not match checkpoint metadata.")

    validate_feature_names(node_features, edge_features)
    dims = feature_dims(node_features, edge_features)
    meta_dims = feature_meta.get("feature_dims")
    if isinstance(meta_dims, dict):
        for key in ("node_total_dim", "edge_total_dim"):
            if key in meta_dims and int(meta_dims[key]) != int(dims[key]):
                raise ValueError(f"Resolved {key} does not match checkpoint feature metadata.")
    return ResolvedGraphSpec(
        include_hydrogens=include_hydrogens,
        node_features=node_features,
        edge_features=edge_features,
        feature_dims=dims,
        onehot_vocabs=onehot_vocabs(node_features, edge_features),
        source=source,
    )


def resolve_model_spec(
    run: RunConfig,
    checkpoint: Any,
    state_dict: Dict[str, Any],
    graph_spec: ResolvedGraphSpec,
) -> ResolvedModelSpec:
    model_meta = checkpoint_model_config(checkpoint)
    state_meta = infer_dims_from_state_dict(state_dict)
    feature_meta = checkpoint_feature_metadata(checkpoint)

    fallback_node_dim = (
        feature_meta.get("resolved_node_feature_dim")
        or feature_meta.get("node_feature_dim")
        or graph_spec.feature_dims["node_total_dim"]
        or state_meta.get("node_feature_dim")
    )
    fallback_edge_dim = (
        feature_meta.get("resolved_edge_feature_dim")
        or feature_meta.get("edge_feature_dim")
        or graph_spec.feature_dims["edge_total_dim"]
        or state_meta.get("edge_feature_dim")
    )
    spec = ResolvedModelSpec(
        node_feature_dim=int(value_or_auto(run.model.node_feature_dim, fallback_node_dim, "model.node_feature_dim")),
        edge_feature_dim=int(value_or_auto(run.model.edge_feature_dim, fallback_edge_dim, "model.edge_feature_dim")),
        node_embedding_dim=int(value_or_auto(run.model.node_embedding_dim, model_meta.get("node_embedding_dim", state_meta.get("node_embedding_dim", 128)), "model.node_embedding_dim")),
        edge_embedding_dim=int(value_or_auto(run.model.edge_embedding_dim, model_meta.get("edge_embedding_dim", state_meta.get("edge_embedding_dim", 64)), "model.edge_embedding_dim")),
        hidden_dim=int(value_or_auto(run.model.hidden_dim, model_meta.get("hidden_dim", state_meta.get("hidden_dim", 256)), "model.hidden_dim")),
        num_gin_layers=int(value_or_auto(run.model.num_gin_layers, model_meta.get("num_gin_layers", state_meta.get("num_gin_layers", 6)), "model.num_gin_layers")),
        dropout=float(value_or_auto(run.model.dropout, model_meta.get("dropout", 0.1), "model.dropout")),
        use_batch_norm=bool(value_or_auto(run.model.use_batch_norm, model_meta.get("use_batch_norm", True), "model.use_batch_norm")),
        pooling=str(value_or_auto(run.model.pooling, model_meta.get("pooling", "mean"), "model.pooling")),
    )
    if spec.node_feature_dim != graph_spec.feature_dims["node_total_dim"]:
        raise ValueError("Resolved model.node_feature_dim does not match graph node feature dimension.")
    if spec.edge_feature_dim != graph_spec.feature_dims["edge_total_dim"]:
        raise ValueError("Resolved model.edge_feature_dim does not match graph edge feature dimension.")
    if spec.pooling not in {"mean", "add"}:
        raise ValueError("model.pooling must be 'mean' or 'add'.")
    return spec


def build_model(spec: ResolvedModelSpec, state_dict: Dict[str, Any], device: Any) -> Any:
    model = GINEEncoder(
        node_feature_dim=spec.node_feature_dim,
        edge_feature_dim=spec.edge_feature_dim,
        node_embedding_dim=spec.node_embedding_dim,
        edge_embedding_dim=spec.edge_embedding_dim,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_gin_layers,
        dropout=spec.dropout,
        use_batch_norm=spec.use_batch_norm,
        pooling=spec.pooling,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def count_csv_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def output_paths(input_path: str, output_dir: str) -> Dict[str, str]:
    stem = Path(input_path).stem
    return {
        "embeddings": str(Path(output_dir) / f"{stem}_embeddings.csv"),
        "failures": str(Path(output_dir) / f"{stem}_failures.csv"),
        "manifest": str(Path(output_dir) / f"{stem}_manifest.json"),
        "done": str(Path(output_dir) / f"{stem}_done.json"),
    }


def count_output_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def shard_completed(input_path: str, paths: Dict[str, str]) -> bool:
    if not os.path.isfile(paths["done"]) or not os.path.isfile(paths["embeddings"]):
        return False
    try:
        with open(paths["done"], "r", encoding="utf-8") as f:
            done = json.load(f)
    except Exception:
        return False
    if not done.get("completed"):
        return False
    return count_output_rows(paths["embeddings"]) == count_csv_rows(input_path)


def read_input_rows(path: str, run: RunConfig) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        smiles_col = resolve_column(fieldnames, run.io.smiles_column, ["SMILES", "smiles", "canonical_smiles"])
        cid_col = resolve_column(fieldnames, run.io.cid_column, ["CID", "cid", "PUBCHEM_COMPOUND_CID"])
        if smiles_col is None:
            raise ValueError(f"SMILES column not found in {path}")
        rows = list(reader)
    return rows, smiles_col, cid_col


def base_output_row(input_path: str, row_index: int, cid: str, smiles: str, status: str, emb_dim: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "cid": cid,
        "smiles": smiles,
        "source_file": input_path,
        "source_row": row_index,
        "status": status,
    }
    for i in range(emb_dim):
        row[f"emb_{i}"] = ""
    return row


def write_csv_atomic(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_json(path: str, data: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def process_file(
    input_path: str,
    run: RunConfig,
    graph_spec: ResolvedGraphSpec,
    model_spec: ResolvedModelSpec,
    model: Any,
    device: Any,
) -> Dict[str, Any]:
    paths = output_paths(input_path, run.io.output_dir)
    if run.inference.skip_completed and shard_completed(input_path, paths):
        print(f"Skipping completed shard: {input_path}")
        return {"input_file": input_path, "skipped": True, "paths": paths}

    rows, smiles_col, cid_col = read_input_rows(input_path, run)
    emb_dim = model_spec.hidden_dim
    output_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    valid_graphs: List[Any] = []
    valid_positions: List[int] = []

    node_dim = graph_spec.feature_dims["node_total_dim"]
    edge_dim = graph_spec.feature_dims["edge_total_dim"]

    for row_num, row in enumerate(rows, start=2):
        smiles = (row.get(smiles_col) or "").strip()
        cid = (row.get(cid_col) or "").strip() if cid_col else ""
        out_row = base_output_row(input_path, row_num, cid, smiles, "pending", emb_dim)
        output_rows.append(out_row)
        if not smiles:
            out_row["status"] = "missing_smiles"
            failures.append({"source_file": input_path, "source_row": row_num, "cid": cid, "smiles": smiles, "reason": "missing_smiles"})
            continue
        mol = prepare_mol(smiles, graph_spec.include_hydrogens)
        if mol is None:
            out_row["status"] = "invalid_smiles_or_sanitize_failed"
            failures.append({"source_file": input_path, "source_row": row_num, "cid": cid, "smiles": smiles, "reason": out_row["status"]})
            continue
        try:
            graph = build_graph(mol, graph_spec)
            ok, reason = is_valid_graph(graph, node_dim, edge_dim)
        except Exception as exc:
            ok, reason, graph = False, f"graph_build_failed:{type(exc).__name__}", None
        if not ok:
            out_row["status"] = reason
            failures.append({"source_file": input_path, "source_row": row_num, "cid": cid, "smiles": smiles, "reason": reason})
            continue
        valid_positions.append(len(output_rows) - 1)
        valid_graphs.append(graph)

    batch_size = run.inference.batch_size
    with torch.no_grad():
        iterator = range(0, len(valid_graphs), batch_size)
        for start in tqdm(iterator, desc=f"infer {Path(input_path).name}", dynamic_ncols=True):
            batch_graphs = valid_graphs[start:start + batch_size]
            batch_positions = valid_positions[start:start + batch_size]
            try:
                batch = Batch.from_data_list(batch_graphs).to(device)
                embeddings = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                if run.inference.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.detach().cpu().numpy()
                for local_idx, pos in enumerate(batch_positions):
                    output_rows[pos]["status"] = "ok"
                    for dim_idx, value in enumerate(embeddings[local_idx].tolist()):
                        output_rows[pos][f"emb_{dim_idx}"] = float(value)
            except Exception as exc:
                reason = f"inference_failed:{type(exc).__name__}"
                for pos in batch_positions:
                    output_rows[pos]["status"] = reason
                    failures.append({
                        "source_file": input_path,
                        "source_row": output_rows[pos]["source_row"],
                        "cid": output_rows[pos]["cid"],
                        "smiles": output_rows[pos]["smiles"],
                        "reason": reason,
                    })

    fieldnames = ["cid", "smiles", "source_file", "source_row", "status"] + [f"emb_{i}" for i in range(emb_dim)]
    failure_fields = ["source_file", "source_row", "cid", "smiles", "reason"]
    write_csv_atomic(paths["embeddings"], fieldnames, output_rows)
    write_csv_atomic(paths["failures"], failure_fields, failures)

    counts = {
        "input_rows": len(rows),
        "ok": sum(1 for row in output_rows if row["status"] == "ok"),
        "failed": sum(1 for row in output_rows if row["status"] != "ok"),
    }
    manifest = {
        "input_file": input_path,
        "output_paths": paths,
        "counts": counts,
        "checkpoint": run.checkpoint.path,
        "graph": asdict(graph_spec),
        "model": asdict(model_spec),
        "inference": asdict(run.inference),
        "cid_column": cid_col,
        "smiles_column": smiles_col,
    }
    write_json(paths["manifest"], manifest)
    write_json(paths["done"], {"completed": True, "input_file": input_path, "counts": counts, "embeddings": paths["embeddings"]})
    print(f"Finished {input_path}: {counts['ok']} ok, {counts['failed']} failed")
    return {"input_file": input_path, "skipped": False, "paths": paths, "counts": counts}


def run_inference(
    run: RunConfig,
    shard_index: Optional[int],
    shard_assignment: Optional[str],
    worker_index: Optional[int],
) -> None:
    validate_config(run)
    if not run.confirmed:
        raise SystemExit("Run blocked: confirmed is false. Approve the full config before execution.")

    ensure_runtime_deps()
    device = choose_device(run.inference.device)
    print(f"Using device: {device}")
    os.makedirs(run.io.output_dir, exist_ok=True)

    checkpoint = torch_load(run.checkpoint.path, map_location="cpu")
    state_dict = get_state_dict(checkpoint)
    feature_meta = checkpoint_feature_metadata(checkpoint)
    graph_spec = resolve_graph_spec(run, feature_meta)
    model_spec = resolve_model_spec(run, checkpoint, state_dict, graph_spec)
    model = build_model(model_spec, state_dict, device)

    if worker_index is not None:
        if not shard_assignment:
            raise ValueError("--worker-index requires --shard-assignment.")
        files = load_worker_shards(shard_assignment, worker_index)
        print(f"Worker {worker_index} assigned {len(files)} shard(s).")
    else:
        files = load_assigned_shards(shard_assignment) if shard_assignment else list_input_files(run.io)
    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(files):
            raise ValueError(f"shard_index {shard_index} outside available shard range 0-{len(files) - 1}.")
        files = [files[shard_index]]

    results = []
    for input_path in files:
        results.append(process_file(input_path, run, graph_spec, model_spec, model, device))

    summary = {
        "processed_files": len(results),
        "results": results,
        "shard_index": shard_index,
        "shard_assignment": shard_assignment,
        "worker_index": worker_index,
    }
    print(json.dumps(summary, indent=2))


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
    parser = argparse.ArgumentParser(description="Run GIN-E-style encoder inference from CSV SMILES.")
    parser.add_argument("--config", help="JSON inference config.")
    parser.add_argument("--write-config", metavar="PATH", help="Write config template and exit.")
    parser.add_argument("--list-features", action="store_true", help="Print selectable graph features.")
    parser.add_argument("--confirmed", action="store_true", help="Set confirmed=true after user approval.")
    parser.add_argument("--shard-index", type=int, help="Process only this sorted shard index.")
    parser.add_argument("--shard-assignment", help="JSON assignment with pending_shards and per-worker shard lists.")
    parser.add_argument("--worker-index", type=int, help="Process shards assigned to this GPU worker index.")
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
        run_inference(run, args.shard_index, args.shard_assignment, args.worker_index)
    except ValueError as exc:
        raise SystemExit(f"Invalid config: {exc}")
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
