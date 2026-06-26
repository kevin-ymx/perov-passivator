"""
Train GIN-E SSL from raw PyTorch Geometric graph caches.

Usage:
    python skills/gine-ssl-train/scripts/gine_ssl_train.py --write-config run_config.json
    python skills/gine-ssl-train/scripts/gine_ssl_train.py --config run_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
DataLoader = None
Dataset = None
Batch = None
GINEConv = None
global_mean_pool = None
global_add_pool = None
GINEEncoder = None
SubgraphRemovalAugmentation = None
NTXentLoss = None
Adam = None
CosineAnnealingLR = None
tqdm = None


@dataclass
class IOConfig:
    cache_dir: str = "/REPLACE/with/raw_graph_cache_output"


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    seed: int = 42


@dataclass
class AugmentationConfig:
    subgraph_removal_ratio: float = 0.25
    mask_value: float = 0.0


@dataclass
class ModelConfig:
    node_feature_dim: Any = "auto"
    edge_feature_dim: Any = "auto"
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 6
    dropout: float = 0.1
    use_batch_norm: bool = True
    pooling: str = "mean"


@dataclass
class TrainingConfig:
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    temperature: float = 0.07
    eta_min: float = 0.000001
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
    num_workers: int = 32
    checkpoint_frequency: int = 3
    resume_checkpoint: Optional[str] = None


@dataclass
class OutputConfig:
    checkpoint_dir: str = "/REPLACE/with/checkpoints/gine_ssl_train"
    log_dir: str = "/REPLACE/with/logs/gine_ssl_train"
    save_best: bool = True
    save_periodic: bool = True


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class FixedPairDataset:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def has_placeholder(value: Optional[str]) -> bool:
    return bool(value) and any(marker in str(value) for marker in PLACEHOLDER_MARKERS)


def default_run_config() -> RunConfig:
    return RunConfig()


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    io_fields = set(IOConfig.__dataclass_fields__.keys())
    io_data = {k: v for k, v in data.get("io", {}).items() if k in io_fields}
    return RunConfig(
        confirmed=bool(data.get("confirmed", False)),
        io=IOConfig(**{**asdict(IOConfig()), **io_data}),
        split=SplitConfig(**{**asdict(SplitConfig()), **data.get("split", {})}),
        augmentation=AugmentationConfig(**{**asdict(AugmentationConfig()), **data.get("augmentation", {})}),
        model=ModelConfig(**{**asdict(ModelConfig()), **data.get("model", {})}),
        training=TrainingConfig(**{**asdict(TrainingConfig()), **data.get("training", {})}),
        output=OutputConfig(**{**asdict(OutputConfig()), **data.get("output", {})}),
    )


def save_run_config(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)
        f.write("\n")


def validate_config(run: RunConfig) -> None:
    for label, value in (
        ("io.cache_dir", run.io.cache_dir),
        ("output.checkpoint_dir", run.output.checkpoint_dir),
        ("output.log_dir", run.output.log_dir),
    ):
        if not value:
            raise ValueError(f"{label} is required.")
        if has_placeholder(value):
            raise ValueError(f"{label} contains a placeholder: {value}")
    if not os.path.isdir(run.io.cache_dir):
        raise ValueError(f"io.cache_dir not found: {run.io.cache_dir}")
    if abs(run.split.train_ratio + run.split.val_ratio - 1.0) > 1e-6:
        raise ValueError("split.train_ratio + split.val_ratio must equal 1.0.")
    if run.split.train_ratio <= 0 or run.split.val_ratio <= 0:
        raise ValueError("split.train_ratio and split.val_ratio must both be > 0.")
    if not 0.0 <= run.augmentation.subgraph_removal_ratio < 1.0:
        raise ValueError("augmentation.subgraph_removal_ratio must be in [0.0, 1.0).")
    if run.model.pooling not in {"mean", "add"}:
        raise ValueError("model.pooling must be 'mean' or 'add'.")
    if run.training.batch_size < 1:
        raise ValueError("training.batch_size must be >= 1.")
    if run.training.num_epochs < 1:
        raise ValueError("training.num_epochs must be >= 1.")
    if run.training.checkpoint_frequency < 1:
        raise ValueError("training.checkpoint_frequency must be >= 1.")
    if run.training.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("training.device must be 'auto', 'cuda', or 'cpu'.")
    if run.training.resume_checkpoint and has_placeholder(run.training.resume_checkpoint):
        raise ValueError("training.resume_checkpoint contains a placeholder.")


def ensure_runtime_deps() -> None:
    global torch, np, nn, F, DataLoader, Dataset, Batch
    global GINEConv, global_mean_pool, global_add_pool
    global GINEEncoder, SubgraphRemovalAugmentation, NTXentLoss
    global Adam, CosineAnnealingLR, tqdm

    if torch is not None:
        return
    try:
        import numpy as numpy_module
        import torch as torch_module
        import torch.nn as nn_module
        import torch.nn.functional as functional_module
        from torch.optim import Adam as adam_class
        from torch.optim.lr_scheduler import CosineAnnealingLR as cosine_class
        from torch.utils.data import DataLoader as data_loader_class
        from torch.utils.data import Dataset as dataset_class
        from torch_geometric.data import Batch as batch_class
        from torch_geometric.nn import GINEConv as gine_conv_class
        from torch_geometric.nn import global_add_pool as global_add_pool_function
        from torch_geometric.nn import global_mean_pool as global_mean_pool_function
        from tqdm import tqdm as tqdm_function
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependency. Install requirements.txt in the Python "
            "environment used for training: torch, torch-geometric, numpy, and tqdm."
        ) from exc

    np = numpy_module
    torch = torch_module
    nn = nn_module
    F = functional_module
    Adam = adam_class
    CosineAnnealingLR = cosine_class
    DataLoader = data_loader_class
    Dataset = dataset_class
    Batch = batch_class
    GINEConv = gine_conv_class
    global_mean_pool = global_mean_pool_function
    global_add_pool = global_add_pool_function
    tqdm = tqdm_function
    install_standalone_components()


def install_standalone_components() -> None:
    """Define model, augmentation, and loss locally after deps are imported."""
    global GINEEncoder, SubgraphRemovalAugmentation, NTXentLoss

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
            node_feature_dim: int = 8,
            edge_feature_dim: int = 2,
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

    class LocalSubgraphRemovalAugmentation:
        def __init__(self, removal_ratio: float = 0.25, seed: Optional[int] = None, mask_value: float = 0.0):
            if not 0.0 <= removal_ratio < 1.0:
                raise ValueError(f"removal_ratio must be in [0.0, 1.0), got {removal_ratio}")
            self.removal_ratio = removal_ratio
            self.seed = seed
            self.mask_value = mask_value
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

        def _find_connected_subgraph(self, edge_index, num_nodes: int, target_size: int) -> Set[int]:
            adj_list = [[] for _ in range(num_nodes)]
            edge_index_np = edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                src, dst = edge_index_np[0, i], edge_index_np[1, i]
                adj_list[src].append(dst)
                adj_list[dst].append(src)
            start_node = random.randint(0, num_nodes - 1)
            visited = {start_node}
            queue = deque([start_node])
            while queue and len(visited) < target_size:
                current = queue.popleft()
                for neighbor in adj_list[current]:
                    if neighbor not in visited and len(visited) < target_size:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return visited

        def augment(self, graph):
            num_nodes = graph.num_nodes
            if num_nodes == 0 or graph.edge_index.size(1) == 0 or num_nodes <= 2:
                return graph
            num_to_remove = max(1, int(num_nodes * self.removal_ratio))
            num_to_remove = min(num_to_remove, num_nodes - 2)
            nodes_to_mask = self._find_connected_subgraph(graph.edge_index, num_nodes, num_to_remove) if num_to_remove > 0 else set()
            x = graph.x.clone()
            mask_vector = torch.full((x.shape[1],), self.mask_value, dtype=torch.float, device=x.device)
            for node_idx in nodes_to_mask:
                x[node_idx] = mask_vector
            edge_index = graph.edge_index.clone()
            edge_attr = graph.edge_attr.clone() if graph.edge_attr is not None else None
            if nodes_to_mask:
                edge_index_np = edge_index.cpu().numpy()
                valid_edges = [
                    i for i in range(edge_index_np.shape[1])
                    if edge_index_np[0, i] not in nodes_to_mask and edge_index_np[1, i] not in nodes_to_mask
                ]
                if valid_edges:
                    valid_edges_tensor = torch.tensor(valid_edges, dtype=torch.long, device=edge_index.device)
                    edge_index = edge_index[:, valid_edges_tensor]
                    if edge_attr is not None:
                        edge_attr = edge_attr[valid_edges_tensor]
                elif graph.edge_index.size(1) > 0:
                    edge_index = graph.edge_index[:, 0:1]
                    if edge_attr is not None:
                        edge_attr = graph.edge_attr[0:1]
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=graph.edge_index.device)
                    if edge_attr is not None:
                        edge_attr = torch.empty((0, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
            return graph.__class__(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

        def create_pair(self, graph):
            return self.augment(graph), self.augment(graph)

        def __call__(self, graph):
            return self.create_pair(graph)

    class LocalNTXentLoss(nn.Module):
        def __init__(self, temperature: float = 0.07):
            super().__init__()
            self.temperature = temperature

        def forward(self, z1, z2):
            batch_size = z1.size(0)
            if batch_size == 0:
                return torch.tensor(0.0, device=z1.device, requires_grad=True)
            if torch.isnan(z1).any() or torch.isnan(z2).any() or torch.isinf(z1).any() or torch.isinf(z2).any():
                return torch.tensor(float("nan"), device=z1.device, requires_grad=True)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            if torch.isnan(z1).any() or torch.isnan(z2).any():
                return torch.tensor(float("nan"), device=z1.device, requires_grad=True)
            z = torch.cat([z1, z2], dim=0)
            similarity_matrix = torch.matmul(z, z.T) / self.temperature
            if torch.isnan(similarity_matrix).any():
                return torch.tensor(float("nan"), device=z1.device, requires_grad=True)
            labels = torch.arange(batch_size, device=z.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
            mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
            similarity_matrix = similarity_matrix.masked_fill(mask, -float("inf"))
            loss = F.cross_entropy(similarity_matrix, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float("nan"), device=z1.device, requires_grad=True)
            return loss

    GINEEncoder = LocalGINEEncoder
    SubgraphRemovalAugmentation = LocalSubgraphRemovalAugmentation
    NTXentLoss = LocalNTXentLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_cache_files(cache_dir: str) -> List[str]:
    paths = sorted(str(p) for p in Path(cache_dir).glob("*_graphs.pt"))
    if not paths:
        raise ValueError(f"No *_graphs.pt files found in {cache_dir}")
    return paths


def infer_cache_dims(cache: Dict[str, Any], source: str) -> Tuple[int, int]:
    dims = cache.get("feature_dims") or {}
    node_dim = dims.get("node_total_dim")
    edge_dim = dims.get("edge_total_dim")
    graphs = cache.get("graphs") or []
    if graphs:
        first = graphs[0]
        if node_dim is None and getattr(first, "x", None) is not None:
            node_dim = int(first.x.size(1))
        if edge_dim is None and getattr(first, "edge_attr", None) is not None:
            edge_dim = int(first.edge_attr.size(1))
    if node_dim is None or edge_dim is None:
        raise ValueError(f"Could not infer feature dims from {source}")
    return int(node_dim), int(edge_dim)


def validate_graph_shape(graph, node_dim: int, edge_dim: int) -> bool:
    if getattr(graph, "num_nodes", 0) < 1:
        return False
    if getattr(graph, "x", None) is None or graph.x.size(0) != graph.num_nodes or graph.x.size(1) != node_dim:
        return False
    if getattr(graph, "edge_index", None) is None or graph.edge_index.size(0) != 2:
        return False
    if getattr(graph, "edge_attr", None) is None or graph.edge_attr.size(1) != edge_dim:
        return False
    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        return False
    if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
        return False
    return True


def load_graph_caches(cache_dir: str) -> Tuple[List[Any], Dict[str, Any]]:
    graphs = []
    files_meta = []
    expected = None
    node_features = None
    edge_features = None
    include_hydrogens = None

    for path in list_cache_files(cache_dir):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict) or "graphs" not in cache:
            raise ValueError(f"Invalid graph cache schema: {path}")

        node_dim, edge_dim = infer_cache_dims(cache, path)
        signature = {
            "node_features": cache.get("node_features"),
            "edge_features": cache.get("edge_features"),
            "feature_dims": cache.get("feature_dims"),
        }
        if expected is None:
            expected = signature
            node_features = cache.get("node_features")
            edge_features = cache.get("edge_features")
            include_hydrogens = cache.get("include_hydrogens")
        elif signature != expected:
            raise ValueError(f"Incompatible graph-cache feature schema: {path}")

        kept = 0
        skipped = 0
        for graph in cache.get("graphs", []):
            if validate_graph_shape(graph, node_dim, edge_dim):
                graphs.append(graph)
                kept += 1
            else:
                skipped += 1
        files_meta.append({
            "cache_file": path,
            "graphs_kept": kept,
            "graphs_skipped": skipped,
        })

    if len(graphs) < 2:
        raise ValueError("Need at least 2 valid graphs for train/val SSL training.")

    metadata = {
        "cache_dir": cache_dir,
        "cache_files": files_meta,
        "node_features": node_features,
        "edge_features": edge_features,
        "feature_dims": expected["feature_dims"] if expected else None,
        "include_hydrogens": include_hydrogens,
        "node_feature_dim": infer_cache_dims({"feature_dims": expected["feature_dims"], "graphs": graphs}, cache_dir)[0],
        "edge_feature_dim": infer_cache_dims({"feature_dims": expected["feature_dims"], "graphs": graphs}, cache_dir)[1],
        "num_graphs": len(graphs),
    }
    return graphs, metadata


def resolve_model_dims(run: RunConfig, metadata: Dict[str, Any]) -> Tuple[int, int]:
    node_dim = metadata["node_feature_dim"]
    edge_dim = metadata["edge_feature_dim"]
    configured_node = run.model.node_feature_dim
    configured_edge = run.model.edge_feature_dim
    if configured_node != "auto":
        configured_node = int(configured_node)
        if configured_node != node_dim:
            raise ValueError(f"model.node_feature_dim={configured_node} does not match cache node dim {node_dim}.")
        node_dim = configured_node
    if configured_edge != "auto":
        configured_edge = int(configured_edge)
        if configured_edge != edge_dim:
            raise ValueError(f"model.edge_feature_dim={configured_edge} does not match cache edge dim {edge_dim}.")
        edge_dim = configured_edge
    return node_dim, edge_dim


def split_graphs(graphs: List[Any], split: SplitConfig) -> Tuple[List[Any], List[Any], List[int], List[int]]:
    indices = list(range(len(graphs)))
    random.Random(split.seed).shuffle(indices)
    n_train = int(len(indices) * split.train_ratio)
    n_train = max(1, min(n_train, len(indices) - 1))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    return train_graphs, val_graphs, train_idx, val_idx


def is_valid_pair(pair) -> bool:
    graph1, graph2 = pair
    return (
        graph1.num_nodes > 0
        and graph2.num_nodes > 0
        and graph1.x is not None
        and graph2.x is not None
        and graph1.x.size(0) == graph1.num_nodes
        and graph2.x.size(0) == graph2.num_nodes
    )


def make_fixed_pairs(graphs: List[Any], aug_cfg: AugmentationConfig, seed: int, desc: str) -> List[Tuple[Any, Any]]:
    set_seed(seed)
    augmentation = SubgraphRemovalAugmentation(
        removal_ratio=aug_cfg.subgraph_removal_ratio,
        seed=seed,
        mask_value=aug_cfg.mask_value,
    )
    pairs = []
    skipped = 0
    for graph in tqdm(graphs, desc=desc):
        try:
            pair = augmentation(graph)
            if is_valid_pair(pair):
                pairs.append(pair)
            else:
                skipped += 1
        except Exception:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} invalid augmented pair(s) for {desc}.")
    return pairs


def collate_pairs(batch):
    valid = [pair for pair in batch if is_valid_pair(pair)]
    if not valid:
        return None
    return Batch.from_data_list([p[0] for p in valid]), Batch.from_data_list([p[1] for p in valid])


def create_loader(pairs, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        FixedPairDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, num_epochs: int, clip_norm: float) -> Tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    batches = 0
    skipped = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            skipped += 1
            continue
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        z1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        z2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        if torch.isnan(z1).any() or torch.isnan(z2).any() or z1.size(0) != z2.size(0):
            skipped += 1
            continue
        loss = criterion(z1, z2)
        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        pbar.set_postfix({"loss": loss.item(), "skip": skipped})
    if batches == 0:
        return float("nan"), batches, skipped
    return total_loss / batches, batches, skipped


def validate(model, loader, criterion, device, epoch: int, num_epochs: int) -> Tuple[float, int, int]:
    model.eval()
    total_loss = 0.0
    batches = 0
    skipped = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} val", leave=False, dynamic_ncols=True)
        for batch in pbar:
            if batch is None:
                skipped += 1
                continue
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            z1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            z2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            if torch.isnan(z1).any() or torch.isnan(z2).any() or z1.size(0) != z2.size(0):
                skipped += 1
                continue
            loss = criterion(z1, z2)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue
            total_loss += float(loss.item())
            batches += 1
            pbar.set_postfix({"loss": loss.item()})
    if batches == 0:
        return float("nan"), batches, skipped
    return total_loss / batches, batches, skipped


def save_checkpoint(path: str, model, optimizer, epoch: int, train_loss: float, val_loss: float, run: RunConfig, metadata: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": val_loss,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "feature_metadata": metadata,
        "run_config": asdict(run),
    }, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str, model, optimizer, scheduler, device) -> Tuple[int, float]:
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("val_loss", checkpoint.get("loss", float("inf"))))
    for _ in range(max(0, start_epoch - 1)):
        scheduler.step()
    return start_epoch, best_val


def append_training_log(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.isfile(path)
    fieldnames = [
        "epoch", "train_loss", "val_loss", "best_val_loss", "lr",
        "train_batches", "val_batches", "train_skipped", "val_skipped",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def choose_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda' but CUDA is not available.")
    return torch.device(name)


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def run_training(run: RunConfig) -> None:
    validate_config(run)
    if not run.confirmed:
        raise SystemExit("Run blocked: confirmed is false. Approve the full config before execution.")

    ensure_runtime_deps()
    set_seed(run.split.seed)
    device = choose_device(run.training.device)
    print(f"Using device: {device}")

    os.makedirs(run.output.checkpoint_dir, exist_ok=True)
    os.makedirs(run.output.log_dir, exist_ok=True)
    write_json(os.path.join(run.output.log_dir, "run_config_used.json"), asdict(run))

    graphs, metadata = load_graph_caches(run.io.cache_dir)
    node_dim, edge_dim = resolve_model_dims(run, metadata)
    metadata["resolved_node_feature_dim"] = node_dim
    metadata["resolved_edge_feature_dim"] = edge_dim

    train_graphs, val_graphs, train_idx, val_idx = split_graphs(graphs, run.split)
    print(f"Loaded {len(graphs):,} graphs from {run.io.cache_dir}")
    print(f"Split: {len(train_graphs):,} train / {len(val_graphs):,} val")

    print("Generating fixed SSL pairs...")
    train_pairs = make_fixed_pairs(train_graphs, run.augmentation, run.split.seed, "fixed train pairs")
    val_pairs = make_fixed_pairs(val_graphs, run.augmentation, run.split.seed + 1, "fixed val pairs")
    if not train_pairs or not val_pairs:
        raise RuntimeError("Fixed pair generation produced empty train or validation pairs.")

    split_manifest = {
        "num_graphs": len(graphs),
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "feature_metadata": metadata,
    }
    write_json(os.path.join(run.output.log_dir, "split_manifest.json"), split_manifest)

    train_loader = create_loader(train_pairs, run.training.batch_size, run.training.num_workers, shuffle=True)
    val_loader = create_loader(val_pairs, run.training.batch_size, run.training.num_workers, shuffle=False)

    model = GINEEncoder(
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        node_embedding_dim=run.model.node_embedding_dim,
        edge_embedding_dim=run.model.edge_embedding_dim,
        hidden_dim=run.model.hidden_dim,
        num_layers=run.model.num_gin_layers,
        dropout=run.model.dropout,
        use_batch_norm=run.model.use_batch_norm,
        pooling=run.model.pooling,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = NTXentLoss(temperature=run.training.temperature)
    optimizer = Adam(model.parameters(), lr=run.training.learning_rate, weight_decay=run.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=run.training.num_epochs, eta_min=run.training.eta_min)

    start_epoch = 1
    best_val_loss = float("inf")
    if run.training.resume_checkpoint:
        if not os.path.isfile(run.training.resume_checkpoint):
            raise FileNotFoundError(f"resume_checkpoint not found: {run.training.resume_checkpoint}")
        start_epoch, best_val_loss = load_checkpoint(run.training.resume_checkpoint, model, optimizer, scheduler, device)

    log_path = os.path.join(run.output.log_dir, "training_log.csv")
    for epoch in range(start_epoch, run.training.num_epochs + 1):
        train_loss, train_batches, train_skipped = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, run.training.num_epochs, run.training.gradient_clip_norm,
        )
        val_loss, val_batches, val_skipped = validate(model, val_loader, criterion, device, epoch, run.training.num_epochs)

        is_valid_val = not (math.isnan(val_loss) or math.isinf(val_loss))
        if run.output.save_best and is_valid_val and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(run.output.checkpoint_dir, "best_model.pt"),
                model, optimizer, epoch, train_loss, val_loss, run, metadata,
            )

        if run.output.save_periodic and epoch % run.training.checkpoint_frequency == 0:
            save_checkpoint(
                os.path.join(run.output.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                model, optimizer, epoch, train_loss, val_loss, run, metadata,
            )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        append_training_log(log_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "lr": lr,
            "train_batches": train_batches,
            "val_batches": val_batches,
            "train_skipped": train_skipped,
            "val_skipped": val_skipped,
        })
        print(
            f"Epoch {epoch}/{run.training.num_epochs}: "
            f"train={train_loss:.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}, lr={lr:.6g}"
        )

    print("Training complete.")
    print(f"Checkpoints: {run.output.checkpoint_dir}")
    print(f"Logs: {run.output.log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GIN-E SSL from raw graph caches.")
    parser.add_argument("--config", help="JSON training config.")
    parser.add_argument("--write-config", metavar="PATH", help="Write config template and exit.")
    parser.add_argument("--confirmed", action="store_true", help="Set confirmed=true after user approval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_config:
        save_run_config(default_run_config(), args.write_config)
        print(f"Wrote config template: {args.write_config}")
        return
    if not args.config:
        raise SystemExit("Provide --config or --write-config.")
    try:
        run = load_run_config(args.config)
        if args.confirmed:
            run.confirmed = True
        run_training(run)
    except ValueError as exc:
        raise SystemExit(f"Invalid config: {exc}")
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
