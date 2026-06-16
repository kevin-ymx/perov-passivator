"""
Configuration for 3-feature comparison: atomic_num + tetrahedral_chirality + coordination_number.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data loading mode
    use_cache: bool = False

    # Data paths
    csv_file: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/combine.csv"
    csv_file_sampled: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/sampled.csv"
    cache_dir: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/cache_3feat_coordination"
    max_molecules: Optional[int] = None

    # Augmentation
    subgraph_removal_ratio: float = 0.25
    train_val_split: float = 0.8
    fixed_augmentation: bool = True

    # Model parameters
    node_feature_dim: int = 3  # atomic_num, chirality, coordination_number
    edge_feature_dim: int = 2  # bond_type, bond_direction
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 6
    dropout: float = 0.1

    # GIN-E training parameters
    batch_size: int = 512
    num_epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07
    checkpoint_frequency: int = 5
    resume_checkpoint: str = "/kfs3/scratch/yeming/ai4m/prediction/comparison_3feat_coordination/checkpoints/checkpoint_epoch_145.pt"  # Empty: train from scratch

    # Downstream data path
    downstream_csv: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/prediction/result/strongest_binding.csv/min_ads_mult1p2_struct_cleaned_merged_wSMILES.csv"

    # Downstream mol source
    downstream_prefer_smiles: bool = True
    downstream_skip_pubchem: bool = True

    # PubChem API (downstream)
    pubchem_request_delay: float = 0.3
    pubchem_max_retries: int = 2
    pubchem_retry_base_delay: float = 2.0

    # Downstream graph cache
    downstream_use_graph_cache: bool = True
    downstream_graph_cache_path: str = "/kfs3/scratch/yeming/ai4m/prediction/cache/downstream_graph_3feat_coordination_cache.pt"
    downstream_zero_binding_tags: bool = False  # No binding_tag in this ablation
    downstream_extra_csv: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/prediction/result/strongest_binding.csv/adsorption_energy_data.csv"

    # Downstream model parameters
    num_property_tasks: int = 1
    downstream_mlp_hidden_dim: int = 512
    downstream_mlp_dropout: float = 0.1
    downstream_task_hidden_dim: int = 256
    downstream_task_dropout: float = 0.1
    freeze_pretrained_encoder: bool = False

    # Downstream data split
    downstream_train_split: float = 0.7
    downstream_val_split: float = 0.2
    downstream_test_split: float = 0.1

    # Downstream training parameters
    downstream_batch_size: int = 128
    downstream_num_epochs: int = 300
    downstream_learning_rate: float = 0.001
    downstream_weight_decay: float = 0
    downstream_rare_loss_weight: float = 1.0  # MSE multiplier for strong (E<-1.3) and weak ([-0.6,0]) binders
    downstream_best_metric: str = "mae"  # "loss" (val MSE) or "mae" (val MAE) for best-checkpoint selection

    # Device
    device: str = "cuda"

    # Output paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Other
    seed: int = 42
    num_workers: int = 64
