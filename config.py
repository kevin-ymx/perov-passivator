"""
Configuration file for contrastive self-supervised learning of charge-aware molecular representation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data loading mode
    use_cache: bool =False  # True = Option 1 (load pre-augmented pairs from cache), False = Option 2 (load CSV, convert, augment in-memory)
    
    # Data paths
    csv_file: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/combine.csv"  # Full CSV with PUBCHEM_COMPOUND_CID, SMILES
    csv_file_sampled: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/sampled.csv"  # Sampled CSV (for train_ssl_sampled.py)
    cache_dir: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/cache"  # val.pt + train_shard_0.pt to train_shard_3.pt (pre-augmented pairs, built by build_graph_cache.py)
    max_molecules: Optional[int] = None  # Limit molecules to load (None = all). Used by visualize_tsne.py and Option 2
    
    # Augmentation (used by Option 2 and build_graph_cache.py)
    subgraph_removal_ratio: float = 0.25
    train_val_split: float = 0.8  # 80% train, 20% val (used by Option 2)
    fixed_augmentation: bool = True  # Option 2 only: True = augment once at start (same pairs each epoch), False = augment on-the-fly (fresh pairs each epoch)
    
    # Model parameters
    node_feature_dim: int = 8  # atomic_num, chirality, partial_charge, hybridization, coordination_num, valence_electrons, electronegativity, binding_tag
    edge_feature_dim: int = 2  # bond_type, bond_direction
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 6
    dropout: float = 0.1
    
    # GIN-E training parameters
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07  # Temperature parameter for NT-Xent loss
    checkpoint_frequency: int = 3  # Save periodic checkpoint every N epochs (each epoch trains on all 4 shards)
    resume_checkpoint: str = "./checkpoints/best_model.pt"  # Path to checkpoint to resume from (e.g., "./checkpoints/best_model.pt")
    
    # Downstream data path
    downstream_csv: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/prediction/result/strongest_binding.csv/min_ads_mult1p2_struct_cleaned_merged_wSMILES.csv"  # Merged cleaned CSV (cid, adsorption_energy, pb_bond_encoding, adsorbate_structure). If empty, train_downstream uses dataset/prediction/min_ads_mult1p2_struct_cleaned_merged.csv relative to script dir.

    # Downstream mol source: use SMILES to skip PubChem API (no 503)
    downstream_prefer_smiles: bool = True  # If True and row has SMILES, build mol from SMILES (no API call)
    downstream_skip_pubchem: bool = True  # If True, never call PubChem; only process rows with valid SMILES (no 503)

    # PubChem API (downstream): only used when downstream_skip_pubchem=False
    pubchem_request_delay: float = 0.3  # Seconds to wait between requests
    pubchem_max_retries: int = 2  # Retries per CID on 503/timeout
    pubchem_retry_base_delay: float = 2.0  # Base delay in seconds; doubled each retry

    # Downstream graph cache: save/load built graphs to skip PubChem + build on next run
    downstream_use_graph_cache: bool = True  # If True, load from cache when present, save after building
    downstream_graph_cache_path: str = "/kfs3/scratch/yeming/ai4m/prediction/cache/downstream_graph_notag_cache.pt"  # If empty, derived from downstream_csv path (same dir, _graph_cache.pt suffix)
    downstream_zero_binding_tags: bool = True  # If True, zero out binding_tag feature for all nodes before caching/training
    downstream_extra_csv: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/prediction/result/strongest_binding.csv/adsorption_energy_data.csv"  # Extra CSV (SMILES, best_adsorption_energy) to extend training set; empty string to skip

    # Downstream model parameters
    num_property_tasks: int = 1  # Number of molecular properties to predict (1 = binding energy only)
    downstream_mlp_hidden_dim: int = 512
    downstream_mlp_dropout: float = 0.1  # Increased dropout for regularization
    downstream_task_hidden_dim: int = 256
    downstream_task_dropout: float = 0.1  # Increased dropout for regularization
    freeze_pretrained_encoder: bool = False  # Whether to freeze pretrained GIN-E encoder
    
    # Downstream data split (train/val/test)
    downstream_train_split: float = 0.7
    downstream_val_split: float = 0.2
    downstream_test_split: float = 0.1
    
    # Downstream training parameters
    downstream_batch_size: int = 128  # Larger batch for more stable gradients
    downstream_num_epochs: int = 300  # Number of training epochs
    downstream_learning_rate: float = 0.001  # Lower learning rate for fine-tuning
    downstream_weight_decay: float = 0  # Increased weight decay for regularization
    downstream_rare_loss_weight: float = 1.0  # MSE multiplier for strong (E<-1.3) and weak ([-0.6,0]) binders
    downstream_best_metric: str = "mae"  # "loss" (val MSE) or "mae" (val MAE) for best-checkpoint selection
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Output paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Other
    seed: int = 42
    num_workers: int = 64  # DataLoader workers