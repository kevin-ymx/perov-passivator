# comparison_3feat_electronegativity

Ablation study: **atomic_num + tetrahedral_chirality + electronegativity** (`node_feature_dim=3`).

Same training pipeline as `comparison_2feat` and main `train_downstream.py` (weighted MSE,
val-MAE checkpoint selection). Differs only in node features and SSL/downstream caches.

## Node features

- atomic_num
- chirality
- electronegativity

## Train SSL (from this directory)

```bash
python train_ssl.py
```

Cache: `cache_3feat_electronegativity`

## Train downstream

```bash
python train_downstream.py --checkpoint-dir ./checkpoints/downstream --log-dir ./logs/downstream
```

Downstream graph cache: `downstream_graph_3feat_electronegativity_cache.pt`

## Pretrained encoder

Place 3-feature SSL `best_model.pt` under `./checkpoints/best_model.pt` (train SSL in this folder first).
