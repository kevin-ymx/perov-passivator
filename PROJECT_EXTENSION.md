# Project Extension: GIN-E SSL + Downstream Prediction

This project now focuses solely on the GIN-E architecture for contrastive SSL pretraining and for downstream molecular property prediction via a shallow multi-head MLP.

## Project Structure

### Key Files

1. **`train.py`**: Contrastive SSL pretraining for the GIN-E encoder using NT-Xent loss and augmented graph pairs.

2. **`models/downstream_model.py`**: Downstream predictor composed of:
   - Pretrained GIN-E encoder
   - Shallow two-layer MLP to refine graph embeddings
   - Three independent prediction heads (each a two-layer MLP) for property regression

3. **`train_downstream.py`**: Training script for the downstream property prediction pipeline. Loads the pretrained GIN-E encoder, attaches the shallow MLP stack, and optimizes the prediction heads.

### Updated Files

1. **`config.py`**: Houses all configuration knobs for SSL training and downstream fine-tuning (batch sizes, learning rates, dropout, etc.).

2. **`requirements.txt`**: Contains the minimal dependencies needed for GIN-E pretraining and downstream tasks (PyTorch, PyG, RDKit, etc.).

## Usage

### Step 1: Pretrain GIN-E Encoder

```bash
python train.py
```

Outputs `checkpoints/best_model.pt`, which stores the SSL-pretrained GIN-E encoder weights.

### Step 2: Train Downstream Predictor

```bash
python train_downstream.py
```

The script:
- Loads `checkpoints/best_model.pt` (if available)
- Freezes or fine-tunes the encoder based on `freeze_pretrained_encoder`
- Trains the shallow MLP + three prediction heads on molecular property targets (currently dummy values—replace with real labels for practical use)

## Configuration Highlights (`config.py`)

- `node_feature_dim`, `edge_feature_dim`: Raw feature dimensions extracted from the SDF file.
- `hidden_dim`, `num_gin_layers`, `dropout`: Define the GIN-E encoder capacity.
- `batch_size`, `num_epochs`, `learning_rate`, `temperature`: SSL training hyperparameters.
- `num_property_tasks`: Number of downstream properties predicted (default: 3).
- `downstream_*`: Controls hidden sizes/dropout for the shallow MLP and each prediction head.
- `freeze_pretrained_encoder`: Freeze (True) or fine-tune (False) the SSL encoder during downstream training.
- `downstream_learning_rate`, `downstream_weight_decay`, `downstream_num_epochs`: Downstream optimization parameters.

## Downstream Architecture

1. **GIN-E encoder** converts molecular graphs to fixed-size embeddings.
2. **Shallow combining MLP** (two linear layers with ReLU + dropout) refines the embeddings.
3. **Three prediction heads** (identical structures) each output a single regression target, enabling multi-task learning.

## Loading Real Property Labels

`train_downstream.py` currently generates random targets for demonstration. Replace this logic inside `PropertyDataset` with actual property extraction (e.g., from SDF tags or an external CSV) to perform meaningful training.

## Notes

- Graphormer dependencies and scripts were removed for simplicity.
- The repository now centers on a single, well-tested SSL backbone (GIN-E) that feeds the downstream predictor.
- All training scripts still save periodic checkpoints (`checkpoints/` folder) for reproducibility.

