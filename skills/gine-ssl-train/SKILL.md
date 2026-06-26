---
name: gine-ssl-train
description: >-
  Train a GIN-E contrastive self-supervised learning model from raw molecular
  PyTorch Geometric graph caches. Use when the user asks to train, pretrain, or
  fine-tune a GIN-E SSL encoder from cached molecule graphs, with configurable
  model architecture, split percentages, augmentation, training parameters, and
  GPU Slurm sbatch submission. Always build full training and Slurm configs from
  the user's request, present them for approval, and execute only after
  "confirmed": true.
---

# GIN-E SSL training

Train a GIN-E SSL encoder from raw PyTorch Geometric graph cache files. This
skill runs through a GPU Slurm batch job by rendering a `.slurm` script and
submitting it with `sbatch`.

## Workflow

1. Build `run_config.json` from `config_template.json`.
2. Build `slurm_config.json` from `slurm_config_template.json`.
3. Present both complete JSON configs for user approval.
4. After approval, set `"confirmed": true` in both configs.
5. Render the Slurm script:

```bash
python skills/gine-ssl-train/scripts/render_slurm_script.py --config slurm_config.json
```

6. Submit:

```bash
sbatch /path/to/gine_ssl_train.slurm
```

The Slurm script changes to `slurm.workdir` before running Python. Use this as
the run directory for relative paths in configs and scripts.

## Training behavior

- Load every `*_graphs.pt` file from `io.cache_dir`.
- Require all graph caches to use compatible node/edge feature metadata.
- Infer model input dimensions from graph-cache metadata unless the config sets
  explicit integer dimensions.
- Split raw graphs into train/validation sets.
- Generate fixed SSL train and validation pairs once before epoch 1.
- Reuse those same pairs for every epoch, matching the current
  `fixed_augmentation=True` SSL pipeline.
- Use the bundled GIN-E encoder, subgraph-removal augmentation, and NT-Xent loss
  implementations inside this skill.

## Outputs

The runner writes:

- `best_model.pt`
- `checkpoint_epoch_<N>.pt`
- `training_log.csv`
- `run_config_used.json`
- `split_manifest.json`

Checkpoints include model/optimizer state, losses, feature metadata, and the
resolved run config.

## Notes

- Default Slurm partition is `short`.
- Default runtime is single-node, single-GPU.
- Multi-node/distributed training is out of scope for this skill version.
- Do not run until both configs are approved with `"confirmed": true`.
