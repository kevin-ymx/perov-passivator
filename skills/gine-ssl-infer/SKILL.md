---
name: gine-ssl-infer
description: Run standalone GPU Slurm inference with a trained GIN-E-style molecular GNN encoder checkpoint. Use when the user wants to embed molecules from CSV SMILES inputs, including single CSV files or sharded CSV directories, with checkpoint-matched graph feature extraction, one-GPU-per-shard Slurm array execution, and restartable shard processing.
---

# GIN-E SSL Inference

Use this skill to generate graph-level embeddings from SMILES using a trained GIN-E-style encoder checkpoint.

## Workflow

1. Build a run config from `config_template.json`.
2. Build a Slurm config from `slurm_config_template.json`.
3. Show both full JSON configs to the user for approval before execution.
4. Set `"confirmed": true` only after approval.
5. Render the Slurm script and worker assignment:

```bash
python skills/gine-ssl-infer/scripts/render_slurm_script.py --config slurm_config.json
```

6. Submit the rendered script:

```bash
sbatch /path/to/gine_ssl_infer.slurm
```

## Runtime

- Run from the directory configured as `slurm.workdir`.
- Use one Slurm worker task per GPU.
- Render the Slurm script before every new submission or restart; rendering scans all shard outputs and writes a fresh worker assignment JSON.
- Each GPU worker gets a unique shard list and processes its shards independently.
- The inference script uses the GPU assigned by Slurm through `CUDA_VISIBLE_DEVICES`.
- Restartability is shard-level: completed shards are removed from the next worker assignment when their embedding CSV row count matches the input row count and their done marker says completion succeeded.

## Input

- `io.mode: "single"` reads `io.input`.
- `io.mode: "shards"` reads all CSV files matching `io.shard_glob` under `io.input_dir`.
- `cid_column` is optional; lowercase fallback matching is supported.
- `smiles_column` is required, with lowercase fallback matching.

## Feature Matching

- Prefer `graph.node_features: "auto"`, `graph.edge_features: "auto"`, and `graph.include_hydrogens: "auto"` so the script resolves graph construction from checkpoint metadata.
- If features are explicitly listed, they must exactly match the checkpoint metadata.
- If the checkpoint does not contain feature metadata, explicit graph features and model dimensions are required.
- Unknown feature names are rejected before inference.

## Outputs

For each input CSV shard, the script writes:

- `<input_stem>_embeddings.csv`
- `<input_stem>_failures.csv`
- `<input_stem>_manifest.json`
- `<input_stem>_done.json`
- A render-time worker assignment file, by default next to the rendered Slurm script as `<script_stem>_worker_assignment.json`

Embedding rows include `cid`, `smiles`, `source_file`, `source_row`, `status`, and `emb_0...emb_N`.
