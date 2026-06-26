# Examples

## Sharded SMILES Inference

Use one Slurm worker per GPU. The renderer computes pending shards from `io.input_dir`, `io.shard_glob`, and existing shard outputs, then partitions them into unique per-worker shard lists.

```bash
python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --write-config run_config.json
python skills/gine-ssl-infer/scripts/render_slurm_script.py --write-config slurm_config.json
```

After editing and approving both configs:

```bash
python skills/gine-ssl-infer/scripts/render_slurm_script.py --config slurm_config.json
sbatch /scratch/yeming/jobs/gine_ssl_infer.slurm
```

For restart, rerun the render command before `sbatch`. The new worker assignment contains only shards missing a complete done marker and full embedding output.

For 16 GPUs as 4 nodes x 4 GPUs/node x 4 tasks/node, use:

```json
{
  "nodes": 4,
  "ntasks": 16,
  "gpus_per_node": 4,
  "cpus_per_task": 32
}
```

With 45 pending shards, the assignment contains 16 workers. Workers 0-12 receive 3 shards each, and workers 13-15 receive 2 shards each.

## Single CSV Debug Run

Set `io.mode` to `"single"` and run one CSV through the same script:

```bash
python skills/gine-ssl-infer/scripts/gine_ssl_infer.py --config run_config.json
```

## Explicit Feature Match

Keep `"auto"` when the checkpoint has feature metadata. If an older checkpoint lacks metadata, list features explicitly:

```json
"graph": {
  "include_hydrogens": false,
  "node_features": ["element_onehot", "partial_charge", "hybridization_onehot", "degree", "valence_electrons", "electronegativity"],
  "edge_features": ["bond_type_onehot", "bond_direction_onehot"]
}
```
