# gine-ssl-train examples

## GPU Slurm training from raw graph cache files

Training config:

```json
{
  "confirmed": true,
  "io": {
    "cache_dir": "/scratch/yeming/raw_graph_cache"
  },
  "split": {
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "seed": 42
  },
  "augmentation": {
    "subgraph_removal_ratio": 0.25,
    "mask_value": 0.0
  },
  "model": {
    "node_feature_dim": "auto",
    "edge_feature_dim": "auto",
    "node_embedding_dim": 128,
    "edge_embedding_dim": 64,
    "hidden_dim": 256,
    "num_gin_layers": 6,
    "dropout": 0.1,
    "use_batch_norm": true,
    "pooling": "mean"
  },
  "training": {
    "batch_size": 512,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "temperature": 0.07,
    "eta_min": 0.000001,
    "gradient_clip_norm": 1.0,
    "device": "cuda",
    "num_workers": 32,
    "checkpoint_frequency": 3,
    "resume_checkpoint": null
  },
  "output": {
    "checkpoint_dir": "/scratch/yeming/gine_ssl_train/checkpoints",
    "log_dir": "/scratch/yeming/gine_ssl_train/logs",
    "save_best": true,
    "save_periodic": true
  }
}
```

Slurm config:

```json
{
  "confirmed": true,
  "job_name": "gine-ssl-train",
  "account": "choiseprojec",
  "partition": "short",
  "nodes": 1,
  "ntasks": 1,
  "gpus_per_node": 1,
  "cpus_per_task": 32,
  "mem": "0",
  "time_limit": "24:00:00",
  "qos": null,
  "workdir": "/scratch/yeming/gine_ssl_train",
  "bashrc": "/home/yeming/.bashrc",
  "conda_env": "/scratch/yeming/conda_envs/ai4m",
  "train_script": "/home/yeming/skills/gine-ssl-train/scripts/gine_ssl_train.py",
  "run_config_path": "/scratch/yeming/gine_ssl_train/run_config.json",
  "output_log": "/scratch/yeming/gine_ssl_train/logs/gine-ssl-train-%j.out",
  "error_log": "/scratch/yeming/gine_ssl_train/logs/gine-ssl-train-%j.err",
  "rendered_script_path": "/scratch/yeming/gine_ssl_train/jobs/gine_ssl_train.slurm"
}
```

Render and submit:

```bash
python /home/yeming/skills/gine-ssl-train/scripts/render_slurm_script.py --config slurm_config.json
sbatch /scratch/yeming/gine_ssl_train/jobs/gine_ssl_train.slurm
```
