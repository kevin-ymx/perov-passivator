# Pipeline Automation Examples

## One-shot local pipeline

```bash
python skills/pipeline-automation/scripts/pipeline_controller.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --execute
```

## Dry-run the next action

```bash
python skills/pipeline-automation/scripts/pipeline_controller.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --dry-run
```

## Cron loop check

```bash
python skills/pipeline-automation/scripts/cron_manager.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --print
```

After approval on the target Linux/HPC machine:

```bash
python skills/pipeline-automation/scripts/cron_manager.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --install \
  --yes
```

Remove the managed cron block:

```bash
python skills/pipeline-automation/scripts/cron_manager.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --remove \
  --yes
```

## Watch mode for temporary sessions

```bash
python skills/pipeline-automation/scripts/pipeline_controller.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --execute \
  --watch \
  --interval-hours 6
```

## Slurm stage pattern

```json
{
  "id": "gine_ssl_infer",
  "name": "Run GNN inference shards",
  "enabled": true,
  "kind": "slurm",
  "depends_on": [],
  "workdir": "/scratch/yeming/perov-passivator",
  "render_command": "python skills/gine-ssl-infer/scripts/render_slurm_script.py --config runs/infer/slurm_config.json",
  "submit_command": "sbatch runs/infer/jobs/gine_ssl_infer.slurm",
  "job_id_regex": "Submitted batch job (\\d+)",
  "check_command": "squeue -j {job_id} -h",
  "expected_outputs": [
    "runs/infer/outputs/*_done.json"
  ],
  "success_markers": [],
  "failure_markers": [],
  "skip_if_outputs_exist": true,
  "timeout_hours": 48,
  "retry": {
    "max_attempts": 2,
    "retry_delay_minutes": 30
  }
}
```
