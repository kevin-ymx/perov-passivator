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

## Watch mode

```bash
python skills/pipeline-automation/scripts/pipeline_controller.py \
  --pipeline-config runs/sam_derivative_discovery/pipeline_config.json \
  --status runs/sam_derivative_discovery/pipeline_status.json \
  --execute \
  --watch \
  --interval-hours 6
```

## Scheduled cron check

```bash
0 */6 * * * cd /path/to/perov-passivator && python skills/pipeline-automation/scripts/pipeline_controller.py --pipeline-config runs/sam_derivative_discovery/pipeline_config.json --status runs/sam_derivative_discovery/pipeline_status.json --execute
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
