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
  "accounting_command": "sacct -X -n -P -j {job_id} -o State",
  "expected_outputs": [
    "runs/infer/outputs/*_done.json"
  ],
  "success_markers": [
    "runs/infer/outputs/_SUCCESS"
  ],
  "failure_markers": [],
  "skip_if_outputs_exist": true,
  "timeout_hours": 48,
  "retry": {
    "max_attempts": 2,
    "retry_delay_minutes": 30
  }
}
```

`check_command` detects jobs that are still active. If the job is no longer in
the queue, `accounting_command` distinguishes `COMPLETED`, `TIMEOUT`, `FAILED`,
`CANCELLED`, `OUT_OF_MEMORY`, and other Slurm states. `TIMEOUT` is restartable:
the stage moves to `retrying` when another configured attempt is available. If
the attempt budget is exhausted, it becomes `blocked`, not `failed`. Hard
failure states use the normal retry/failure policy even if partial output files
exist. An active job remains waiting even when output paths have appeared. A
`COMPLETED` job is still considered unsuccessful when declared outputs or the
final `_SUCCESS` marker are missing. The Slurm workflow must create `_SUCCESS`
only after all shards have finished and their outputs have been validated.
`retry.max_attempts` includes the initial submission, so a value of `2` permits
one automatic restart after `TIMEOUT`.

For a direct command stage, append the marker creation only after the scientific
command succeeds:

```bash
python run_stage.py && mkdir -p runs/example/markers && touch runs/example/markers/stage.success
```

Set `timeout_hours` longer than the Slurm job time limit plus at least one
monitoring interval. This prevents the controller from retrying a job that
Slurm still considers active.
