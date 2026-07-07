---
name: pipeline-automation
description: Deterministically orchestrate multi-stage agent skill pipelines with config, status, journal, restart, and monitor-loop files. Use when the user wants to execute an entire skill pipeline, chain multiple skills, automate long-running local or Slurm stages, resume unfinished work, schedule regular status checks, or maintain a pipeline status file without requiring LLM reasoning at each check.
---

# Pipeline Automation

Use this skill to run a whole multi-stage skill workflow from one pipeline config.
The controller is deterministic: it checks stage status, dependencies, expected
outputs, retry limits, and Slurm job state, then chooses the next mechanical
action.

## Workflow

1. Build a pipeline config from `config_template.json`.
2. Put every stage command, dependency, expected output, and retry policy in the
   config.
3. Show the full config to the user for approval.
4. Execute only after `"confirmed": true`.
5. Initialize or update the status file.
6. For loop checks or long-running HPC workflows, use cron as the default
   scheduler. Render the cron entry, show it to the user, and install it only
   after explicit approval.

```bash
python skills/pipeline-automation/scripts/pipeline_controller.py --write-config runs/my_pipeline/pipeline_config.json
python skills/pipeline-automation/scripts/pipeline_controller.py --pipeline-config runs/my_pipeline/pipeline_config.json --status runs/my_pipeline/pipeline_status.json --init-status
python skills/pipeline-automation/scripts/pipeline_controller.py --pipeline-config runs/my_pipeline/pipeline_config.json --status runs/my_pipeline/pipeline_status.json --execute
python skills/pipeline-automation/scripts/cron_manager.py --pipeline-config runs/my_pipeline/pipeline_config.json --status runs/my_pipeline/pipeline_status.json --print
```

## Stage Types

Use `kind: "command"` for direct local/HPC-node commands. The controller runs
the command synchronously and marks the stage succeeded only when the return code
is zero and configured expected outputs or success markers exist.

Use `kind: "slurm"` for long GPU/HPC work. The controller optionally runs a
render command, submits the job, stores the parsed job id, and later checks the
job with `check_command`. A Slurm stage is marked succeeded when expected outputs
or success markers exist.

## Restart Behavior

- Completed stages are skipped on later runs.
- Stages with `skip_if_outputs_exist: true` are marked succeeded if their
  expected outputs already exist.
- Failed stages retry until `retry.max_attempts` is exhausted.
- Slurm stages can be resubmitted on restart after a failed or incomplete job.
- The controller appends deterministic events to `pipeline_journal.jsonl`.

## Status States

Stages use these statuses: `pending`, `running`, `waiting_external_job`,
`succeeded`, `failed`, `retrying`, `blocked`, and `skipped`.

The whole pipeline is `pending`, `running`, `waiting`, `succeeded`, `failed`, or
`blocked`.

## Regular Monitoring

For periodic loop checks, prefer cron on Linux/HPC. First render the exact cron
entry:

```bash
python skills/pipeline-automation/scripts/cron_manager.py \
  --pipeline-config runs/my_pipeline/pipeline_config.json \
  --status runs/my_pipeline/pipeline_status.json \
  --print
```

After the user approves the rendered entry, install it on the target Linux/HPC
machine:

```bash
python skills/pipeline-automation/scripts/cron_manager.py \
  --pipeline-config runs/my_pipeline/pipeline_config.json \
  --status runs/my_pipeline/pipeline_status.json \
  --install \
  --yes
```

The controller uses the status file to avoid repeating finished stages.
Use `--remove --yes` with the same config and status paths to remove the
managed cron block after the pipeline is done or no longer needs monitoring.

## Notes

- This skill does not replace individual scientific skills. It calls their
  approved commands in a configured order.
- Individual skill run configs still need their own approval gates when those
  skills require `"confirmed": true`.
- Reject placeholder paths, placeholder accounts, and unconfirmed pipeline
  configs before execution.
- Do not silently edit cron. Always show the rendered cron entry first and
  install only after explicit user approval.
