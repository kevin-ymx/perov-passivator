# PubChem molecule filter — examples

**Kestrel paths:** skill on `/home/yeming/skills/pubchem-mol-filter`; run workspace
(configs, logs, jobs, output) on `/scratch/yeming/pubchem-mol-filter/` (see SKILL.md).
**Input CSV/shards already exist on the HPC — the user provides that path
separately; it is not under the run folder's `data/`.** **Do not** use slurm_mcp
default directories (`/datasets`, `/results`, `/logs`, `list_datasets`, etc.) —
use only paths from confirmed `run_config.json` and `slurm_config.json`.

Each run is fully specified by `run_config.json` (+ `slurm_config.json` on HPC).

**Before running:** present the **complete** `run_config.json` (+ `slurm_config.json`
on HPC) for user approval. Use template defaults for fields the user did not
mention. Set `"confirmed": true` and execute only after approval.

## Confirmation example (required step)

User: *Filter shards at `/projects/ai4m/pubchem_shards` with standard criteria on Kestrel, account m3342, via slurm_mcp.*

Agent builds configs (defaults for filters, workers=64, output under run folder),
then presents the **full JSON** (not field-by-field questions):

> Here are the draft configs. Unspecified fields use template defaults. Please
> confirm or tell me what to change:
>
> **`run_config.json`** — (full JSON shown)
> **`slurm_config.json`** — (full JSON shown)

Only after user says "confirmed" → set `"confirmed": true` → slurm_mcp workflow.

If the user had **not** given the input path or account, ask for those **only**,
then include them in the draft before presenting for approval.

## Example 1: Default criteria (single file, Kestrel)

User: *Filter `/projects/ai4m/data/combine.csv` (input already on HPC, path given
by me) and write the result under my run folder, 64 workers.*

Input path is whatever the user provides; output defaults under `RUN_ROOT/data/`.

```json
{
  "confirmed": true,
  "io": {
    "mode": "single",
    "input": "/projects/ai4m/data/combine.csv",
    "output": "/scratch/yeming/pubchem-mol-filter/data/combine_filtered.csv",
    "input_dir": null,
    "output_dir": null,
    "workers": 64,
    "save_config_used": "/scratch/yeming/pubchem-mol-filter/run_configs/filter_config_used.json"
  },
  "filters": { "...": "confirmed with user" }
}
```

## Example 2: Shard batch via slurm_mcp (Kestrel)

User: *Filter shards in `/projects/ai4m/pubchem_shards` (input already on HPC,
path given by me), output to my run folder, 104 workers, account m3342, via slurm_mcp.*

**`run_config.json`:**

```json
{
  "confirmed": true,
  "io": {
    "mode": "shards",
    "input": null,
    "output": null,
    "input_dir": "/projects/ai4m/pubchem_shards",
    "output_dir": "/scratch/yeming/pubchem-mol-filter/data/output_shards",
    "workers": 104,
    "save_config_used": "/scratch/yeming/pubchem-mol-filter/run_configs/filter_config_used.json"
  },
  "filters": { "...": "confirmed with user" }
}
```

**`slurm_config.json`:**

```json
{
  "confirmed": true,
  "job_name": "pubchem-mol-filter",
  "account": "m3342",
  "partition": "cpu",
  "nodes": 1,
  "cpus_per_task": 104,
  "time_limit": "24:00:00",
  "output_log": "/scratch/yeming/pubchem-mol-filter/logs/mol-filter-%j.out",
  "error_log": "/scratch/yeming/pubchem-mol-filter/logs/mol-filter-%j.err",
  "bashrc": "/home/yeming/.bashrc",
  "conda_env": "/scratch/yeming/conda_envs/ai4m",
  "filter_script": "/home/yeming/skills/pubchem-mol-filter/scripts/filter_molecules_configurable.py",
  "run_config_path": "/scratch/yeming/pubchem-mol-filter/run_configs/run_config.json",
  "rendered_script_path": "/scratch/yeming/pubchem-mol-filter/jobs/mol_filter.slurm"
}
```

**slurm_mcp sequence:**

```
1. write_file(/scratch/yeming/pubchem-mol-filter/run_configs/run_config.json, ...)
2. write_file(/scratch/yeming/pubchem-mol-filter/run_configs/slurm_config.json, ...)
3. run_shell_command(
     "python /home/yeming/skills/pubchem-mol-filter/scripts/render_slurm_script.py "
     "--config /scratch/yeming/pubchem-mol-filter/run_configs/slurm_config.json"
   )
4. read_file(/scratch/yeming/pubchem-mol-filter/jobs/mol_filter.slurm)
5. submit_job(script_content=<step 4>, partition="cpu", account="m3342",
              time_limit="24:00:00", cpus=104)
6. get_job_details(job_id=...)
7. read_file(/scratch/yeming/pubchem-mol-filter/logs/mol-filter-<id>.out, tail_lines=200)
8. list_directory(/scratch/yeming/pubchem-mol-filter/data/output_shards)
```

## Custom run folder under scratch

User: *Use run folder `ai4m/mol-filter` on scratch.*

Set `RUN_ROOT=/scratch/yeming/ai4m/mol-filter` and update all scratch paths
(configs, logs, jobs, output) under that root. Skill path stays on home:

`/home/yeming/skills/pubchem-mol-filter/scripts/...`

Input path is unchanged — it stays wherever the user said the data already lives.
Confirm the folder name with the user before execution.

## Prompt → config mapping

| User phrase | Config field |
|-------------|--------------|
| Kestrel scratch run folder | `RUN_FOLDER` → output/logs/configs under `/scratch/yeming/<folder>/` |
| input CSV/dir already on HPC | `io.input` / `io.input_dir` (**user-provided path**) |
| output location | `io.output` / `io.output_dir` (default under `RUN_ROOT/data/`) |
| "use N workers" | `io.workers` == `slurm.cpus_per_task` |
| filter criteria | `filters.*` |
| Slurm account | `slurm.account` |
| skill script location | fixed: `/home/yeming/skills/pubchem-mol-filter/scripts/...` |
