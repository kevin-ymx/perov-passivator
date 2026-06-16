---
name: pubchem-mol-filter
description: >-
  Filter PubChem CSV molecules (PUBCHEM_COMPOUND_CID, SMILES) with configurable
  RDKit criteria. Use when the user asks to filter, clean, or subset PubChem
  CSV/shard data by element types, molecular weight, heavy atoms, ring size,
  composition (N/O/S/P), sanitization, radicals, isotopes, or single-component
  requirements.   Always configure filters from the user prompt before running.
  Present complete config file(s) for user approval before execution; use template
  defaults for fields the user did not mention. On Kestrel HPC use stable paths:
  skill on /home/yeming/skills/pubchem-mol-filter, run data on
  /scratch/yeming/pubchem-mol-filter (see SKILL.md Kestrel paths). slurm_mcp:
  render on cluster, read_file the .slurm, submit_job(script_content=...).
  Do not use slurm_mcp default directories for I/O; use user-confirmed paths
  from run_config and slurm_config only.
---

# PubChem CSV molecule filter

Self-contained, portable skill for molecule filtering. All code lives in
`scripts/` inside this folder, so it runs from any agent (Claude Code, Codex,
Cursor, etc.) and is transferable across machines/HPC. The only prerequisite is
the PubChem CSV data and a Python env with RDKit.

## When to use

Apply this skill when the user wants to filter molecules from PubChem CSV files
(single file or shard directory). Criteria come from the **user prompt** and must
be written to a JSON config **before** execution.

## Prerequisites

- Python 3.8+ with RDKit and tqdm: `pip install -r requirements.txt`
  (or `conda install -c conda-forge rdkit tqdm`).
- PubChem CSV input with columns `PUBCHEM_COMPOUND_CID` (or `cid`) and
  `SMILES` (or `smiles`). Single file or a directory of `*.csv` shards.

## Files in this skill

```
pubchem-mol-filter/
├── SKILL.md                  # this file
├── examples.md               # prompt → config examples
├── requirements.txt          # rdkit, tqdm
├── config_template.json      # full run config (io + filters) — Kestrel scratch I/O defaults
├── slurm_config_template.json # HPC Slurm settings — home skill + scratch run paths
└── scripts/
    ├── molecule_filter.py                # FilterConfig/IOConfig/RunConfig + check_filters (core)
    ├── filter_molecules_configurable.py  # config-driven runner
    ├── submit_filter.slurm.template      # Slurm script template ({{PLACEHOLDERS}})
    └── render_slurm_script.py            # render template from slurm_config.json
```

`SKILL_DIR` below refers to this skill's directory. On **Kestrel**, install the
skill under home; run all job data and I/O under scratch (see [Kestrel paths](#kestrel-paths)).

## Kestrel paths

Stable path facts for Kestrel (use these unless the user specifies otherwise):

| Symbol | Path | Purpose |
|--------|------|---------|
| `HOME_ROOT` | `/home/yeming` | Home directory |
| `SCRATCH_ROOT` | `/scratch/yeming` | Scratch / job workspace root |
| `SKILL_DIR` | `/home/yeming/skills/pubchem-mol-filter` | Skill install (code + `SKILL.md`) — **home** |
| `RUN_FOLDER` | `pubchem-mol-filter` (default) | Subfolder under scratch for this workflow |
| `RUN_ROOT` | `/scratch/yeming/pubchem-mol-filter` | Run workspace — **scratch** |

**Rule:** skill files live on **home**; configs, logs, Slurm scripts, and
**output** live on **scratch** under `RUN_ROOT` (or a user-confirmed folder under
`SCRATCH_ROOT`). **Input CSV data already exists on the HPC at a path the user
provides separately** — it is **not** assumed to be under `RUN_ROOT/data/`.

Default layout on scratch (`RUN_ROOT`):

```
/scratch/yeming/pubchem-mol-filter/
├── run_configs/     # run_config.json, slurm_config.json, filter_config_used.json
├── jobs/            # rendered mol_filter.slurm
├── logs/            # Slurm .out / .err
└── data/
    └── output_shards/   # filtered OUTPUT (default; user may override)
```

**Input is NOT here by default.** The user supplies the input path/dir
(`io.input` / `io.input_dir`) separately — use exactly what they give. Only the
output location defaults under `RUN_ROOT/data/` (and only if the user doesn't
specify one).

| Path key | Default |
|----------|---------|
| Filter script | `/home/yeming/skills/pubchem-mol-filter/scripts/filter_molecules_configurable.py` |
| Render script | `/home/yeming/skills/pubchem-mol-filter/scripts/render_slurm_script.py` |
| Run config | `/scratch/yeming/pubchem-mol-filter/run_configs/run_config.json` |
| Slurm config | `/scratch/yeming/pubchem-mol-filter/run_configs/slurm_config.json` |
| Rendered Slurm | `/scratch/yeming/pubchem-mol-filter/jobs/mol_filter.slurm` |
| Logs | `/scratch/yeming/pubchem-mol-filter/logs/mol-filter-%j.out` (and `.err`) |
| Bashrc | `/home/yeming/.bashrc` |

If the user names a different `RUN_FOLDER` (e.g. `ai4m/mol-filter`), set
`RUN_ROOT=/scratch/yeming/ai4m/mol-filter` and update all scratch paths
accordingly — include in the draft config for user approval.

### I/O paths — do not use slurm_mcp defaults

**All project I/O must come from user-confirmed paths in `run_config.json` and
`slurm_config.json`.** Do **not** infer or substitute slurm_mcp built-in
directory shortcuts or env defaults.

| Do **not** use for this skill | Use instead (from confirmed configs) |
|-------------------------------|--------------------------------------|
| `list_datasets`, `list_model_checkpoints`, `list_job_logs` | `list_directory` on `io.input_dir`, `io.output_dir`, or `slurm.output_log` parent |
| `SLURM_DIR_DATASETS` / `/datasets` mount | `run_config.io.input` / `io.input_dir` (**user-provided HPC path**) |
| `SLURM_DIR_RESULTS` / `/results` mount | `run_config.io.output` / `io.output_dir` |
| `SLURM_DIR_LOGS` / default logs under `SLURM_USER_ROOT` | `slurm.output_log`, `slurm.error_log` |
| `SLURM_USER_ROOT`-relative guesses (`data/`, `results/`, `logs/`) | Full paths under `RUN_ROOT` or user-specified scratch paths |
| `get_cluster_directories` to pick I/O locations | Kestrel paths table + user confirmation |

When calling slurm_mcp file tools, pass **full absolute paths** from the configs:

- `write_file(path=slurm.run_config_path, ...)`
- `write_file(path=<scratch>/run_configs/slurm_config.json, ...)`
- `read_file(path=slurm.rendered_script_path)`
- `read_file(path=slurm.output_log)` (replace `%j` with job id)
- `list_directory(path=run_config.io.output_dir)` or `io.output`

If the user gives a path, use it. If not, apply defaults from this skill
(Kestrel `RUN_ROOT` layout) in the draft config — then include them in the
**whole-config review** for user approval. Never silently use MCP directory
aliases.

## Run config (everything lives in one JSON)

A single run config fully specifies a run, so any agent platform just edits this
file and runs — no other arguments required. It has two blocks:

- **`io`** — input/output paths, mode, workers (execution settings)
- **`filters`** — the RDKit criteria
- **`confirmed`** — must be `true` before execution (set only after user approves all fields)

```json
{
  "confirmed": false,
  "io": {
    "mode": "single",                 // "single" or "shards"
    "input": "/abs/path/input.csv",   // mode=single
    "output": "/abs/path/out.csv",    // mode=single
    "input_dir": null,                 // mode=shards (directory of *.csv)
    "output_dir": null,                // mode=shards
    "workers": 64,
    "save_config_used": "filter_config_used.json"
  },
  "filters": { "...": "see criteria table below" }
}
```

| `io` field | Meaning |
|------------|---------|
| `mode` | `"single"` (one CSV) or `"shards"` (directory of `*.csv`) |
| `input` / `output` | input/output CSV paths (mode=single) |
| `input_dir` / `output_dir` | input/output directories (mode=shards) |
| `workers` | parallel worker processes (default 64) |
| `save_config_used` | path to archive the exact resolved run config before filtering |

Use **absolute paths**. On Kestrel: skill code under `/home/yeming/skills/...`;
data, configs, logs, and outputs under `/scratch/yeming/<run_folder>/...`.

## User confirmation (required before execution)

**Do not execute until the user approves the complete config file(s).** You do
**not** need to ask about each field one-by-one.

### How to confirm

1. **Build draft config(s)** — merge the user prompt with `config_template.json`
   and (on HPC) `slurm_config_template.json`. For any field the user did not
   mention, **keep the template default**.
2. **Ask only when required** — if the user did not provide:
   - `io.input` / `io.input_dir` (input already on HPC; no default path)
   - `slurm.account` if still `YOUR_ACCOUNT` or another placeholder
3. **Present the full JSON** — show `run_config.json` and, on HPC,
   `slurm_config.json` in one summary. Optionally note which values came from
   the user vs defaults.
4. **Wait for approval** — user says e.g. "confirmed", "looks good", "run it", or
   requests specific edits. If they edit, update the config and re-present.
5. **Execute** — set `"confirmed": true` in both files, write to cluster, then run.

Do **not** run with `"confirmed": false`. Do **not** interrogate every filter or
Slurm field when defaults apply.

### `run_config.json` — field reference

| Field | Default / notes |
|-------|-----------------|
| `mode` | `"shards"` in template; use `"single"` if user gives one CSV |
| `input` / `input_dir` | **No default** — user must provide HPC path |
| `output` / `output_dir` | Under `RUN_ROOT/data/` unless user specifies |
| `workers` | `64` (must match `slurm.cpus_per_task` on HPC) |
| `save_config_used` | `RUN_ROOT/run_configs/filter_config_used.json` |
| `filters.*` | See `config_template.json`; override only what user specifies |
| `confirmed` | `false` in draft; set `true` only after user approves the full file |

For `mode == "single"`, `input_dir`/`output_dir` are `null`. For `mode == "shards"`,
`input`/`output` are `null`.

**Input:** CSV/shards already on HPC — user provides `io.input` / `io.input_dir`.
Do not assume input under `RUN_ROOT/data/`.

### `slurm_config.json` — field reference (HPC)

| Field | Default / notes |
|-------|-----------------|
| `job_name` | `pubchem-mol-filter` |
| `account` | **Must be set** — ask if user did not give an account |
| `partition` | `cpu` |
| `nodes` | `1` |
| `cpus_per_task` | `64` — **must match `io.workers`** |
| `time_limit` | `24:00:00` |
| `output_log` / `error_log` | `RUN_ROOT/logs/mol-filter-%j.out` / `.err` |
| `bashrc` | `/home/yeming/.bashrc` |
| `conda_env` | `/scratch/yeming/conda_envs/ai4m` |
| `filter_script` | `/home/yeming/skills/pubchem-mol-filter/scripts/filter_molecules_configurable.py` |
| `run_config_path` | `RUN_ROOT/run_configs/run_config.json` |
| `rendered_script_path` | `RUN_ROOT/jobs/mol_filter.slurm` |
| `confirmed` | `false` in draft; set `true` only after user approves the full file |

Render the batch script **only after** the user approves both configs with
`"confirmed": true`:

```bash
python /home/yeming/skills/pubchem-mol-filter/scripts/render_slurm_script.py \
  --config /scratch/yeming/pubchem-mol-filter/run_configs/slurm_config.json
```

The renderer **blocks** if `confirmed` is false or placeholder paths remain
(same rules as the filter runner). Output is written to `rendered_script_path`.

## Execution via slurm_mcp

Use this path when Claude Code / Cursor has the [slurm_mcp](https://github.com/yidong72/slurm_mcp)
MCP server configured (`SLURM_SSH_HOST`, `SLURM_USER_ROOT`, etc.). The agent runs
on your desktop; all file writes and job submission happen on the cluster over SSH.

**Prerequisites on Kestrel**

- Skill installed at `/home/yeming/skills/pubchem-mol-filter/` (scripts on **home**)
- Conda env with RDKit (`slurm.conda_env`)
- PubChem CSV data under **scratch** (paths in `run_config.io`; confirm with user)
- Run workspace on scratch, default `/scratch/yeming/pubchem-mol-filter/`

**All paths in JSON configs must be cluster absolute paths** (home for skill code,
scratch for run data), not local desktop paths and **not** slurm_mcp default
directory aliases (`/datasets`, `/results`, `/logs`, etc.).

### slurm_mcp workflow (required order)

1. **Confirm** the user has approved both config files (`"confirmed": true`).
2. **`write_file`** — upload confirmed `run_config.json` to `slurm.run_config_path`.
3. **`write_file`** — upload confirmed `slurm_config.json` to a cluster path (audit trail).
4. **`run_shell_command`** — render the Slurm script **on the cluster**:

   ```bash
   python /home/yeming/skills/pubchem-mol-filter/scripts/render_slurm_script.py \
     --config /scratch/yeming/pubchem-mol-filter/run_configs/slurm_config.json
   ```

   Output: `/scratch/yeming/pubchem-mol-filter/jobs/mol_filter.slurm` (or confirmed `rendered_script_path`).

5. **`read_file`** — read the rendered script from `slurm.rendered_script_path`.
6. **`submit_job`** — submit with the file body as `script_content`:

   ```
   submit_job(
     script_content=<exact text from read_file>,
     partition=<slurm.partition>,
     time_limit=<slurm.time_limit>,
     cpus=<slurm.cpus_per_task>,
     account=<slurm.account>,
   )
   ```

   Do **not** use `run_shell_command("sbatch ...")` as the primary path; always
   render → `read_file` → `submit_job` so submission goes through the MCP job tool.

7. **`get_job_details`** — poll until completed; note `job_id` and log paths.
8. **`read_file`** — stdout/stderr logs (`slurm.output_log`, `slurm.error_log`, replace `%j` with job id).
9. **`list_directory`** — verify outputs under `run_config.io.output` or `io.output_dir`.
10. **Report** — pass/reject stats from logs, output paths, job id.

Do **not** call `submit_job` until both configs have `"confirmed": true`.

### slurm_mcp tool map

| Step | MCP tool | Purpose |
|------|----------|---------|
| Upload filter config | `write_file` | Write `run_config.json` to `run_config_path` |
| Upload Slurm config | `write_file` | Write `slurm_config.json` on cluster |
| Render batch script | `run_shell_command` | Run `render_slurm_script.py` on cluster |
| Load rendered script | `read_file` | Read `slurm.rendered_script_path` → `script_content` |
| Submit | `submit_job` | `script_content=<read_file body>` |
| Monitor | `get_job_details`, `list_jobs` | Job state, exit code, log paths |
| Logs | `read_file` | Tail `.out` / `.err` logs |
| Verify outputs | `list_directory` | `run_config.io.output_dir` or `io.output` — **not** `list_datasets` / `/results` |

**Do not use:** `list_datasets`, `list_job_logs`, `get_cluster_directories` for
path discovery on this skill — paths are defined in the confirmed JSON configs.

### `submit_job` (required submission path)

After steps 4–5 (render on cluster, then `read_file` on `mol_filter.slurm`):

```
submit_job(
  script_content=<contents of mol_filter.slurm>,
  partition=<slurm.partition>,
  time_limit=<slurm.time_limit>,
  cpus=<slurm.cpus_per_task>,
  account=<slurm.account>,
)
```

If `submit_job` adds its own `#SBATCH` wrappers, ensure they match the confirmed
`slurm_config.json` values (partition, account, cpus, time). Re-confirm with the
user if the tool overrides differ from what was approved.

### Example agent prompt

> Use **pubchem-mol-filter**: build run + Slurm configs from my request (defaults
> where I didn't specify), show me the full JSON for approval, then via
> **slurm_mcp** write, render, `read_file` the `.slurm`, `submit_job`, and report stats.

## Workflow (required order)

1. **Parse** the user message and merge with `config_template.json` (+ `slurm_config_template.json` on HPC). Use template defaults for unspecified fields.
2. **Ask only for missing required values** — mainly input path (`io.input` / `io.input_dir`) and Slurm account if needed.
3. **Present full config file(s)** for user approval (whole JSON, not field-by-field).
4. **Write configs** with `"confirmed": true` after user approves.
5. **Render Slurm script (HPC)** — on cluster via `run_shell_command`.
6. **Execute via slurm_mcp** — `write_file` → render → `read_file` (`.slurm`) → `submit_job` → monitor logs.
7. **Report results** — pass rate, rejection counts, output paths, job id/logs.

Local-only (no slurm_mcp):

```bash
python "$SKILL_DIR/scripts/filter_molecules_configurable.py" --config run_config.json
```

The filter runner **refuses to execute** if `confirmed` is false or paths contain
placeholders (e.g. `/ABSOLUTE/PATH`). The Slurm renderer uses the same gate.

Optional CLI overrides (merge on top of the config) are available, e.g.
`--input_dir`, `--output_dir`, `--input`, `--output`, `--mode`, `--workers`.
Re-confirm with the user if any change is made after they approved the config.

Do **not** skip the whole-config approval step.

## Filter criteria reference

| User intent | JSON field | Type / notes |
|-------------|------------|--------------|
| No valence / sanitization errors | `require_sanitization` | `true` → RDKit `SanitizeMol` must pass |
| Single connected component | `require_single_component` | `true` → reject multi-fragment SMILES |
| Allowed element types | `allowed_elements` | list of symbols, e.g. `["H","C","N","O"]`; `[]` disables |
| At least one of (composition) | `require_any_elements` | e.g. `["N","O","S","P"]`; `[]` disables |
| Must contain all of | `require_all_elements` | e.g. `["N","S"]`; `[]` disables |
| Forbidden elements | `forbidden_elements` | reject if any present |
| Heavy atom count (non-H) | `max_heavy_atoms` | int upper bound; `null` disables |
| Molecular weight | `max_mol_weight` | float Da upper bound; `null` disables |
| Max ring size | `max_ring_size` | int (largest ring); `null` disables |
| No radicals | `reject_radicals` | `true` rejects radical electrons |
| No isotopes | `reject_isotopes` | `true` rejects labeled isotopes |
| No zwitterion | `reject_zwitterion` | `true` rejects + and − on different atoms |
| Heteroatom lone pair | `require_heteroatom_lone_pair` | N/O/S/P with lone pair; only if `require_any_elements` non-empty |

### Default `filters` block

`config_template.json` defaults: allowed H,C,N,O,S,P,F,Cl,Br,I; require any of
N/O/S/P; max 30 heavy atoms; max ring 6; MW < 500; no radicals, isotopes,
zwitterions; sanitization + single component + heteroatom lone pair. Used when
the user does not specify filter criteria.

Write a fresh full run-config template (`io` + `filters`) to edit:

```bash
python "$SKILL_DIR/scripts/filter_molecules_configurable.py" --write-config run_config.json
```

Write a fresh Slurm config template:

```bash
python "$SKILL_DIR/scripts/render_slurm_script.py" --write-config slurm_config.json
```

## CLI overrides (optional, merge on top of `--config`)

I/O: `--mode single|shards`, `--input`, `--output`, `--input_dir`,
`--output_dir`, `--workers`, `--save-config-used`

Filters:
- `--allowed-elements H,C,N,O`
- `--require-any-elements N,O,S,P` / `--no-require-any-elements`
- `--require-all-elements N` / `--forbidden-elements Fe,Pt`
- `--max-heavy-atoms 30` / `--no-max-heavy-atoms`
- `--max-mol-weight 500` / `--no-max-mol-weight`
- `--max-ring-size 6` / `--no-max-ring-size`
- `--no-sanitization` / `--no-single-component`
- `--allow-radicals` / `--allow-isotopes` / `--allow-zwitterion`
- `--no-heteroatom-lone-pair`

## Input / output format

- **Input CSV columns**: `PUBCHEM_COMPOUND_CID` (or `cid`), `SMILES` (or `smiles`)
- **Output CSV columns**: `PUBCHEM_COMPOUND_CID`, `SMILES`
- Shard mode skips outputs that already exist (resume-friendly)

## Rejection reason keys

`sanitization`, `single_component`, `allowed_elements`, `require_any_elements`,
`require_all_elements`, `forbidden_elements`, `max_heavy_atoms`, `no_radicals`,
`max_ring_size`, `max_mol_weight`, `no_zwitterion`, `no_isotope`,
`heteroatom_lone_pair`, `null_mol`, `no_smiles`

## Agent checklist

- [ ] Kestrel paths: skill on `/home/yeming/skills/...`, run workspace on `/scratch/yeming/<run_folder>/...`
- [ ] **No slurm_mcp default dirs** — I/O from confirmed configs only
- [ ] Draft config(s) built: user values + template defaults
- [ ] Input path provided (asked only if user omitted it)
- [ ] **Full config file(s) approved by user** before execution
- [ ] `"confirmed": true` set; no placeholder paths
- [ ] `cpus_per_task` == `io.workers` (HPC)
- [ ] slurm_mcp: render → `read_file` → `submit_job` → monitor logs
- [ ] Results summarized

## More examples

See [examples.md](examples.md).
