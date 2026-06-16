---
name: pubchem-mol-availability
description: >-
  For a batch of neutral molecules (cid, smiles), look up via PubChem whether
  each has a halide-salt form (typically HCl/HBr), whether the salt and/or
  neutral form is a powder vs liquid, and whether it is purchasable (vendor
  count). Resolves physical form with a cascade: PubChem text -> melting-point
  heuristic -> LLM fallback (powder/liquid), then drops confirmed-liquid rows.
  Use when the user asks to check salt forms, powder/liquid form, or
  purchasability/vendor availability for PubChem molecules. Configure from the
  user prompt and present the full run_config.json (+ slurm_config.json on HPC)
  for approval before execution. On Kestrel: skill on
  /home/yeming/skills/pubchem-mol-availability, run data on
  /scratch/yeming/pubchem-mol-availability. Runs as one end-to-end Slurm job
  (compute nodes have internet); LLM API key sourced from a private $HOME file.
  Do not use slurm_mcp default directories; use confirmed config paths only.
disable-model-invocation: true
---

# PubChem molecule salt-form / form / availability lookup

Self-contained, portable skill. For each **neutral input molecule** it answers
three questions using **PubChem** (free, no API key for lookups):

1. **Does it have a halide-salt form?** (e.g. the hydrochloride of an amine) —
   found via PubChem *same-parent connectivity* related CIDs, then classified
   with RDKit (multi-component + F/Cl/Br/I counterion).
2. **Powder vs liquid?** — resolved per CID with a cascade:
   PubChem "Physical Description"/"Color/Form" → **melting-point heuristic** →
   **LLM** fallback (`solid`/`liquid`). Unknown stays unknown.
3. **Is it purchasable?** — PubChem "Chemical Vendors" count (`> 0` ⇒ purchasable).

Rows whose decision form is confirmed **`liquid`** are **dropped** (kept rows →
`output`; dropped rows → `dropped_output`). `unknown` is **kept**.

> The input molecules are all neutral. This skill does **not** test whether the
> input itself is a salt — it looks up whether a **salt form of it exists**.

## When to use

The user provides a CSV of neutral molecules (`cid`, `smiles`) and wants salt
form / powder-vs-liquid / purchasability annotations, with liquids removed.

## Prerequisites

- Python 3.8+ with `rdkit`, `requests`, `pandas`, `tqdm`, `openai`
  (`pip install -r requirements.txt`).
- Outbound internet (PubChem REST + LLM API). On Kestrel, **compute nodes have
  internet**, so the whole pipeline runs in one Slurm job.
- LLM API key in env (`OPENAI_API_KEY` by default), sourced from a private
  `$HOME` file by the Slurm script. If unavailable, the LLM step is skipped and
  those molecules stay `unknown` (the run still completes).

## Files in this skill

```
pubchem-mol-availability/
├── SKILL.md                       # this file
├── examples.md                    # prompt -> config examples
├── requirements.txt               # rdkit, requests, pandas, tqdm, openai
├── config_template.json           # run config (io + lookup + llm + filter)
├── slurm_config_template.json     # HPC Slurm settings (single end-to-end job)
└── scripts/
    ├── availability_core.py            # configs, salt detection, form/MP logic, cascade
    ├── pubchem_client.py               # PUG REST/View + LLM clients, throttle + cache
    ├── run_availability.py             # config-driven runner (online + offline in one pass)
    ├── submit_availability.slurm.template  # Slurm script template ({{PLACEHOLDERS}})
    └── render_slurm_script.py          # render template from slurm_config.json
```

## Kestrel paths

| Symbol | Path | Purpose |
|--------|------|---------|
| `HOME_ROOT` | `/home/yeming` | Home directory |
| `SCRATCH_ROOT` | `/scratch/yeming` | Scratch / job workspace root |
| `SKILL_DIR` | `/home/yeming/skills/pubchem-mol-availability` | Skill install (code) — **home** |
| `RUN_ROOT` | `/scratch/yeming/pubchem-mol-availability` | Run workspace — **scratch** |

**Rule:** skill code on **home**; configs, caches, logs, jobs, and **output** on
**scratch** under `RUN_ROOT`. The **input CSV already exists on the HPC at a path
the user provides** — it is not assumed under `RUN_ROOT/data/`.

Default layout on scratch:

```
/scratch/yeming/pubchem-mol-availability/
├── run_configs/   # run_config.json, slurm_config.json
├── jobs/          # rendered mol_availability.slurm
├── logs/          # Slurm .out / .err
├── cache/         # pubchem_cache.json, llm_cache.json (idempotent re-runs)
└── data/
    ├── availability.csv      # kept rows (OUTPUT)
    └── dropped_liquids.csv   # rows dropped as liquid
```

### I/O paths — do not use slurm_mcp defaults

All project I/O must come from confirmed paths in `run_config.json` /
`slurm_config.json`. Do **not** use slurm_mcp built-in directory shortcuts
(`/datasets`, `/results`, `/logs`, `list_datasets`, `get_cluster_directories`,
etc.). Pass **full absolute paths** from the configs to all MCP file tools.

## Run config (one JSON fully specifies a run)

```json
{
  "confirmed": false,
  "io": {
    "input": "/abs/path/molecules.csv",
    "output": "/scratch/yeming/pubchem-mol-availability/data/availability.csv",
    "dropped_output": "/scratch/yeming/pubchem-mol-availability/data/dropped_liquids.csv",
    "cache_dir": "/scratch/yeming/pubchem-mol-availability/cache",
    "workers": 64,
    "cid_column": null,
    "smiles_column": null
  },
  "lookup": {
    "request_rate_per_sec": 5.0,
    "max_retries": 4,
    "timeout_sec": 30,
    "halide_only": true,
    "report_neutral": true,
    "row_granularity": "per_salt",
    "mp_solid_threshold_c": 25.0
  },
  "llm": {
    "enabled": true,
    "model": "gpt-5.5",
    "temperature": 0.0,
    "confidence_threshold": 0.7,
    "api_key_env": "OPENAI_API_KEY",
    "base_url": null,
    "max_retries": 3
  },
  "filter": { "drop_liquids": true }
}
```

| Field | Meaning / default |
|-------|-------------------|
| `io.input` | **No default** — user-provided CSV (`cid`/`PUBCHEM_COMPOUND_CID`, `smiles`) |
| `io.output` / `io.dropped_output` | kept rows / dropped-liquid rows (default under `RUN_ROOT/data/`) |
| `io.cache_dir` | PubChem + LLM caches (idempotent/resumable); default `RUN_ROOT/cache` |
| `io.cid_column` / `io.smiles_column` | `null` → auto-detect (`PUBCHEM_COMPOUND_CID`/`cid`/`CID`, `SMILES`/`smiles`); set to a header name for custom CSVs |
| `lookup.request_rate_per_sec` | PubChem throttle (≤ ~5/s; PubChem caps at 5/s, 400/min) |
| `lookup.halide_only` | `true` → count only F/Cl/Br/I salts as a "salt form" |
| `lookup.report_neutral` | `true` → also enrich/report the neutral input |
| `lookup.row_granularity` | `per_salt` → one row per halide-salt CID |
| `lookup.mp_solid_threshold_c` | MP (°C) above which a compound is treated as solid (default 25) |
| `llm.enabled` | `true` → LLM fallback when PubChem text + MP are inconclusive |
| `llm.model` | `gpt-5.5` (GPT-5 series flagship); use `gpt-5.5-mini` for lower cost/latency |
| `llm.temperature` / `confidence_threshold` | LLM call params; low-confidence → `unknown` |
| `llm.api_key_env` | env var holding the API key (**never** the key itself) |
| `filter.drop_liquids` | `true` → drop rows whose decision form is `liquid` (`unknown` kept) |

## Output columns

`availability.csv` (kept) and `dropped_liquids.csv` (dropped) share columns:

```
input_cid, input_smiles, has_halide_salt,
salt_cid, salt_smiles, salt_counterion,
salt_physical_form, salt_n_vendors, salt_purchasable, salt_vendor_examples,
parent_physical_form, parent_n_vendors, parent_purchasable,
form_source, form_confidence, kept
```

- One row **per halide-salt CID**; molecules with no halide salt get one
  neutral-only row (salt columns blank, `has_halide_salt=false`).
- `form_source` ∈ `pubchem` / `mp_heuristic` / `llm` / `none`; `form_confidence`
  is set only for `llm`. The decision form (used for the liquid filter) is the
  salt form on salt rows, else the parent form.

**Caveats:** "no halide salt found" means none registered in PubChem (not proof
none can exist). `purchasable` = listed by a PubChem vendor (triage, not
in-stock). Many CIDs lack a Physical Description → `unknown` is common.

## User confirmation (required before execution)

Do **not** execute until the user approves the complete config(s). Do **not**
interrogate every field — merge the user prompt with `config_template.json`
(+ `slurm_config_template.json` on HPC) and keep template defaults for the rest.

1. **Build draft config(s)** from the user request + template defaults.
2. **Ask only for missing required values:** `io.input` (no default) and, on HPC,
   `slurm.account` (if still `YOUR_ACCOUNT`).
3. **Present the full JSON** for approval (whole file, not field-by-field).
4. **Set `"confirmed": true`** in both files only after approval, then run.

The runner and the Slurm renderer both **refuse to execute** if `confirmed` is
false or any path still contains a placeholder (e.g. `/REPLACE`).

## slurm_config.json — field reference (HPC)

| Field | Default / notes |
|-------|-----------------|
| `job_name` | `pubchem-mol-availability` |
| `account` | **Must be set** — ask if user did not give an account |
| `partition` | `short` |
| `nodes` | `1` |
| `cpus_per_task` | `64` (offline RDKit phase; PubChem stays rate-limited regardless) |
| `time_limit` | `24:00:00` (size for the **network/rate-limited** phase, not CPU) |
| `output_log` / `error_log` | `RUN_ROOT/logs/mol-availability-%j.out` / `.err` |
| `bashrc` | `/home/yeming/.bashrc` |
| `secrets_file` | `/home/yeming/.secrets/openai` — sourced to export the API key; `null` to skip |
| `conda_env` | `/scratch/yeming/conda_envs/ai4m` |
| `run_script` | `SKILL_DIR/scripts/run_availability.py` |
| `run_config_path` | `RUN_ROOT/run_configs/run_config.json` |
| `rendered_script_path` | `RUN_ROOT/jobs/mol_availability.slurm` |
| `confirmed` | `false` in draft; `true` only after approval |

The rendered Slurm script runs one end-to-end job:
`source bashrc` → `source secrets_file` → `conda activate` → `run_availability.py`.

**Secrets:** keep the API key in `secrets_file` (e.g. `~/.secrets/openai`,
`chmod 600`, containing `export OPENAI_API_KEY=...`). Never put the key in
`run_config.json` or on scratch.

## Execution via slurm_mcp

1. **Confirm** both configs approved (`"confirmed": true`).
2. **`write_file`** — `run_config.json` → `run_config_path`.
3. **`write_file`** — `slurm_config.json` → cluster path (audit trail).
4. **`run_shell_command`** — render on the cluster:

   ```bash
   python /home/yeming/skills/pubchem-mol-availability/scripts/render_slurm_script.py \
     --config /scratch/yeming/pubchem-mol-availability/run_configs/slurm_config.json
   ```

5. **`read_file`** — read `rendered_script_path`.
6. **`submit_job`** — `script_content=<read_file body>`, `partition`,
   `time_limit`, `cpus`, `account` from `slurm_config.json`.
7. **`get_job_details`** — poll until complete.
8. **`read_file`** — logs (`output_log` / `error_log`, replace `%j`).
9. **`list_directory`** — verify `io.output` / `io.dropped_output`.

Do **not** use `sbatch` via shell as the primary path; always render →
`read_file` → `submit_job`.

## Local / single-node run (no slurm_mcp)

```bash
# Write a fresh run-config template, then edit it:
python scripts/run_availability.py --write-config run_config.json

# Run from the config (export the LLM key first if llm.enabled):
export OPENAI_API_KEY=...   # or: source ~/.secrets/openai
python scripts/run_availability.py --config run_config.json
```

## Workflow (required order)

1. **Parse** the user message; merge with `config_template.json` (+ slurm template on HPC).
2. **Ask only for missing required values** — `io.input`, and `slurm.account` on HPC.
3. **Present full config file(s)** for approval (whole JSON).
4. **Write configs** with `"confirmed": true` after approval.
5. **Render** the Slurm script on the cluster (HPC).
6. **Execute** via slurm_mcp: `write_file` → render → `read_file` → `submit_job` → monitor.
7. **Report** kept/dropped counts, salt-form hits, purchasable counts, output paths, job id.

## Agent checklist

- [ ] Kestrel paths: skill on `/home/yeming/skills/...`, run data on `/scratch/yeming/...`
- [ ] **No slurm_mcp default dirs** — I/O from confirmed configs only
- [ ] Draft config(s) built: user values + template defaults
- [ ] Input CSV path provided (asked only if user omitted it)
- [ ] **Full config file(s) approved** before execution; `"confirmed": true`; no placeholders
- [ ] LLM key via `secrets_file` (not in configs/scratch)
- [ ] `time_limit` sized for the network-bound lookup phase
- [ ] slurm_mcp: render → `read_file` → `submit_job` → monitor logs
- [ ] Results summarized

## More examples

See [examples.md](examples.md).
