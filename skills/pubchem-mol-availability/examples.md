# PubChem salt-form / form / availability — examples

**Kestrel paths:** skill on `/home/yeming/skills/pubchem-mol-availability`; run
workspace (configs, caches, logs, jobs, output) on
`/scratch/yeming/pubchem-mol-availability/`. **Input CSV already exists on the
HPC — the user provides that path separately.** **Do not** use slurm_mcp default
directories — use only paths from confirmed `run_config.json` /
`slurm_config.json`.

Each run is fully specified by `run_config.json` (+ `slurm_config.json` on HPC).

**Before running:** present the **complete** config(s) for approval, using
template defaults for unspecified fields. Set `"confirmed": true` and execute
only after approval.

## Input format

CSV with neutral molecules (header flexible):

```
PUBCHEM_COMPOUND_CID,SMILES
6342,CCN
1146,CC(=O)Nc1ccc(O)cc1
```

(`cid` / `CID` and `smiles` are also accepted.) For other headers, set
`io.cid_column` / `io.smiles_column` in `run_config.json` to the exact column
names; leave them `null` to auto-detect.

## Confirmation example (required step)

User: *For the molecules in `/projects/ai4m/neutral_mols.csv`, check halide-salt
forms, powder vs liquid, and whether purchasable; drop liquids. Kestrel, account
m3342, via slurm_mcp.*

Agent builds configs (defaults for lookup/llm/filter, workers=64, output under
the run folder), then presents the **full JSON** (not field-by-field):

> Here are the draft configs. Unspecified fields use template defaults. Please
> confirm or tell me what to change:
>
> **`run_config.json`** — (full JSON shown)
> **`slurm_config.json`** — (full JSON shown)

Only after the user says "confirmed" → set `"confirmed": true` → slurm_mcp workflow.

If the user had **not** given the input path or account, ask for those **only**,
then include them in the draft before presenting for approval.

## Example: shard batch via slurm_mcp (Kestrel)

**`run_config.json`:**

```json
{
  "confirmed": true,
  "io": {
    "input": "/projects/ai4m/neutral_mols.csv",
    "output": "/scratch/yeming/pubchem-mol-availability/data/availability.csv",
    "dropped_output": "/scratch/yeming/pubchem-mol-availability/data/dropped_liquids.csv",
    "cache_dir": "/scratch/yeming/pubchem-mol-availability/cache",
    "workers": 64
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

**`slurm_config.json`:**

```json
{
  "confirmed": true,
  "job_name": "pubchem-mol-availability",
  "account": "m3342",
  "partition": "short",
  "nodes": 1,
  "cpus_per_task": 64,
  "time_limit": "24:00:00",
  "output_log": "/scratch/yeming/pubchem-mol-availability/logs/mol-availability-%j.out",
  "error_log": "/scratch/yeming/pubchem-mol-availability/logs/mol-availability-%j.err",
  "bashrc": "/home/yeming/.bashrc",
  "secrets_file": "/home/yeming/.secrets/openai",
  "conda_env": "/scratch/yeming/conda_envs/ai4m",
  "run_script": "/home/yeming/skills/pubchem-mol-availability/scripts/run_availability.py",
  "run_config_path": "/scratch/yeming/pubchem-mol-availability/run_configs/run_config.json",
  "rendered_script_path": "/scratch/yeming/pubchem-mol-availability/jobs/mol_availability.slurm"
}
```

**slurm_mcp sequence:**

```
1. write_file(/scratch/yeming/pubchem-mol-availability/run_configs/run_config.json, ...)
2. write_file(/scratch/yeming/pubchem-mol-availability/run_configs/slurm_config.json, ...)
3. run_shell_command(
     "python /home/yeming/skills/pubchem-mol-availability/scripts/render_slurm_script.py "
     "--config /scratch/yeming/pubchem-mol-availability/run_configs/slurm_config.json"
   )
4. read_file(/scratch/yeming/pubchem-mol-availability/jobs/mol_availability.slurm)
5. submit_job(script_content=<step 4>, partition="short", account="m3342",
              time_limit="24:00:00", cpus=64)
6. get_job_details(job_id=...)
7. read_file(/scratch/yeming/pubchem-mol-availability/logs/mol-availability-<id>.out, tail_lines=200)
8. list_directory(/scratch/yeming/pubchem-mol-availability/data)
```

## Example output row

Input `6342, CCN` (ethylamine, a neutral free base) → PubChem finds its
hydrochloride salt; the salt is a purchasable powder, so the row is kept:

```
input_cid,input_smiles,has_halide_salt,salt_cid,salt_smiles,salt_counterion,salt_physical_form,salt_n_vendors,salt_purchasable,salt_vendor_examples,parent_physical_form,parent_n_vendors,parent_purchasable,form_source,form_confidence,kept
6342,CCN,True,..., CC[NH3+].[Cl-],Cl,powder,7,True,VWR; Sigma,unknown,2,True,pubchem,,True
```

A neutral molecule that is a liquid with no halide salt is written to
`dropped_liquids.csv` instead (`kept=False`).

## Prompt -> config mapping

| User phrase | Config field |
|-------------|--------------|
| input CSV already on HPC | `io.input` (**user-provided path**) |
| custom CID / SMILES header names | `io.cid_column` / `io.smiles_column` (else auto-detect) |
| output / dropped output location | `io.output` / `io.dropped_output` (default under `RUN_ROOT/data/`) |
| "halide salts only" / "any salt" | `lookup.halide_only` (`true` / `false`) |
| "also report the neutral form" | `lookup.report_neutral` |
| "one row per salt" | `lookup.row_granularity = per_salt` |
| MP threshold for solid | `lookup.mp_solid_threshold_c` |
| LLM model / off | `llm.model` / `llm.enabled` |
| LLM key location | `slurm.secrets_file` (env `llm.api_key_env`) |
| "drop liquids" | `filter.drop_liquids` |
| Slurm account | `slurm.account` |
| skill script location | fixed: `/home/yeming/skills/pubchem-mol-availability/scripts/...` |
