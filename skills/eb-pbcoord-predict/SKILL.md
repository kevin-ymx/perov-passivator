---
name: eb-pbcoord-predict
description: >-
  Predict the binding energy (Eb, in eV) of a Lewis base molecule coordinating to undercoordinated Pb on the FAPbI3 perovskite surface, from the molecule's SMILES, using a trained GIN-E downstream model. Queries may be entered inline by the user (name + SMILES, optional CID) or loaded from a CSV — choose based on the user prompt. Loads a downstream checkpoint plus a finetuned GIN-E encoder checkpoint and runs batched inference, writing a fixed-schema table with `cid`/`smiles` passthrough columns, the predicted binding energy, status, optional sorting by Eb, and an optional strong-binder threshold flag. Use when the user asks to predict binding energy, adsorption energy, Eb, Pb-coordination/Pb-binding energy, Lewis base–Pb binding on FAPbI3, or to rank/screen passivator molecules by how strongly they coordinate to surface Pb on FAPbI3. Always build a run config from the user prompt and present it for approval before running; execution is blocked until "confirmed": true. Runs as a direct `python` CLI command — locally or on an HPC node — where the model checkpoints and PyTorch/RDKit env are available.
---

# Lewis base–Pb binding-energy (Eb) on FAPbI3

Self-contained skill for predicting the binding energy of a Lewis base molecule coordinating to undercoordinated **Pb on the FAPbI3 perovskite surface**, from the molecule's SMILES. For molecules the **user provides** (inline or via CSV), it runs a trained GIN-E downstream regression model and returns the predicted Pb–molecule binding energy on FAPbI3 in **eV** per molecule (more negative = stronger coordination / better surface passivation). All orchestration code lives in `scripts/`; the skill loads the model implementation from a user-specified backend path (`io.project_root`).

## When to use

The user wants the predicted Pb-surface binding energy (Eb) on **FAPbI3** of one or more Lewis base / passivator molecules — e.g. "predict how strongly these molecules coordinate to surface Pb on FAPbI3" or "rank these analogs by Pb-binding energy on FAPbI3". Pick the query source from the prompt:

- **Inline** — the user lists molecules (SMILES, optionally a name/CID) directly.
- **CSV** — the user points to a file of molecules.

## Prerequisites

- Python 3.8+ with `torch`, `torch-geometric`, `rdkit`, `numpy`, `tqdm` (the model backend's environment).
- A trained **downstream checkpoint** (`model.downstream_checkpoint`, e.g. `downstream_best_model.pt`).
- A finetuned **GIN-E encoder checkpoint** (`model.gin_e_checkpoint`, e.g. `gin_e_finetuned.pt`).
- A **model backend** at `io.project_root` — the codebase that provides `inference_Eb.py`, `config.py`, and `models/` (required when the skill is not co-located with that code; see [HPC execution](#hpc-execution)).
- A GPU is recommended for large batches; CPU works for small lists (`model.device: "cpu"`).

## Files in this skill

```
eb-pbcoord-predict/
├── SKILL.md                 # this file
├── examples.md              # prompt → config examples
├── requirements.txt         # runtime notes (backend deps via project_root)
├── config_template.json     # run config (io + model)
└── scripts/
    └── eb_pbcoord_predict.py   # config-driven runner
```

`SKILL_DIR` below refers to this skill's directory. Run as a `python` CLI command locally **or** on an HPC node (see [HPC execution](#hpc-execution)).

## Query input: inline OR CSV (not both)

Each query needs a **SMILES** (the model predicts from SMILES); `name` and `cid` are optional labels passed through to the output. Provide exactly one source:

**Inline** (`io.queries`):

```json
"queries": [
  {"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null},
  {"name": "Ethylenediamine", "smiles": "NCCN", "cid": 3301}
]
```

**CSV** (`io.query_csv` + column names): set `io.queries` to `[]` and point `query_csv` at the file; `name_column` / `cid_column` / `smiles_column` select the columns (SMILES required).

## Run config (one JSON specifies the whole run)

```json
{
  "confirmed": false,
  "io": {
    "queries": [{"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null}],
    "query_csv": null,
    "name_column": "molecule_name",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "project_root": null,
    "output_csv": "eb_pbcoord_predictions.csv",
    "sort_by_energy": true
  },
  "model": {
    "downstream_checkpoint": "/abs/path/downstream_best_model.pt",
    "gin_e_checkpoint": "/abs/path/gin_e_finetuned.pt",
    "device": "auto",
    "batch_size": 64,
    "energy_threshold": null
  }
}
```

| `io` field | Meaning |
|------------|---------|
| `queries` | Inline list of `{name, smiles, cid}` (SMILES required); empty when using CSV |
| `query_csv` | Path to a query CSV (null when using inline) |
| `name_column` / `cid_column` / `smiles_column` | Column names in `query_csv` |
| `project_root` | Root of the model backend (`null` = auto-detect when skill and backend share a tree; **set explicitly** when installed separately, e.g. on HPC) |
| `output_csv` | Prediction table (one row per molecule) |
| `sort_by_energy` | Sort output ascending by predicted Eb (most negative = strongest binder); failures last |

| `model` field | Meaning |
|---------------|---------|
| `downstream_checkpoint` | Trained downstream regression model — required |
| `gin_e_checkpoint` | Finetuned GIN-E encoder checkpoint — required |
| `device` | `auto` / `cuda` / `cpu` |
| `batch_size` | Molecules per inference batch (default 64) |
| `energy_threshold` | If set (eV), adds a `below_threshold` flag for rows with Eb ≤ threshold |

## User confirmation (required before execution)

**Do not execute until the user approves the complete config.**

1. **Build a draft config** — merge the user prompt with `config_template.json`; keep template defaults for unspecified fields.
2. **Ask only for missing required values** — `downstream_checkpoint`, `gin_e_checkpoint`, `project_root` (if not auto-detectable), and the query source (inline molecules or `query_csv`).
3. **Present the full JSON** for approval (whole config, not field-by-field).
4. **Wait for approval** ("confirmed", "looks good", "run it", or edits).
5. **Execute** — set `"confirmed": true`, then run.

The runner refuses to execute while `confirmed` is `false` or a required path contains a placeholder.

## Workflow (required order)

1. **Determine the query source** from the prompt (inline vs CSV).
2. **Build draft config** (checkpoints, query source, optional threshold).
3. **Present the full config** for user approval.
4. **Run** (after approval, with `"confirmed": true`):

```bash
python "$SKILL_DIR/scripts/eb_pbcoord_predict.py" --config run_config.json
```

5. **Report** — output path, #success / #failed, and (if a threshold was set) the number of strong binders; offer downstream use of the table (e.g. vendor lookup by `cid`/`smiles`).

### Scaffold a fresh config

```bash
python "$SKILL_DIR/scripts/eb_pbcoord_predict.py" --write-config run_config.json
```

### Optional CLI overrides (merge on top of `--config`)

`--smiles "S1,S2"` (quick inline), `--query-csv`, `--downstream-checkpoint`, `--gin-e-checkpoint`, `--project-root`, `--output-csv`, `--no-sort`, `--batch-size`, `--energy-threshold`, `--device`, `--confirmed`.

## HPC execution

No Slurm batch job is needed — run the same `python` CLI command directly on an HPC node (an interactive GPU session, e.g. via `salloc`/`srun`, or a login node with the env loaded). The only difference from local is absolute paths and activating the cluster env first.

### Kestrel paths (defaults)

| Symbol | Path | Purpose |
|--------|------|---------|
| `SKILL_DIR` | `/home/yeming/skills/eb-pbcoord-predict` | Skill code (home) |
| `BACKEND_ROOT` | `/home/yeming/perov-passivator` (example) | Model backend (`io.project_root`) |
| `RUN_ROOT` | `/scratch/yeming/eb-pbcoord-predict` | Run workspace (scratch) |

When the skill lives under `home/skills/` (separate from the backend), auto-detection of `project_root` will fail. **Set `io.project_root` to `BACKEND_ROOT`** in the run config. Checkpoints are user-provided absolute paths — do not assume they live under `RUN_ROOT`.

### Run it (after the config is confirmed)

```bash
# 1. Load the cluster env
source /home/yeming/.bashrc
conda activate /scratch/yeming/conda_envs/ai4m

# 2. Run the same CLI as local (config has io.project_root + absolute paths)
python /home/yeming/skills/eb-pbcoord-predict/scripts/eb_pbcoord_predict.py \
  --config /scratch/yeming/eb-pbcoord-predict/run_configs/run_config.json
```

For a GPU node, grab one interactively first, e.g. `salloc -A <account> -p gpu --gpus=1 -t 02:00:00` (or `srun ... --pty bash`), then run the command above. The `confirmed`/placeholder gate is identical to local.

## Output format

**`output_csv`** — one row per molecule:

```
name, cid, smiles, predicted_binding_energy_eV, prediction_status
```

- `cid` / `smiles` use lowercase headers — compatible with CSV-based vendor lookup skills that expect those column names.
- `predicted_binding_energy_eV` is empty when prediction fails; `prediction_status` carries `OK` or the error message.
- When `model.energy_threshold` is set, an extra `below_threshold` column flags strong binders (Eb ≤ threshold).
- With `io.sort_by_energy: true`, rows are sorted ascending by Eb (strongest binders first); failed rows go last.

## Downstream use

The prediction table is a ranked candidate list with unique `cid`/`smiles` pairs and predicted binding energy. Pass it to any workflow that needs molecule identifiers — e.g. physical-form and salt-vendor lookup — using the `cid` and `smiles` columns as input. A natural pipeline:

```
ssl-neighbor-search  →  eb-pbcoord-predict  →  mol-salt-vendor
   (find analogs)        (rank by Eb)            (buyable forms + vendors)
```

## Notes & limits

- Predicts from **SMILES**; queries without a SMILES are skipped.
- Predicts the binding energy of the molecule (as a Lewis base) coordinating to undercoordinated Pb on the **FAPbI3** perovskite surface, reported in **eV** (more negative = stronger coordination / better passivation).
- Requires **two** checkpoints trained for this FAPbI3 Lewis base–Pb binding task: the downstream regression head and the finetuned GIN-E encoder.
- Runtime scales with the number of molecules and `batch_size`; use a GPU for large lists.

## Agent checklist

- [ ] Query source decided (inline vs CSV) and SMILES present for each query
- [ ] `downstream_checkpoint`, `gin_e_checkpoint`, and `project_root` (if needed) provided and valid
- [ ] Backend env available (torch / PyG / RDKit) where this runs
- [ ] Full config approved by user; `"confirmed": true`; no placeholder paths
- [ ] **HPC:** `io.project_root` set to backend path; env activated; run as a direct `python ... --config` CLI command
- [ ] Run executed; prediction table + #success/#failed reported
- [ ] Offered downstream use of output CSV if relevant

## More examples

See [examples.md](examples.md).
