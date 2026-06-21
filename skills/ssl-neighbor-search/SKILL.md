---
name: ssl-neighbor-search
description: >-
  Find the nearest neighbors of user-specified molecules in a GIN-E SSL embedding space. Queries may be entered inline by the user (name + SMILES, optional CID) or loaded from a CSV — choose based on the user prompt. Computes query embeddings with a trained checkpoint and runs k-NN over a directory of reference embedding CSVs, writing a long-form neighbor table and a deduplicated table with `cid`/`smiles` columns suitable for downstream vendor lookup. Use when the user asks to find similar molecules, SSL neighbors, nearest neighbors, embedding neighbors, or analogs of given molecules. Always build a run config from the user prompt and present it for approval before running; execution is blocked until "confirmed": true. Runs as a direct `python` CLI command — locally or on an HPC node — where the encoder backend, checkpoint, embeddings, and PyTorch/RDKit env are available.
---

# SSL embedding nearest-neighbor search

Self-contained skill for k-nearest-neighbor search in a fixed GIN-E embedding space. For molecules the **user provides** (inline or via CSV), it embeds queries with a trained checkpoint and returns the `k` closest molecules from a precomputed reference set. All orchestration code lives in `scripts/`; the skill loads the encoder implementation from a user-specified backend path (`io.project_root`).

## When to use

The user wants embedding-space neighbors (analogs) of one or more molecules — e.g. "find molecules similar to phenethylamine and EDA in SSL space". Pick the query source from the prompt:

- **Inline** — the user lists molecules (SMILES, optionally a name/CID) directly.
- **CSV** — the user points to a file of molecules.

## Prerequisites

- Python 3.8+ with `torch`, `torch-geometric`, `rdkit`, `numpy`, `scipy`, `tqdm` (install from the encoder backend's environment, or `pip install` the packages listed in `requirements.txt`).
- A trained GIN-E **checkpoint** (`io.checkpoint`).
- A **reference embedding directory** (`io.embedding_dir`): CSV shards, each with `emb_0..emb_255` plus `cid` and `smiles`.
- An **encoder backend** at `io.project_root` — the codebase that provides the GIN-E model and embedding utilities (required when the skill is not co-located with that code; see [HPC execution](#hpc-execution)).
- A GPU is strongly recommended for embedding many queries.

## Files in this skill

```
ssl-neighbor-search/
├── SKILL.md                 # this file
├── examples.md              # prompt → config examples
├── requirements.txt         # runtime notes (encoder deps via project_root)
├── config_template.json     # run config (io + knn)
└── scripts/
    └── ssl_neighbor_search.py   # config-driven runner
```

`SKILL_DIR` below refers to this skill's directory. Run as a `python` CLI command locally **or** on an HPC node (see [HPC execution](#hpc-execution)).

## Query input: inline OR CSV (not both)

Each query needs a **SMILES** (the encoder embeds by SMILES); `name` and `cid` are optional labels passed through to the output. Provide exactly one source:

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
    "embedding_dir": "/abs/path/reference_embeddings",
    "checkpoint": "/abs/path/checkpoints/best_model.pt",
    "project_root": null,
    "output_csv": "ssl_neighbors.csv",
    "output_dedup_csv": "ssl_neighbors_dedup.csv",
    "write_dedup": true
  },
  "knn": {
    "k": 21,
    "chunk_size": 100000,
    "device": "auto",
    "drop_query_isotopes": true
  }
}
```

| `io` field | Meaning |
|------------|---------|
| `queries` | Inline list of `{name, smiles, cid}` (SMILES required); empty when using CSV |
| `query_csv` | Path to a query CSV (null when using inline) |
| `name_column` / `cid_column` / `smiles_column` | Column names in `query_csv` |
| `embedding_dir` | Directory of reference embedding CSVs (`emb_0..emb_255`) — required |
| `checkpoint` | Trained GIN-E checkpoint — required |
| `project_root` | Root of the encoder backend (`null` = auto-detect when skill and backend share a tree; **set explicitly** when installed separately, e.g. on HPC) |
| `output_csv` | Long-form table: one row per (query, neighbor) |
| `output_dedup_csv` | Deduplicated table: one row per unique neighbor |
| `write_dedup` | Whether to also write the deduplicated table |

| `knn` field | Meaning |
|-------------|---------|
| `k` | Neighbors per query (default 21) |
| `chunk_size` | Reference rows per chunk when streaming CSVs (default 100000) |
| `device` | `auto` / `cuda` / `cpu` |
| `drop_query_isotopes` | Drop query SMILES with isotope labels (e.g. `[13C]`) |

## User confirmation (required before execution)

**Do not execute until the user approves the complete config.**

1. **Build a draft config** — merge the user prompt with `config_template.json`; keep template defaults for unspecified fields.
2. **Ask only for missing required values** — `embedding_dir`, `checkpoint`, `project_root` (if not auto-detectable), and the query source (inline molecules or `query_csv`).
3. **Present the full JSON** for approval (whole config, not field-by-field).
4. **Wait for approval** ("confirmed", "looks good", "run it", or edits).
5. **Execute** — set `"confirmed": true`, then run.

The runner refuses to execute while `confirmed` is `false` or a required path contains a placeholder.

## Workflow (required order)

1. **Determine the query source** from the prompt (inline vs CSV).
2. **Build draft config** (paths, query source, `k`).
3. **Present the full config** for user approval.
4. **Run** (after approval, with `"confirmed": true`):

```bash
python "$SKILL_DIR/scripts/ssl_neighbor_search.py" --config run_config.json
```

5. **Report** — output paths, #queries embedded, #unique neighbors, and offer downstream use of the dedup CSV (e.g. vendor lookup by `cid`/`smiles`).

### Scaffold a fresh config

```bash
python "$SKILL_DIR/scripts/ssl_neighbor_search.py" --write-config run_config.json
```

### Optional CLI overrides (merge on top of `--config`)

`--smiles "S1,S2"` (quick inline), `--query-csv`, `--embedding-dir`, `--checkpoint`, `--project-root`, `--output-csv`, `--output-dedup-csv`, `--no-dedup`, `-k/--k`, `--chunk-size`, `--device`, `--confirmed`.

## HPC execution

No Slurm batch job is needed — run the same `python` CLI command directly on an HPC node (an interactive GPU session, e.g. via `salloc`/`srun`, or a login node with the env loaded). The only difference from local is absolute paths and activating the cluster env first.

### Kestrel paths (defaults)

| Symbol | Path | Purpose |
|--------|------|---------|
| `SKILL_DIR` | `/home/yeming/skills/ssl-neighbor-search` | Skill code (home) |
| `BACKEND_ROOT` | `/home/yeming/gin-e-encoder` (example) | Encoder backend (`io.project_root`) |
| `RUN_ROOT` | `/scratch/yeming/ssl-neighbor-search` | Run workspace (scratch) |

When the skill lives under `home/skills/` (separate from the encoder backend), auto-detection of `project_root` will fail. **Set `io.project_root` to `BACKEND_ROOT`** in the run config. Checkpoint and reference embeddings are user-provided absolute paths — do not assume they live under `RUN_ROOT`.

### Run it (after the config is confirmed)

```bash
# 1. Load the cluster env
source /home/yeming/.bashrc
conda activate /scratch/yeming/conda_envs/ai4m

# 2. Run the same CLI as local (config has io.project_root + absolute paths)
python /home/yeming/skills/ssl-neighbor-search/scripts/ssl_neighbor_search.py \
  --config /scratch/yeming/ssl-neighbor-search/run_configs/run_config.json
```

For a GPU node, grab one interactively first, e.g. `salloc -A <account> -p gpu --gpus=1 -t 04:00:00` (or `srun ... --pty bash`), then run the command above. The `confirmed`/placeholder gate is identical to local.

## Output format

**`output_csv`** (long form) — one row per query–neighbor pair:

`query_name, query_cid, query_smiles, rank, ref_cid, ref_smiles, ref_status, distance`

**`output_dedup_csv`** (deduplicated) — one row per unique neighbor, sorted by closest distance:

```
cid, smiles, ref_status, matched_query_molecules, n_query_matches, best_rank, best_distance
```

- `cid` / `smiles` use lowercase headers — compatible with CSV-based vendor lookup skills that expect those column names.
- `matched_query_molecules` lists which queries had this molecule as a neighbor; `best_rank` / `best_distance` are the closest match across queries.

## Downstream use

The deduplicated CSV is a compact candidate list: unique `cid`/`smiles` pairs ranked by embedding similarity. Pass it to any workflow that needs molecule identifiers — e.g. physical-form and salt-vendor lookup — using the `cid` and `smiles` columns as input.

## Notes & limits

- Embeds by **SMILES**; queries without a SMILES are skipped (name/CID alone cannot be embedded).
- Distances are **L2** in 256-d GIN-E space; smaller = more similar.
- Runtime scales with the reference set size (streams every reference CSV) and with `k`; use a GPU and a sensible `k`.

## Agent checklist

- [ ] Query source decided (inline vs CSV) and SMILES present for each query
- [ ] `embedding_dir`, `checkpoint`, and `project_root` (if needed) provided and valid
- [ ] Encoder env available (torch / PyG / RDKit) where this runs
- [ ] Full config approved by user; `"confirmed": true`; no placeholder paths
- [ ] **HPC:** `io.project_root` set to backend path; env activated; run as a direct `python ... --config` CLI command
- [ ] Run executed; long-form + dedup outputs reported
- [ ] Offered downstream use of dedup CSV if relevant

## More examples

See [examples.md](examples.md).
