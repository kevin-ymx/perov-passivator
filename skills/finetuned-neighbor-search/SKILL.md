---
name: finetuned-neighbor-search
description: >-
  Find nearest-neighbor molecules in a finetuned GIN-E encoder embedding space.
  Queries may be entered inline by the user (name + SMILES, optional CID) or
  loaded from a CSV. Computes query embeddings with a finetuned GIN-E encoder
  checkpoint and runs k-NN over finetuned reference embedding CSVs, writing a
  long-form neighbor table and a deduplicated table with `cid`/`smiles` columns
  suitable for downstream lookup. Use when the user asks to find similar
  molecules, nearest neighbors, embedding neighbors, or analogs using the
  finetuned GIN-E encoder rather than the SSL checkpoint. Always build a run
  config from the user prompt and present it for approval before running;
  execution is blocked until "confirmed": true.
---

# Finetuned GIN-E nearest-neighbor search

Self-contained skill for k-nearest-neighbor search in a fixed **finetuned GIN-E
embedding space**. For molecules the user provides (inline or via CSV), it
embeds queries with a finetuned GIN-E encoder checkpoint and returns the `k`
closest molecules from a precomputed finetuned reference embedding set. All
orchestration code lives in `scripts/`; the skill loads the backend implementation
from `io.project_root`.

## When to use

The user wants embedding-space neighbors (analogs) from the **finetuned GIN-E
encoder**, not the SSL GIN-E checkpoint. Pick the query source from the prompt:

- **Inline** - the user lists molecules (SMILES, optionally a name/CID) directly.
- **CSV** - the user points to a file of molecules.

Use `ssl-neighbor-search` instead when the user explicitly wants the original SSL
embedding space.

## Prerequisites

- Python 3.8+ with `torch`, `torch-geometric`, `rdkit`, `numpy`, `scipy`, `tqdm`.
- A finetuned **GIN-E encoder checkpoint** (`model.gin_e_checkpoint`).
- A **finetuned reference embedding directory** (`io.embedding_dir`): CSV shards
  with `emb_0..emb_255` plus `cid`/`smiles` or `PUBCHEM_COMPOUND_CID`/`SMILES`.
- A backend at `io.project_root` that provides `knn_finetunedembedding_search.py`,
  `inference_Eb.py`, `config.py`, and `models/`.
- A GPU is recommended for embedding many queries.

Reference embeddings must have been generated from the same finetuned encoder
family as `model.gin_e_checkpoint`; distances are only meaningful within the
same embedding space.

## Files in this skill

```text
finetuned-neighbor-search/
|-- SKILL.md
|-- examples.md
|-- requirements.txt
|-- config_template.json
`-- scripts/
    `-- finetuned_neighbor_search.py
```

`SKILL_DIR` below refers to this skill's directory.

## Query input: inline OR CSV

Each query needs a **SMILES**; `name` and `cid` are optional labels passed
through to the output. Provide exactly one source:

**Inline** (`io.queries`):

```json
"queries": [
  {"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null},
  {"name": "Ethylenediamine", "smiles": "NCCN", "cid": 3301}
]
```

**CSV** (`io.query_csv` + column names): set `io.queries` to `[]` and point
`query_csv` at the file; `name_column` / `cid_column` / `smiles_column` select
the columns.

## Run config

```json
{
  "confirmed": false,
  "io": {
    "queries": [{"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null}],
    "query_csv": null,
    "name_column": "molecule_name",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "embedding_dir": "/abs/path/filtered_csv_latest_embeddings_finetuned",
    "project_root": null,
    "output_csv": "finetuned_neighbors.csv",
    "output_dedup_csv": "finetuned_neighbors_dedup.csv",
    "write_dedup": true
  },
  "model": {
    "gin_e_checkpoint": "/abs/path/gin_e_finetuned.pt",
    "device": "auto"
  },
  "knn": {
    "k": 21,
    "chunk_size": 100000,
    "drop_query_isotopes": true
  }
}
```

| Field | Meaning |
|-------|---------|
| `io.embedding_dir` | Directory of finetuned reference embedding CSVs (`emb_0..emb_255`) |
| `io.project_root` | Backend root (`null` = auto-detect when co-located) |
| `model.gin_e_checkpoint` | Finetuned GIN-E encoder checkpoint |
| `knn.k` | Neighbors per query |
| `knn.chunk_size` | Reference rows per streamed chunk |
| `knn.drop_query_isotopes` | Drop query SMILES with isotope labels |

## User confirmation

Do not execute until the user approves the complete config.

1. Build a draft config from the prompt and `config_template.json`.
2. Ask only for missing required values: query source, `embedding_dir`,
   `gin_e_checkpoint`, and `project_root` if auto-detection will not work.
3. Present the full JSON for approval.
4. After approval, set `"confirmed": true` and run.

The runner refuses to execute while `confirmed` is `false` or required paths are
missing/placeholders.

## Workflow

1. Determine inline vs CSV query source.
2. Build and present the full run config.
3. Run after approval:

```bash
python "$SKILL_DIR/scripts/finetuned_neighbor_search.py" --config run_config.json
```

4. Report output paths, number of embedded queries, and number of unique
   deduplicated neighbors.

### Scaffold a fresh config

```bash
python "$SKILL_DIR/scripts/finetuned_neighbor_search.py" --write-config run_config.json
```

### Optional CLI overrides

`--smiles "S1,S2"`, `--query-csv`, `--embedding-dir`, `--gin-e-checkpoint`,
`--project-root`, `--output-csv`, `--output-dedup-csv`, `--no-dedup`, `-k/--k`,
`--chunk-size`, `--device`, `--confirmed`.

## HPC execution

Run the same Python CLI directly on an HPC node or interactive GPU session. Use
absolute paths and set `io.project_root` when the skill is installed separately
from the backend.

### Kestrel paths (defaults)

| Symbol | Path | Purpose |
|--------|------|---------|
| `SKILL_DIR` | `/home/yeming/skills/finetuned-neighbor-search` | Skill code |
| `BACKEND_ROOT` | `/home/yeming/perov-passivator` | Backend root |
| `RUN_ROOT` | `/scratch/yeming/finetuned-neighbor-search` | Run workspace |

Example:

```bash
source /home/yeming/.bashrc
conda activate /scratch/yeming/conda_envs/ai4m

python /home/yeming/skills/finetuned-neighbor-search/scripts/finetuned_neighbor_search.py \
  --config /scratch/yeming/finetuned-neighbor-search/run_configs/run_config.json
```

## Output format

**`output_csv`**:

`query_name, query_cid, query_smiles, rank, ref_cid, ref_smiles, ref_status, distance`

**`output_dedup_csv`**:

```text
cid, smiles, ref_status, matched_query_molecules, n_query_matches, best_rank, best_distance
```

The deduplicated CSV is sorted by closest distance and keeps lowercase `cid` and
`smiles` headers for downstream tools.

## Agent checklist

- [ ] Query source decided and SMILES present for each query
- [ ] `embedding_dir`, `gin_e_checkpoint`, and `project_root` if needed are valid
- [ ] Full config approved by user; `"confirmed": true`; no placeholder paths
- [ ] Backend env available where this runs
- [ ] Run executed; long-form + dedup outputs reported

## More examples

See [examples.md](examples.md).
