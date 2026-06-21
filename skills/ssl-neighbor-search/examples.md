# ssl-neighbor-search examples

Each example shows a user prompt and the run config the agent should present for
approval before running. Defaults come from `config_template.json`; only fields
the user specified are changed.

## Example 1 — inline queries (user lists molecules)

**Prompt:** "Find the 21 SSL neighbors of phenethylamine (`NCCc1ccccc1`) and
ethylenediamine (`NCCN`)."

```json
{
  "confirmed": false,
  "io": {
    "queries": [
      {"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null},
      {"name": "Ethylenediamine", "smiles": "NCCN", "cid": null}
    ],
    "query_csv": null,
    "embedding_dir": "/abs/path/filtered_csv_embeddings",
    "checkpoint": "/abs/path/checkpoints/best_model.pt",
    "project_root": null,
    "output_csv": "ssl_neighbors.csv",
    "output_dedup_csv": "ssl_neighbors_dedup.csv",
    "write_dedup": true
  },
  "knn": {"k": 21, "chunk_size": 100000, "device": "auto", "drop_query_isotopes": true}
}
```

Run: `python "$SKILL_DIR/scripts/ssl_neighbor_search.py" --config run_config.json`

Quick CLI form (inline via flag, still needs `--confirmed` after approval):

```bash
python "$SKILL_DIR/scripts/ssl_neighbor_search.py" --config run_config.json \
  --smiles "NCCc1ccccc1,NCCN" --confirmed
```

## Example 2 — queries from a CSV

**Prompt:** "Find SSL neighbors for all molecules in `my_actives.csv` (columns
`name`, `smiles`)."

```json
{
  "confirmed": false,
  "io": {
    "queries": [],
    "query_csv": "/abs/path/my_actives.csv",
    "name_column": "name",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "embedding_dir": "/abs/path/filtered_csv_embeddings",
    "checkpoint": "/abs/path/checkpoints/best_model.pt",
    "project_root": null,
    "output_csv": "actives_neighbors.csv",
    "output_dedup_csv": "actives_neighbors_dedup.csv",
    "write_dedup": true
  },
  "knn": {"k": 50, "chunk_size": 100000, "device": "cuda", "drop_query_isotopes": true}
}
```

## Example 3 — chain into mol-salt-vendor

**Prompt:** "Get neighbors of these passivators, then check their vendors."

1. Run this skill (Example 1 or 2) → produces `ssl_neighbors_dedup.csv` with
   `cid` / `smiles` columns.
2. Configure **mol-salt-vendor** with `io.input` = that dedup CSV (its defaults
   `cid_column: "cid"`, `smiles_column: "smiles"` already match):

```json
{
  "io": {
    "input": "/abs/path/ssl_neighbors_dedup.csv",
    "cid_column": "cid",
    "smiles_column": "smiles"
  }
}
```

3. Run mol-salt-vendor to get physical form + HCl/HBr/HI salt + vendor info for
   every unique neighbor.

## Output columns

**Long form (`output_csv`):**
`query_name, query_cid, query_smiles, rank, ref_cid, ref_smiles, ref_status, distance`

**Deduplicated (`output_dedup_csv`):**
`cid, smiles, ref_status, matched_query_molecules, n_query_matches, best_rank, best_distance`
(one row per unique neighbor, sorted by closest `best_distance`).
