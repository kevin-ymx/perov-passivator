# finetuned-neighbor-search examples

Each example shows a prompt and the run config the agent should present for
approval before running. Defaults come from `config_template.json`; only fields
the user specified are changed.

## Example 1 - inline queries

**Prompt:** "Find the 21 finetuned GIN-E neighbors of phenethylamine
(`NCCc1ccccc1`) and ethylenediamine (`NCCN`)."

```json
{
  "confirmed": false,
  "io": {
    "queries": [
      {"name": "Phenethylamine", "smiles": "NCCc1ccccc1", "cid": null},
      {"name": "Ethylenediamine", "smiles": "NCCN", "cid": null}
    ],
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
  "knn": {"k": 21, "chunk_size": 100000, "drop_query_isotopes": true}
}
```

Run: `python "$SKILL_DIR/scripts/finetuned_neighbor_search.py" --config run_config.json`

Quick CLI form (inline via flag, still needs `--confirmed` after approval):

```bash
python "$SKILL_DIR/scripts/finetuned_neighbor_search.py" --config run_config.json \
  --smiles "NCCc1ccccc1,NCCN" --confirmed
```

## Example 2 - queries from a CSV

**Prompt:** "Find finetuned embedding neighbors for all molecules in
`my_actives.csv` (columns `name`, `smiles`)."

```json
{
  "confirmed": false,
  "io": {
    "queries": [],
    "query_csv": "/abs/path/my_actives.csv",
    "name_column": "name",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "embedding_dir": "/abs/path/filtered_csv_latest_embeddings_finetuned",
    "project_root": null,
    "output_csv": "actives_finetuned_neighbors.csv",
    "output_dedup_csv": "actives_finetuned_neighbors_dedup.csv",
    "write_dedup": true
  },
  "model": {
    "gin_e_checkpoint": "/abs/path/gin_e_finetuned.pt",
    "device": "cuda"
  },
  "knn": {"k": 50, "chunk_size": 100000, "drop_query_isotopes": true}
}
```

## Example 3 - chain into downstream lookup

Run this skill first to produce `finetuned_neighbors_dedup.csv` with `cid` and
`smiles` columns, then feed that CSV to a vendor or property lookup skill.

## Output columns

**Long form (`output_csv`):**
`query_name, query_cid, query_smiles, rank, ref_cid, ref_smiles, ref_status, distance`

**Deduplicated (`output_dedup_csv`):**
`cid, smiles, ref_status, matched_query_molecules, n_query_matches, best_rank, best_distance`
