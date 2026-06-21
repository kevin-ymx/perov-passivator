# eb-pbcoord-predict examples

Each example shows a user prompt and the run config the agent should present for
approval before running. Defaults come from `config_template.json`; only fields
the user specified are changed.

## Example 1 — inline queries (user lists molecules)

**Prompt:** "Predict the Lewis base–Pb binding energy on FAPbI3 of phenethylamine
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

Run: `python "$SKILL_DIR/scripts/eb_pbcoord_predict.py" --config run_config.json`

Quick CLI form (inline via flag, still needs `--confirmed` after approval):

```bash
python "$SKILL_DIR/scripts/eb_pbcoord_predict.py" --config run_config.json \
  --smiles "NCCc1ccccc1,NCCN" --confirmed
```

## Example 2 — queries from a CSV, with a strong-binder threshold

**Prompt:** "Predict Eb on FAPbI3 for all molecules in `candidates.csv` (columns `name`,
`smiles`) and flag the ones below -1.5 eV."

```json
{
  "confirmed": false,
  "io": {
    "queries": [],
    "query_csv": "/abs/path/candidates.csv",
    "name_column": "name",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "project_root": null,
    "output_csv": "candidates_eb.csv",
    "sort_by_energy": true
  },
  "model": {
    "downstream_checkpoint": "/abs/path/downstream_best_model.pt",
    "gin_e_checkpoint": "/abs/path/gin_e_finetuned.pt",
    "device": "cuda",
    "batch_size": 128,
    "energy_threshold": -1.5
  }
}
```

The output adds a `below_threshold` column (`True` for Eb ≤ −1.5 eV).

## Example 3 — chain from ssl-neighbor-search, then into mol-salt-vendor

**Prompt:** "Take the neighbors I just found, rank them by binding energy, then
check vendors for the strongest binders."

1. Run **ssl-neighbor-search** → produces `ssl_neighbors_dedup.csv` with
   `cid` / `smiles` columns.
2. Configure this skill with `io.query_csv` = that dedup CSV:

```json
{
  "io": {
    "query_csv": "/abs/path/ssl_neighbors_dedup.csv",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "output_csv": "neighbors_eb.csv",
    "sort_by_energy": true
  },
  "model": {
    "downstream_checkpoint": "/abs/path/downstream_best_model.pt",
    "gin_e_checkpoint": "/abs/path/gin_e_finetuned.pt"
  }
}
```

3. Run this skill → `neighbors_eb.csv` (ranked by Eb, `cid`/`smiles` preserved).
4. Configure **mol-salt-vendor** with `io.input` = `neighbors_eb.csv` (its defaults
   `cid_column: "cid"`, `smiles_column: "smiles"` already match) to get physical
   form + HCl/HBr/HI salt + vendor info for the top candidates.

## Output columns

```
name, cid, smiles, predicted_binding_energy_eV, prediction_status
```

Plus `below_threshold` when `model.energy_threshold` is set. Rows are sorted by
predicted Eb (ascending; strongest binders first) when `io.sort_by_energy: true`.
