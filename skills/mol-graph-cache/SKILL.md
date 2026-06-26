---
name: mol-graph-cache
description: >-
  Convert molecule batches from CSV files into raw PyTorch Geometric molecular
  graph caches with user-selected node and edge features. Use when the user asks
  to build, cache, precompute, or materialize molecule graphs from SMILES/CID
  CSV data, including single CSV files or directories of CSV shards. Always show
  the available feature catalog before building a run config, present the full
  config for approval, and execute only after "confirmed": true.
---

# Molecule graph cache

Build raw molecular graph caches from CSV files with SMILES strings. The runner
is a direct Python CLI; do not use Slurm or sbatch for this skill.

## Workflow

1. Run `--list-features` and show the user the available node and edge features.
2. Build a run config from `config_template.json`; keep all selected features
   explicitly listed in `graph.node_features` and `graph.edge_features`.
3. Present the full JSON config for approval.
4. Execute only after the user approves and `confirmed` is set to `true`.
5. Report output paths, kept/skipped counts, and feature dimensions.

```bash
python skills/mol-graph-cache/scripts/mol_graph_cache.py --list-features
python skills/mol-graph-cache/scripts/mol_graph_cache.py --write-config run_config.json
python skills/mol-graph-cache/scripts/mol_graph_cache.py --config run_config.json
```

## Config

The config has no preset field. It must explicitly list all selected features:

```json
{
  "confirmed": false,
  "io": {
    "mode": "single",
    "input": "/REPLACE/with/input.csv",
    "input_dir": null,
    "output_dir": "/REPLACE/with/output_graph_cache",
    "cid_column": "CID",
    "smiles_column": "SMILES"
  },
  "graph": {
    "include_hydrogens": false,
    "node_features": [
      "element_onehot",
      "partial_charge",
      "hybridization_onehot",
      "degree",
      "valence_electrons",
      "electronegativity"
    ],
    "edge_features": [
      "bond_type_onehot",
      "bond_direction_onehot"
    ]
  }
}
```

`include_hydrogens: false` builds heavy-atom graphs. `true` applies
`Chem.AddHs` before graph construction.

## Input and output

Input CSV files need a SMILES column. CID is optional but preserved when present.
The configured column names are tried first, then lowercase/uppercase fallbacks.

Outputs:

- `<output_dir>/<input_stem>_graphs.pt`
- `<output_dir>/manifest.json`
- `<output_dir>/index.csv`
- `<output_dir>/failures.csv`

The `.pt` cache contains `schema_version`, `node_features`, `edge_features`,
`feature_dims`, `onehot_vocabs`, `include_hydrogens`, `graphs`, `cids`,
`smiles`, `source_file`, and `source_row_indices`.

## Notes

- Categorical features are one-hot only.
- One-hot dimensions come from fixed vocabularies saved into the cache and
  manifest.
- This skill only converts molecules to raw graphs. It does not augment graphs,
  split train/validation sets, fetch PubChem data, attach labels, or run models.

