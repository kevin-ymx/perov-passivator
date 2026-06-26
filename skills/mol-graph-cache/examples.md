# mol-graph-cache examples

## Single CSV

```json
{
  "confirmed": true,
  "io": {
    "mode": "single",
    "input": "combined_data.csv",
    "input_dir": null,
    "output_dir": "graph_cache",
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

Run:

```bash
python skills/mol-graph-cache/scripts/mol_graph_cache.py --config run_config.json
```

## CSV shard directory

```json
{
  "confirmed": true,
  "io": {
    "mode": "shards",
    "input": null,
    "input_dir": "/scratch/yeming/pubchem_shards",
    "output_dir": "/scratch/yeming/mol_graph_cache",
    "cid_column": "cid",
    "smiles_column": "smiles"
  },
  "graph": {
    "include_hydrogens": true,
    "node_features": [
      "element_onehot",
      "formal_charge",
      "partial_charge",
      "total_degree",
      "total_num_hs",
      "hybridization_onehot",
      "is_aromatic",
      "is_in_ring"
    ],
    "edge_features": [
      "bond_type_onehot",
      "bond_order",
      "bond_stereo_onehot",
      "is_conjugated",
      "is_in_ring"
    ]
  }
}
```

