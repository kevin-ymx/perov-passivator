# mol-salt-vendor examples

Each example shows a user prompt and the run config the agent should present for
approval before running. Defaults come from `config_template.json`; only fields
the user specified are changed.

## Example 1 — CSV with cid + smiles columns

**Prompt:** "Run mol-salt-vendor on `molecules.csv` (columns `cid`, `smiles`).
Use gpt-5.5."

```json
{
  "confirmed": false,
  "io": {
    "input": "/abs/path/molecules.csv",
    "cid_column": "cid",
    "smiles_column": "smiles",
    "name_column": null,
    "output_jsonl": "mol_salt_vendor_results.jsonl",
    "output_csv": "mol_salt_vendor_table.csv",
    "batch_size": 100,
    "limit": null,
    "resume": true
  },
  "llm": {
    "model": "gpt-5.5",
    "use_web_search": true,
    "web_search_tool_type": "web_search",
    "max_vendors_per_form": 3,
    "sleep_between_calls": 1.0,
    "max_retries": 3
  }
}
```

Run: `python "$SKILL_DIR/scripts/mol_salt_vendor.py" --config run_config.json`

## Example 2 — quick test on first 5 rows

**Prompt:** "Test it on the first 5 rows first."

Set `io.limit` to `5`. After it looks good, set `limit` back to `null` and rerun
— `resume: true` skips the 5 already done. CLI form:

```bash
python "$SKILL_DIR/scripts/mol_salt_vendor.py" --config run_config.json --limit 5
```

## Example 4 — large run, batches of 100

**Prompt:** "Process all of them in batches of 100."

Keep `io.batch_size` at `100` (the default). The CSV is flushed and a progress
summary printed after each batch; interrupting and rerunning with `resume: true`
continues where it left off. CLI form:

```bash
python "$SKILL_DIR/scripts/mol_salt_vendor.py" --config run_config.json --batch-size 100
```

## Example 3 — custom column names + a name column

**Prompt:** "My file uses `PUBCHEM_COMPOUND_CID` and `SMILES`, and has a
`compound_name` column."

```json
{
  "io": {
    "cid_column": "PUBCHEM_COMPOUND_CID",
    "smiles_column": "SMILES",
    "name_column": "compound_name"
  }
}
```

(Other fields as in Example 1.) The `compound_name` column is sent to the model
to improve identification; the CID/SMILES values are passed through to the output
`CID` / `SMILES` columns.

## Output table columns (fixed schema)

```
CID, SMILES,
preferred_name, CAS_if_found,
free_base_physical_form, free_base_powder_or_solid_vendor,
free_base_vendor_source, free_base_vendor_notes,
HCl_salt_found, HCl_salt_vendor, HCl_salt_source,
HBr_salt_found, HBr_salt_vendor, HBr_salt_source,
HI_or_iodide_salt_found, HI_or_iodide_salt_vendor, HI_or_iodide_salt_source,
confidence, notes
```

- `CID` / `SMILES` are copied from the user-specified input columns.
- `*_found` is `yes` / `no` / blank; vendor cells are short strings like
  `TCI (P0090, >99%, 25 mL); Sigma-Aldrich (128945)`.
- Failed rows have `notes = ERROR: <message>` and `confidence = low`; delete
  their lines from the JSONL and rerun (with `resume: true`) to retry just those.
