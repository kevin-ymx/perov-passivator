---
name: mol-salt-vendor
description: >-
  For a batch of molecules in a CSV, use an OpenAI LLM (e.g. gpt-5.5) with web
  search to determine (1) each molecule's free-base physical form
  (liquid/powder/solid) and vendor info, and (2) whether HCl/HBr/HI (iodide) salt
  forms exist and their vendor info. Emits a fixed-schema web-search table (CID,
  SMILES, preferred_name, CAS_if_found, free_base_*, HCl/HBr/HI salt
  found/vendor/source, confidence, notes). The user-specified CID and SMILES
  columns are passed through to the output. Use when the user asks for physical
  form, commercial form, salt form, or vendor/supplier/availability info for a
  list of molecules given by CID and/or SMILES (and optional name). Always build a
  run config from the user prompt and present it for approval before running;
  execution is blocked until "confirmed": true.
---

# Molecule physical form & halide-salt vendor lookup

Self-contained skill. For every molecule in an input CSV, an LLM (default
`gpt-5.5`) with web search answers:

1. **Free-base physical form** (`liquid` / `powder` / `solid`), plus **vendor
   info** for the free-base compound that is sold.
2. **Halide salt form** existence for the **HCl, HBr, and HI/iodide** salts, plus
   **vendor info** and **source** for each salt that is sold.

The result is written to a fixed-schema **web-search table** (see
[Output format](#output-format)). All code lives in `scripts/` so it runs from
any agent. The only prerequisites are the input CSV, a Python env with `openai`,
and an `OPENAI_API_KEY`.

## When to use

The user has a list of molecules (CSV) and wants their physical/commercial form,
salt forms, and where to buy them. The user tells you which columns hold the
**CID** and **SMILES** (either may be absent); an optional **name** column
improves accuracy. Build a config, get approval, then run.

## Prerequisites

- Python 3.8+ with `openai` (and `tqdm`): `pip install -r requirements.txt`
- `OPENAI_API_KEY` set in the environment (never hard-code keys in files).
- Input CSV with at least one identifier column (CID, SMILES, or name).
- A model that supports the Responses API web-search tool (e.g. `gpt-5.5`). If
  web search is unavailable, set `llm.use_web_search` to `false` (results then
  rely on model knowledge only and are less reliable for catalog data).

## Files in this skill

```
mol-salt-vendor/
├── SKILL.md                 # this file
├── examples.md              # prompt → config examples
├── requirements.txt         # openai, tqdm
├── config_template.json     # full run config (io + llm)
└── scripts/
    └── mol_salt_vendor.py   # config-driven runner (Responses API + web search)
```

`SKILL_DIR` below refers to this skill's directory.

## Input

Any CSV with a CID and/or SMILES column. Point `cid_column` / `smiles_column` at
the right headers; the user-specified CID and SMILES values are passed through
verbatim to the output `CID` and `SMILES` columns. An optional `name_column`
improves identification accuracy.

## Run config (one JSON specifies the whole run)

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
    "batch_size": 200,
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

| `io` field | Meaning |
|------------|---------|
| `input` | Path to the molecules CSV (required) |
| `cid_column` | Column holding the CID → passed through to output `CID` (ignored if absent) |
| `smiles_column` | Column holding the SMILES → passed through to output `SMILES` |
| `name_column` | Optional column with a chemical name (improves vendor matching) |
| `output_jsonl` | Full structured result per molecule (one JSON object per line) |
| `output_csv` | The web-search table (fixed schema below) |
| `batch_size` | Molecules per batch (default `200`); the CSV is flushed and a progress summary printed after each batch |
| `limit` | Process only the first N rows (`null` = all) — useful for a test run |
| `resume` | Skip molecules already present in `output_jsonl` |

| `llm` field | Meaning |
|-------------|---------|
| `model` | OpenAI model, e.g. `gpt-5.5` |
| `use_web_search` | Attach the Responses API web-search tool (recommended for vendor data) |
| `web_search_tool_type` | Tool type string (`web_search`; use `web_search_preview` if your API version requires it) |
| `max_vendors_per_form` | Cap on vendors listed per form/salt cell |
| `sleep_between_calls` | Seconds between molecules (rate-limit safety) |
| `max_retries` | Retries per molecule on API/parse error |

At least one of `cid_column` / `smiles_column` / `name_column` must match a real
column. A row's identity key for resume is CID, else SMILES, else name.

## User confirmation (required before execution)

**Do not execute until the user approves the complete config.** You do **not**
need to ask about each field one-by-one.

1. **Build a draft config** — merge the user prompt with `config_template.json`.
   For fields the user did not mention, keep template defaults.
2. **Ask only for missing required values** — mainly `io.input` and the actual
   `cid_column` / `smiles_column` / `name_column` names in their CSV.
3. **Present the full JSON** for approval (whole config, not field-by-field).
4. **Wait for approval** ("confirmed", "looks good", "run it", or edits).
5. **Execute** — set `"confirmed": true`, then run.

The runner refuses to execute while `confirmed` is `false` or `io.input`
contains a placeholder.

## Workflow (required order)

1. **Inspect the CSV header** to learn the column names (read the file or ask).
2. **Build draft config** from the user prompt + template defaults.
3. **Confirm `OPENAI_API_KEY`** is set in the environment (do not store it in any file).
4. **Present the full config** for user approval.
5. **Run** (after approval, with `"confirmed": true`):

```bash
python "$SKILL_DIR/scripts/mol_salt_vendor.py" --config run_config.json
```

6. **Report** — output paths, count processed, and any rows with `error`.

Tip: For a first pass, set `io.limit` to a small number (e.g. 3) to sanity-check
results and cost before running the full batch.

### Scaffold a fresh config

```bash
python "$SKILL_DIR/scripts/mol_salt_vendor.py" --write-config run_config.json
```

### Optional CLI overrides (merge on top of `--config`)

`--input`, `--cid-column`, `--smiles-column`, `--name-column`,
`--output-jsonl`, `--output-csv`, `--batch-size`, `--limit`, `--no-resume`,
`--model`, `--no-web-search`, `--sleep`, `--confirmed`.

Re-confirm with the user if any change is made after they approved the config.

## Output format

**`output_csv`** — the web-search table, one row per molecule, with **exactly
these columns in this order**:

```
CID
SMILES
preferred_name
CAS_if_found
free_base_physical_form
free_base_powder_or_solid_vendor
free_base_vendor_source
free_base_vendor_notes
HCl_salt_found
HCl_salt_vendor
HCl_salt_source
HBr_salt_found
HBr_salt_vendor
HBr_salt_source
HI_or_iodide_salt_found
HI_or_iodide_salt_vendor
HI_or_iodide_salt_source
confidence
notes
```

- `CID` and `SMILES` are copied verbatim from the user-specified input columns.
- `*_found` cells are `yes` / `no` / empty (unknown).
- Vendor cells are short strings like `TCI (P0090, >99%, 25 mL); Sigma-Aldrich (128945)`.
- On a failed lookup, `notes` is `ERROR: <message>` and `confidence` is `low`.

**`output_jsonl`** — one JSON object per molecule (audit trail / reprocessing):

```json
{
  "_key": "<cid|smiles|name>",
  "identifiers": {"cid": "...", "smiles": "...", "name": "..."},
  "result": {
    "preferred_name": "...",
    "CAS_if_found": "...",
    "free_base_physical_form": "liquid|powder|solid|unknown",
    "free_base_powder_or_solid_vendor": "...",
    "free_base_vendor_source": "...",
    "free_base_vendor_notes": "...",
    "HCl_salt": {"found": true,  "vendor": "...", "source": "..."},
    "HBr_salt": {"found": false, "vendor": "",    "source": ""},
    "HI_or_iodide_salt": {"found": true, "vendor": "...", "source": "..."},
    "confidence": "high|medium|low",
    "notes": "..."
  }
}
```

The CSV is rebuilt from the full JSONL each run, so it always reflects every
record (including resumed ones).

## Batching

Molecules are processed in batches of `io.batch_size` (default **200**). Each
molecule still gets its own LLM + web-search call (one compound per call, for
quality); `batch_size` controls checkpoint granularity, not how many compounds
share a prompt. After every batch the CSV is rewritten and a progress line is
printed, so a long run can be interrupted and resumed (`resume: true`) without
losing completed work. Set a smaller `batch_size` for more frequent checkpoints.

## Notes & limits

- Vendor/catalog data comes from web search at run time and can be incomplete or
  stale; treat `catalog_number`/`price` as leads to verify, not ground truth.
- The model is instructed to leave a list empty rather than fabricate listings,
  but always spot-check `sources` for important molecules.
- `confidence` is the model's self-report; low confidence warrants manual review.

## Agent checklist

- [ ] CSV column names confirmed (CID / SMILES / optional name)
- [ ] `OPENAI_API_KEY` present in environment
- [ ] Draft config built: user values + template defaults
- [ ] Full config approved by user; `"confirmed": true` set; no placeholder paths
- [ ] (Optional) `limit` used for a small test run first
- [ ] Run executed; output JSONL + CSV paths reported; error rows flagged

## More examples

See [examples.md](examples.md).
