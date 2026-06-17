"""
Config-driven batch lookup of molecule free-base physical form + halide-salt
availability, with vendor information, using an OpenAI LLM (e.g. gpt-5.5) and web
search. The user-specified CID and SMILES columns are passed through to the
output as the CID and SMILES columns.

For every molecule in an input CSV the model answers:
  1. Free-base physical form (liquid / powder / solid) and its vendor info.
  2. Whether the HCl / HBr / HI (iodide) salt is found, and its vendor + source.

Output is a fixed-schema web-search table (see SKILL.md) plus a JSONL audit
trail. A single JSON run config fully specifies a run: the "io" block holds
input/output paths and column names; the "llm" block holds model + search
settings. Execution is blocked until "confirmed": true (present config first).

Usage:
    # 1. Write a config template, then edit it:
    python mol_salt_vendor.py --write-config run_config.json

    # 2. Run from the config (after user approves and confirmed: true):
    python mol_salt_vendor.py --config run_config.json

    # Optional CLI overrides:
    python mol_salt_vendor.py --config run_config.json \\
        --input mols.csv --cid-column cid --smiles-column smiles --model gpt-5.5

Requires: OPENAI_API_KEY environment variable; `pip install -r requirements.txt`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; degrade to a no-op wrapper.
    def tqdm(iterable=None, **_kwargs):  # type: ignore
        return iterable if iterable is not None else []

PLACEHOLDER_MARKERS = ("/REPLACE/", "/ABSOLUTE/PATH", "REPLACE_ME")

SYSTEM_PROMPT = """You are a meticulous chemical-procurement curator. For a single \
compound you determine its free-base physical form and its halide-salt availability, \
with concrete vendor information. Use web search to verify current catalog listings \
from reputable chemical suppliers (e.g. Sigma-Aldrich/Merck, TCI, Thermo/Alfa Aesar, \
Fisher, VWR, Apollo Scientific, BLD Pharm, Macklin, Aladdin, Strem, Combi-Blocks).

Rules:
- Identify the compound from the identifiers given (name, CID, and/or SMILES).
- "free_base_physical_form" is the physical state of the neutral/parent (free-base) \
compound at room temperature: one of "liquid", "powder", "solid", or "unknown" \
(use "solid" only when you cannot distinguish powder vs other solid).
- "free_base_powder_or_solid_vendor": vendor(s) selling the free-base compound (in its \
sold form, whether powder/solid or liquid). Give a short string like \
"TCI (cat A1234, >98%, 25 g); Sigma-Aldrich (cat E702)". Empty string if none found.
- "free_base_vendor_source": source URL(s) for the free-base vendor listing(s).
- "free_base_vendor_notes": brief notes (e.g. "sold as anhydrous liquid", "discontinued").
- For each halide salt consider: HCl salt (hydrochloride), HBr salt (hydrobromide), and \
HI/iodide salt (hydroiodide or the iodide salt). For each: "found" is true only if that \
salt is a real, known compound AND you found evidence of it (commercial or literature); \
"vendor" lists suppliers actually selling it (short string, empty if none); "source" is \
the URL(s) you relied on.
- Prefer accuracy over completeness: if unsure, use empty string / null and lower the \
confidence. NEVER fabricate vendors, catalog numbers, prices, or URLs.
- "confidence" is your overall confidence: "high" | "medium" | "low".
- "notes": anything important (ambiguous identity, salt vs free-base distinction, etc.).
- List at most {max_vendors} vendors per field.
- Output STRICT JSON ONLY, no prose, no markdown fences, matching the schema exactly.

JSON schema:
{
  "preferred_name": str|null,
  "CAS_if_found": str|null,
  "free_base_physical_form": "liquid"|"powder"|"solid"|"unknown",
  "free_base_powder_or_solid_vendor": str,
  "free_base_vendor_source": str,
  "free_base_vendor_notes": str,
  "HCl_salt": {"found": bool, "vendor": str, "source": str},
  "HBr_salt": {"found": bool, "vendor": str, "source": str},
  "HI_or_iodide_salt": {"found": bool, "vendor": str, "source": str},
  "confidence": "high"|"medium"|"low",
  "notes": str
}
"""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class IOConfig:
    input: Optional[str] = None
    cid_column: str = "cid"
    smiles_column: str = "smiles"
    name_column: Optional[str] = None
    output_jsonl: str = "mol_salt_vendor_results.jsonl"
    output_csv: str = "mol_salt_vendor_table.csv"
    batch_size: int = 200
    limit: Optional[int] = None
    resume: bool = True

    def validate(self) -> None:
        if not self.input:
            raise ValueError("io.input is required (path to the molecules CSV).")
        if _has_placeholder(self.input):
            raise ValueError(f"io.input still contains a placeholder: {self.input}")
        if not os.path.isfile(self.input):
            raise ValueError(f"io.input file not found: {self.input}")
        if not self.cid_column and not self.smiles_column and not self.name_column:
            raise ValueError(
                "At least one of cid_column / smiles_column / name_column must be set."
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("io.batch_size must be >= 1.")


@dataclass
class LLMConfig:
    model: str = "gpt-5.5"
    use_web_search: bool = True
    web_search_tool_type: str = "web_search"
    max_vendors_per_form: int = 3
    sleep_between_calls: float = 1.0
    max_retries: int = 3


@dataclass
class RunConfig:
    confirmed: bool = False
    io: IOConfig = field(default_factory=IOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def default_run_config() -> RunConfig:
    return RunConfig(
        confirmed=False,
        io=IOConfig(input="/REPLACE/with/your/molecules.csv"),
        llm=LLMConfig(),
    )


def _has_placeholder(value: Optional[str]) -> bool:
    return bool(value) and any(m in value for m in PLACEHOLDER_MARKERS)


def load_run_config_json(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    io = IOConfig(**{**asdict(IOConfig()), **data.get("io", {})})
    llm = LLMConfig(**{**asdict(LLMConfig()), **data.get("llm", {})})
    return RunConfig(confirmed=bool(data.get("confirmed", False)), io=io, llm=llm)


def save_run_config_json(run: RunConfig, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)


# --------------------------------------------------------------------------- #
# Input / output
# --------------------------------------------------------------------------- #
def load_rows(io: IOConfig) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(io.input, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if io.limit is not None:
        rows = rows[: io.limit]
    return rows


def row_identifiers(row: Dict[str, str], io: IOConfig) -> Dict[str, str]:
    def get(col: Optional[str]) -> str:
        if not col:
            return ""
        return (row.get(col) or "").strip()

    return {
        "cid": get(io.cid_column),
        "smiles": get(io.smiles_column),
        "name": get(io.name_column),
    }


def row_key(ident: Dict[str, str]) -> str:
    return ident.get("cid") or ident.get("smiles") or ident.get("name") or ""


def load_done_keys(jsonl_path: str) -> set:
    done: set = set()
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("_key")
            if key:
                done.add(key)
    return done


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def build_user_prompt(ident: Dict[str, str], llm: LLMConfig) -> str:
    lines = ["Compound identifiers:"]
    if ident.get("name"):
        lines.append(f"- name: {ident['name']}")
    if ident.get("cid"):
        lines.append(f"- PubChem CID: {ident['cid']}")
    if ident.get("smiles"):
        lines.append(f"- SMILES: {ident['smiles']}")
    lines.append("")
    lines.append(
        "Report the free-base physical form and vendor, and for the HCl, HBr, and "
        "HI/iodide salts report whether each is found and its vendor/source. "
        "Return STRICT JSON per the schema."
    )
    return "\n".join(lines)


def call_llm(client, ident: Dict[str, str], llm: LLMConfig) -> Dict[str, Any]:
    user_prompt = build_user_prompt(ident, llm)
    instructions = SYSTEM_PROMPT.replace("{max_vendors}", str(llm.max_vendors_per_form))
    kwargs: Dict[str, Any] = {
        "model": llm.model,
        "instructions": instructions,
        "input": user_prompt,
    }
    if llm.use_web_search:
        kwargs["tools"] = [{"type": llm.web_search_tool_type}]

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, llm.max_retries) + 1):
        try:
            response = client.responses.create(**kwargs)
            raw = (response.output_text or "").strip()
            return parse_json_response(raw)
        except Exception as exc:  # network / API / parse errors -> retry
            last_err = exc
            if attempt < llm.max_retries:
                time.sleep(min(2.0 * attempt, 10.0))
    return {"_error": str(last_err) if last_err else "unknown error"}


def parse_json_response(raw: str) -> Dict[str, Any]:
    if not raw:
        raise ValueError("Empty response from API")
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    # If the model wrapped JSON in prose, grab the outermost object.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Flatten -> CSV
# --------------------------------------------------------------------------- #
CSV_COLUMNS = [
    "CID",
    "SMILES",
    "preferred_name",
    "CAS_if_found",
    "free_base_physical_form",
    "free_base_powder_or_solid_vendor",
    "free_base_vendor_source",
    "free_base_vendor_notes",
    "HCl_salt_found",
    "HCl_salt_vendor",
    "HCl_salt_source",
    "HBr_salt_found",
    "HBr_salt_vendor",
    "HBr_salt_source",
    "HI_or_iodide_salt_found",
    "HI_or_iodide_salt_vendor",
    "HI_or_iodide_salt_source",
    "confidence",
    "notes",
]

# Output salt column prefix -> JSON key returned by the model.
_SALT_KEYS = {
    "HCl_salt": "HCl_salt",
    "HBr_salt": "HBr_salt",
    "HI_or_iodide_salt": "HI_or_iodide_salt",
}


def _s(value: Any) -> str:
    """Coerce a scalar/list value to a clean string cell."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(_s(v) for v in value if v not in (None, ""))
    return str(value).strip()


def _bool_str(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def flatten_result(ident: Dict[str, str], result: Dict[str, Any]) -> Dict[str, str]:
    row = {c: "" for c in CSV_COLUMNS}
    # Passthrough identifier columns from the input file.
    row["CID"] = ident.get("cid", "")
    row["SMILES"] = ident.get("smiles", "")

    if result.get("_error"):
        row["notes"] = f"ERROR: {result['_error']}"
        row["confidence"] = "low"
        return row

    row["preferred_name"] = _s(result.get("preferred_name"))
    row["CAS_if_found"] = _s(result.get("CAS_if_found"))
    row["free_base_physical_form"] = _s(result.get("free_base_physical_form"))
    row["free_base_powder_or_solid_vendor"] = _s(result.get("free_base_powder_or_solid_vendor"))
    row["free_base_vendor_source"] = _s(result.get("free_base_vendor_source"))
    row["free_base_vendor_notes"] = _s(result.get("free_base_vendor_notes"))

    for prefix, json_key in _SALT_KEYS.items():
        salt = result.get(json_key) or {}
        if not isinstance(salt, dict):
            salt = {}
        row[f"{prefix}_found"] = _bool_str(salt.get("found"))
        row[f"{prefix}_vendor"] = _s(salt.get("vendor"))
        row[f"{prefix}_source"] = _s(salt.get("source"))

    row["confidence"] = _s(result.get("confidence"))
    row["notes"] = _s(result.get("notes"))
    return row


def append_jsonl(path: str, key: str, ident: Dict[str, str], result: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    record = {"_key": key, "identifiers": ident, "result": result}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM lookup of molecule physical form + halide-salt vendors."
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON run config (confirmed/io/llm). Blocked until confirmed: true.")
    p.add_argument("--write-config", type=str, default=None, metavar="PATH",
                   help="Write a config template (confirmed: false) and exit.")
    p.add_argument("--confirmed", action="store_true",
                   help="Mark run as user-confirmed. Use only after the full config was approved.")
    p.add_argument("--force-unconfirmed", action="store_true",
                   help="Bypass the confirmed gate (not for agent use).")

    p.add_argument("--input", type=str, default=None)
    p.add_argument("--cid-column", type=str, default=None)
    p.add_argument("--smiles-column", type=str, default=None)
    p.add_argument("--name-column", type=str, default=None)
    p.add_argument("--output-jsonl", type=str, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Molecules per batch (default 200); CSV is flushed after each batch.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")

    p.add_argument("--model", type=str, default=None)
    p.add_argument("--no-web-search", action="store_true")
    p.add_argument("--sleep", type=float, default=None)
    return p.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    run = load_run_config_json(args.config) if args.config else default_run_config()

    if args.input is not None:
        run.io.input = args.input
    if args.cid_column is not None:
        run.io.cid_column = args.cid_column
    if args.smiles_column is not None:
        run.io.smiles_column = args.smiles_column
    if args.name_column is not None:
        run.io.name_column = args.name_column
    if args.output_jsonl is not None:
        run.io.output_jsonl = args.output_jsonl
    if args.output_csv is not None:
        run.io.output_csv = args.output_csv
    if args.batch_size is not None:
        run.io.batch_size = args.batch_size
    if args.limit is not None:
        run.io.limit = args.limit
    if args.no_resume:
        run.io.resume = False

    if args.model is not None:
        run.llm.model = args.model
    if args.no_web_search:
        run.llm.use_web_search = False
    if args.sleep is not None:
        run.llm.sleep_between_calls = args.sleep

    if args.confirmed:
        run.confirmed = True
    return run


def make_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Export it before running:\n"
            "  export OPENAI_API_KEY=sk-...   (PowerShell: $env:OPENAI_API_KEY=\"sk-...\")"
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install -r requirements.txt")
    return OpenAI(api_key=api_key)


def main() -> None:
    args = parse_args()

    if args.write_config:
        save_run_config_json(default_run_config(), args.write_config)
        print(f"Wrote config template: {args.write_config}")
        return

    if not args.config and args.input is None:
        raise SystemExit("Provide --config <run_config.json> (or --write-config to scaffold one).")

    run = build_run_config(args)
    try:
        run.io.validate()
    except ValueError as exc:
        raise SystemExit(f"Invalid io config: {exc}")

    if not run.confirmed and not args.force_unconfirmed:
        raise SystemExit(
            "Run blocked: confirmed is false.\n"
            "Present the full run config to the user for approval, then set "
            '"confirmed": true (or pass --confirmed) before executing. See SKILL.md.'
        )

    io, llm = run.io, run.llm
    client = make_client()

    rows = load_rows(io)
    done = load_done_keys(io.output_jsonl) if io.resume else set()

    # Keep only rows that still need processing (skip blanks and resumed keys).
    pending: List[Dict[str, str]] = []
    for row in rows:
        ident = row_identifiers(row, io)
        key = row_key(ident)
        if not key:
            continue
        if io.resume and key in done:
            continue
        pending.append(row)

    batch_size = io.batch_size if io.batch_size and io.batch_size > 0 else len(pending) or 1
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    print(f"Loaded {len(rows)} molecules from {io.input}")
    print(f"Model: {llm.model} | web_search: {llm.use_web_search} | resume: {io.resume}")
    if io.resume and done:
        print(f"Resume: {len(done)} already in {io.output_jsonl}; skipping those.")
    print(f"To process: {len(pending)} molecule(s) in {len(batches)} batch(es) "
          f"of up to {batch_size}.")

    summary_rows: List[Dict[str, str]] = []
    processed = 0
    errors = 0
    for b_idx, batch in enumerate(batches, start=1):
        print(f"\n{'=' * 60}\nBatch {b_idx}/{len(batches)} ({len(batch)} molecules)\n{'=' * 60}")
        batch_errors = 0
        desc = f"Batch {b_idx}/{len(batches)}"
        for row in tqdm(batch, desc=desc):
            ident = row_identifiers(row, io)
            key = row_key(ident)
            result = call_llm(client, ident, llm)
            append_jsonl(io.output_jsonl, key, ident, result)
            summary_rows.append(flatten_result(ident, result))
            done.add(key)
            processed += 1
            if result.get("_error"):
                errors += 1
                batch_errors += 1
            if llm.sleep_between_calls > 0:
                time.sleep(llm.sleep_between_calls)

        # Flush the CSV after each batch so progress is checkpointed.
        all_summary = rebuild_summary_from_jsonl(io.output_jsonl) if io.resume else summary_rows
        write_csv(io.output_csv, all_summary if all_summary else summary_rows)
        print(f"  Batch {b_idx} done: {len(batch)} processed ({batch_errors} error(s)). "
              f"CSV updated: {io.output_csv}")

    # Final CSV write (covers the no-batches / nothing-pending case too).
    all_summary = rebuild_summary_from_jsonl(io.output_jsonl) if io.resume else summary_rows
    write_csv(io.output_csv, all_summary if all_summary else summary_rows)

    print(f"\nDone. Processed {processed} new molecule(s) in {len(batches)} batch(es); "
          f"{errors} error(s).")
    print(f"  JSONL: {io.output_jsonl}")
    print(f"  CSV:   {io.output_csv}")


def rebuild_summary_from_jsonl(jsonl_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.isfile(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ident = rec.get("identifiers") or {}
            result = rec.get("result") or {}
            rows.append(flatten_result(ident, result))
    return rows


if __name__ == "__main__":
    sys.exit(main())
