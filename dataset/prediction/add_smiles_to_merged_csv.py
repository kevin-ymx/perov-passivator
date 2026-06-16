"""
Add a SMILES column to min_ads_mult1p2_struct_cleaned_merged.csv by looking up
each row's cid (first column) in sampled_details.csv, where cid and SMILES are
in the first and second columns respectively.

Usage:
  python add_smiles_to_merged_csv.py --merged_csv /path/to/min_ads_mult1p2_struct_cleaned_merged.csv --lookup_csv /path/to/sampled_details.csv [--output /path/to/output.csv]
  If --output is omitted, the merged CSV is updated in place (SMILES column added).
"""
import argparse
import csv
import os
import sys

from tqdm import tqdm

DEFAULT_LOOKUP = "/kfs3/scratch/yeming/ai4m/prediction/dataset/prediction/sampled_details.csv"


def main():
    parser = argparse.ArgumentParser(description="Add SMILES column to merged CSV from cid lookup.")
    parser.add_argument("--merged_csv", required=True, help="Path to min_ads_mult1p2_struct_cleaned_merged.csv")
    parser.add_argument("--lookup_csv", default=DEFAULT_LOOKUP, help="Path to sampled_details.csv (col1=cid, col2=SMILES)")
    parser.add_argument("--output", default="", help="Output path. If empty, overwrite merged_csv.")
    parser.add_argument("--cid_col", default="cid", help="Column name for CID in merged CSV")
    parser.add_argument("--smiles_col", default="SMILES", help="Name of the SMILES column to add")
    args = parser.parse_args()

    merged_path = os.path.abspath(args.merged_csv)
    lookup_path = os.path.abspath(args.lookup_csv)
    out_path = os.path.abspath(args.output) if args.output.strip() else merged_path

    if not os.path.isfile(merged_path):
        print(f"Error: merged CSV not found: {merged_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(lookup_path):
        print(f"Error: lookup CSV not found: {lookup_path}", file=sys.stderr)
        sys.exit(1)

    # Build cid -> SMILES from lookup (first col = cid, second col = SMILES)
    print(f"Loading lookup from {lookup_path}...")
    cid_to_smiles = {}
    with open(lookup_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in tqdm(reader, desc="Lookup"):
            if len(row) >= 2:
                try:
                    cid = str(int(row[0].strip()))
                    smiles = (row[1] or "").strip()
                    cid_to_smiles[cid] = smiles
                except (ValueError, TypeError):
                    continue
    print(f"Loaded {len(cid_to_smiles)} cid -> SMILES pairs")

    # Read merged CSV
    print(f"Reading merged CSV: {merged_path}")
    with open(merged_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if args.cid_col not in fieldnames:
        print(f"Error: merged CSV has no column '{args.cid_col}'. Columns: {fieldnames}", file=sys.stderr)
        sys.exit(1)
    if args.smiles_col in fieldnames:
        print(f"Warning: column '{args.smiles_col}' already exists; values will be overwritten from lookup.")

    # Insert SMILES column after cid (or at end if cid not in expected position)
    if args.smiles_col not in fieldnames:
        cid_idx = fieldnames.index(args.cid_col)
        fieldnames.insert(cid_idx + 1, args.smiles_col)

    matched = 0
    for row in rows:
        cid = (row.get(args.cid_col) or "").strip()
        try:
            cid_key = str(int(cid))
        except (ValueError, TypeError):
            row[args.smiles_col] = ""
            continue
        smiles = cid_to_smiles.get(cid_key, "")
        row[args.smiles_col] = smiles
        if smiles:
            matched += 1

    print(f"Matched SMILES for {matched} / {len(rows)} rows")

    # Write output
    in_place = out_path == merged_path
    if in_place:
        write_path = merged_path + ".tmp"
    else:
        write_path = out_path
    os.makedirs(os.path.dirname(os.path.abspath(write_path)) or ".", exist_ok=True)
    with open(write_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if in_place:
        os.replace(write_path, merged_path)
        print(f"Updated {merged_path} with SMILES column")
    else:
        print(f"Wrote {out_path} with SMILES column")


if __name__ == "__main__":
    main()
