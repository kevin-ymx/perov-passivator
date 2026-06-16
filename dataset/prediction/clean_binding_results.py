"""
Clean binding energy result CSVs by keeping, for each molecule (cid), only the row
with the smallest adsorption_energy (strongest binding; more negative = stronger).
All other rows for that molecule are removed.

Input: folder containing any number of CSV files.
Output: cleaned CSVs with same filenames in output folder, or a single merged CSV with --merge.

Usage:
  python clean_binding_results.py --input_dir /path/to/shards --output_dir /path/to/shards_cleaned
  python clean_binding_results.py --input_dir /path/to/shards --output_dir /path/to/out --merge
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

from tqdm import tqdm

REQUIRED_COLUMNS = ["cid", "functional_group", "formula", "pb_bond_encoding", "adsorption_energy", "config_name", "adsorbate_structure"]


def parse_adsorption_energy(s: str):
    """Parse adsorption_energy to float. Returns None if invalid."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


def keep_best_per_molecule(rows: list, cid_col: str = "cid", energy_col: str = "adsorption_energy"):
    """
    For each molecule (unique cid), keep only the row with smallest adsorption_energy.
    Ties: keep first occurrence. Rows with invalid cid or energy are dropped.
    Returns (kept_rows, stats_dict).
    """
    by_cid = defaultdict(list)
    dropped_invalid = 0
    for row in rows:
        cid = row.get(cid_col)
        if cid is None or (isinstance(cid, str) and not str(cid).strip()):
            dropped_invalid += 1
            continue
        cid_key = str(cid).strip()
        energy = parse_adsorption_energy(row.get(energy_col))
        if energy is None:
            dropped_invalid += 1
            continue
        by_cid[cid_key].append((energy, row))

    kept = []
    total_duplicates_dropped = 0
    for cid_key, candidates in by_cid.items():
        # smallest adsorption_energy (most negative = strongest binding)
        best_energy, best_row = min(candidates, key=lambda x: x[0])
        kept.append(best_row)
        total_duplicates_dropped += len(candidates) - 1

    stats = {
        "dropped_invalid": dropped_invalid,
        "duplicates_removed": total_duplicates_dropped,
        "molecules_kept": len(kept),
    }
    return kept, stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean binding CSVs: keep one row per molecule (smallest adsorption_energy = strongest binding)."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing CSV files")
    parser.add_argument("--output_dir", required=True, help="Output folder (or single file if --merge)")
    parser.add_argument("--merge", action="store_true", help="Write one merged CSV instead of per-file outputs")
    parser.add_argument("--suffix", default=".csv", help="Only process files with this suffix (default: .csv)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = os.path.join(input_dir, "*" + args.suffix)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No files matching {pattern} found.", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(input_files)} file(s) in {input_dir}")

    if not args.merge:
        os.makedirs(output_dir, exist_ok=True)

    total_read = 0
    total_kept = 0
    total_dropped_invalid = 0
    total_duplicates_removed = 0

    merge_fieldnames = None
    merged_rows = [] if args.merge else None

    for in_path in tqdm(input_files, desc="Files", unit="file"):
        filename = os.path.basename(in_path)

        with open(in_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for col in REQUIRED_COLUMNS:
                if col not in fieldnames:
                    print(f"Error: missing column '{col}' in {in_path}. Found: {fieldnames}", file=sys.stderr)
                    sys.exit(1)
            rows = list(reader)

        total_read += len(rows)
        kept, stats = keep_best_per_molecule(rows)
        total_dropped_invalid += stats["dropped_invalid"]
        total_duplicates_removed += stats["duplicates_removed"]
        total_kept += len(kept)

        if args.merge:
            if merge_fieldnames is None:
                merge_fieldnames = fieldnames
            merged_rows.extend(kept)
        else:
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)

    if args.merge and merge_fieldnames:
        out_path = os.path.join(output_dir, "min_ads_mult1p2_struct_cleaned_merged.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=merge_fieldnames)
            w.writeheader()
            w.writerows(merged_rows)
        print(f"Merged output: {out_path}")

    print(f"Total rows read:        {total_read}")
    print(f"Rows kept (best/mol):  {total_kept}")
    print(f"Dropped (invalid):     {total_dropped_invalid}")
    print(f"Duplicates removed:    {total_duplicates_removed}")


if __name__ == "__main__":
    main()
