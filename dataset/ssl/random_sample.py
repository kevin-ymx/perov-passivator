"""
Randomly sample a fraction of molecules from a CSV file.

Usage:
    python random_sample.py --input combine.csv --output sampled.csv --fraction 0.1
    python random_sample.py --input combine.csv --output sampled.csv --count 10000
"""
import argparse
import csv
import random
import os


def random_sample_csv(
    input_file: str,
    output_file: str,
    fraction: float = None,
    count: int = None,
    seed: int = 42
):
    """
    Randomly sample rows from a CSV file.
    
    Args:
        input_file: Path to input CSV file.
        output_file: Path to output CSV file.
        fraction: Fraction of rows to sample (0.0 to 1.0). Mutually exclusive with count.
        count: Number of rows to sample. Mutually exclusive with fraction.
        seed: Random seed for reproducibility.
    """
    if fraction is None and count is None:
        raise ValueError("Must specify either --fraction or --count")
    if fraction is not None and count is not None:
        raise ValueError("Cannot specify both --fraction and --count")
    
    random.seed(seed)
    
    # Read all rows
    print(f"Reading from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"Total rows: {total_rows:,}")
    
    # Calculate sample size
    if fraction is not None:
        sample_size = int(total_rows * fraction)
        print(f"Sampling {fraction*100:.1f}% = {sample_size:,} rows")
    else:
        sample_size = min(count, total_rows)
        print(f"Sampling {sample_size:,} rows")
    
    # Random sample
    if sample_size >= total_rows:
        sampled_rows = rows
        print("Sample size >= total rows, keeping all rows")
    else:
        sampled_rows = random.sample(rows, sample_size)
    
    # Write output
    print(f"Writing to {output_file}...")
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)
    
    print(f"Done! Sampled {len(sampled_rows):,} rows ({len(sampled_rows)/total_rows*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample rows from a CSV file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="combine.csv",
        help="Input CSV file (default: combine.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sampled.csv",
        help="Output CSV file (default: sampled.csv)"
    )
    parser.add_argument(
        "--fraction", "-f",
        type=float,
        default=None,
        help="Fraction of rows to sample (0.0 to 1.0)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        help="Number of rows to sample"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Default to 10% if neither fraction nor count specified
    if args.fraction is None and args.count is None:
        print("No --fraction or --count specified, defaulting to 10%")
        args.fraction = 0.1
    
    random_sample_csv(
        input_file=args.input,
        output_file=args.output,
        fraction=args.fraction,
        count=args.count,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
