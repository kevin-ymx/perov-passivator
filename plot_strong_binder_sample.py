"""
Randomly sample 40 query molecules from strong_binder_knn_results.csv and plot
their 2D structures in a 5x8 grid (single image).

Usage:
    python plot_strong_binder_sample.py
    python plot_strong_binder_sample.py --input strong_binder_knn_results.csv --output strong_binder_sample.png --seed 42
"""
import argparse
import csv
import random
import os
import sys
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

DEFAULT_INPUT_CSV = "strong_binder_knn_results.csv"
DEFAULT_OUTPUT = "strong_binder_sample.png"
MOL_IMAGE_SIZE = (300, 300)
LABEL_HEIGHT = 36
GRID_ROWS = 5
GRID_COLS = 8
NUM_SAMPLES = 40
GAP = 4  # pixels between cells


def smiles_to_image(smiles: str, size: Tuple[int, int] = MOL_IMAGE_SIZE) -> Optional[Image.Image]:
    """Draw 2D structure from SMILES; return PIL Image or None."""
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size, kekulize=True)
    except Exception:
        return None


def get_font(size: int = 18):
    """Try to load a TrueType font; fall back to default."""
    for path in [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_unique_queries(csv_path: str) -> List[dict]:
    """
    Load strong_binder_knn_results.csv and deduplicate by query_cid.
    Returns one dict per unique query with keys: query_cid, query_smiles, query_energy.
    """
    seen = set()
    queries = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("query_cid") or "").strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            queries.append({
                "query_cid": cid,
                "query_smiles": (row.get("query_smiles") or "").strip(),
                "query_energy": (row.get("query_energy") or "").strip(),
            })
    return queries


def make_grid(
    molecules: List[dict],
    mol_size: Tuple[int, int] = MOL_IMAGE_SIZE,
) -> Image.Image:
    """
    Build a 5x8 grid image.  Each cell shows a 2D structure with a label
    (CID and binding energy) underneath.
    """
    w, h = mol_size
    cell_w = w
    cell_h = h + LABEL_HEIGHT
    grid_w = GRID_COLS * cell_w + (GRID_COLS - 1) * GAP
    grid_h = GRID_ROWS * cell_h + (GRID_ROWS - 1) * GAP

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = get_font(16)

    for idx, mol_info in enumerate(molecules):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = col * (cell_w + GAP)
        y = row * (cell_h + GAP)

        # Draw molecule
        img = smiles_to_image(mol_info["query_smiles"], mol_size)
        if img is None:
            img = Image.new("RGB", mol_size, (240, 240, 240))
        grid.paste(img, (x, y))

        # Draw label: CID and energy
        energy_raw = mol_info.get("query_energy", "")
        try:
            energy = f"{float(energy_raw):.3f}"
        except (ValueError, TypeError):
            energy = energy_raw
        label = f"CID {mol_info['query_cid']}  {energy} eV"
        if len(label) > 35:
            label = label[:32] + "..."
        draw.text((x + 4, y + h + 2), label, fill=(0, 0, 0), font=font)

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Sample 40 query molecules from strong_binder_knn_results.csv and plot 5x8 grid."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV,
                        help="Path to strong_binder_knn_results.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output image path (PNG)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--size", type=int, default=300,
                        help="Molecule image width/height in pixels")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    print(f"Loading {args.input} ...")
    queries = load_unique_queries(args.input)
    print(f"Found {len(queries)} unique query molecules")

    if len(queries) < NUM_SAMPLES:
        print(f"Warning: only {len(queries)} queries available (< {NUM_SAMPLES}), using all")
        sampled = queries
    else:
        random.seed(args.seed)
        sampled = random.sample(queries, NUM_SAMPLES)
    print(f"Sampled {len(sampled)} molecules")

    print("Plotting 5x8 grid ...")
    mol_size = (args.size, args.size)
    grid_img = make_grid(sampled, mol_size=mol_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    grid_img.save(args.output, dpi=(150, 150))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
