"""
Plot literature molecules from a CSV as a 5x6 structure grid.

Expected CSV columns:
    molecule_name, cid, smiles, journal
(SMILES is used only to draw structures; it is not shown in panel labels.)

Usage:
    python plot_passivator_molecules_grid.py
    python plot_passivator_molecules_grid.py --input "dataset/literature/molecule_images_by_journal/molecules_cid_smiles_passivator.csv"
"""

import argparse
import csv
import os
import textwrap
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


DEFAULT_INPUT = "/kfs3/scratch/yeming/ai4m/prediction/dataset/literature/molecule_images_by_journal/molecules_cid_smiles.csv"
DEFAULT_OUTPUT = "passivator_molecules_grid.png"
GRID_ROWS = 6
GRID_COLS = 6


def abbreviation_from_name(molecule_name: str) -> str:
    """Return text inside the last parentheses, e.g. 'Long name (BAE)' -> 'BAE'. Empty if none."""
    if not molecule_name or not isinstance(molecule_name, str):
        return ""
    s = molecule_name.strip()
    idx = s.rfind("(")
    if idx != -1 and ")" in s[idx:]:
        end = s.index(")", idx)
        return s[idx + 1 : end].strip()
    return ""


def duplicate_abbreviations(names: List[str]) -> Set[str]:
    """Abbreviations that appear more than once (e.g. two different molecules both '(BA)')."""
    counts: Dict[str, int] = {}
    for raw in names:
        abbr = abbreviation_from_name(raw or "")
        if not abbr:
            continue
        counts[abbr] = counts.get(abbr, 0) + 1
    return {a for a, n in counts.items() if n > 1}


def display_name_for_panel(molecule_name: str, duplicate_abbrs: Set[str]) -> str:
    """Prefer abbreviation unless it collides with another row; then use full name."""
    full = (molecule_name or "").strip() or "Unknown"
    abbr = abbreviation_from_name(molecule_name)
    if abbr and abbr in duplicate_abbrs:
        return full
    if abbr:
        return abbr
    return full


def _norm_row(row: Dict[str, str]) -> Dict[str, str]:
    """Normalize header keys and strip whitespace/BOM."""
    out: Dict[str, str] = {}
    for key, value in row.items():
        # csv.DictReader can emit key=None for malformed/overflow columns.
        if key is None:
            continue
        clean_key = str(key).strip().lstrip("\ufeff")
        if not clean_key:
            continue
        out[clean_key] = value.strip() if isinstance(value, str) else str(value or "").strip()
    return out


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    """Load rows with required columns.

    Handles both properly quoted CSV and malformed lines where molecule_name
    contains unquoted commas by reconstructing fields from the right:
    molecule_name, cid, smiles, journal.
    """
    required = {"molecule_name", "cid", "smiles", "journal"}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV has no header row.")

        header_norm = [str(h).strip().lstrip("\ufeff") for h in header]
        headers = set(header_norm)
        missing = sorted(required - headers)
        if missing:
            raise ValueError("CSV missing required column(s): {}".format(", ".join(missing)))

        idx_name = header_norm.index("molecule_name")
        idx_cid = header_norm.index("cid")
        idx_smiles = header_norm.index("smiles")
        idx_journal = header_norm.index("journal")
        rows: List[Dict[str, str]] = []

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue

            # If molecule_name has unquoted commas, rebuild by taking last 3 fields
            # as cid/smiles/journal and joining the rest as molecule_name.
            if len(row) > len(header_norm):
                name = ",".join(row[: len(row) - 3]).strip()
                cid = (row[-3] if len(row) >= 3 else "").strip()
                smiles = (row[-2] if len(row) >= 2 else "").strip()
                journal = (row[-1] if len(row) >= 1 else "").strip()
                rows.append(
                    {
                        "molecule_name": name,
                        "cid": cid,
                        "smiles": smiles,
                        "journal": journal,
                    }
                )
                continue

            if len(row) < len(header_norm):
                print("Warning: skipping malformed row {} (too few columns).".format(line_no))
                continue

            rows.append(
                {
                    "molecule_name": row[idx_name].strip(),
                    "cid": row[idx_cid].strip(),
                    "smiles": row[idx_smiles].strip(),
                    "journal": row[idx_journal].strip(),
                }
            )

    return rows


def smiles_to_image(smiles: str, size: int) -> Optional["Draw.Image"]:
    """Convert SMILES to a PIL image."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(size, size), kekulize=True)


def panel_label(name: str, cid: str, duplicate_abbrs: Set[str]) -> str:
    """Build per-panel text label: abbreviation unless duplicated; then full name; plus CID."""
    display = display_name_for_panel(name, duplicate_abbrs)
    abbr = abbreviation_from_name(name)
    if abbr and abbr in duplicate_abbrs:
        name_line = textwrap.fill(
            display,
            width=44,
            break_long_words=True,
            break_on_hyphens=False,
        )
    else:
        name_line = textwrap.shorten(display, width=40, placeholder="...")
    cid_line = "CID: {}".format(cid if cid else "N/A")
    return "{}\n{}".format(name_line, cid_line)


def plot_grid(rows: List[Dict[str, str]], output_path: str, mol_size: int = 260) -> None:
    """Plot molecules in a fixed 5x6 grid and save to file."""
    total_slots = GRID_ROWS * GRID_COLS
    rows_to_plot = rows[:total_slots]

    if len(rows_to_plot) < total_slots:
        print(
            "Warning: CSV has {} rows; grid has {} slots. Remaining slots will be blank.".format(
                len(rows_to_plot), total_slots
            )
        )
    elif len(rows) > total_slots:
        print(
            "Warning: CSV has {} rows; plotting first {} only.".format(
                len(rows), total_slots
            )
        )

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(28, 24))
    axes_flat = axes.flatten()

    dup_abbr = duplicate_abbreviations(
        [r.get("molecule_name", "") for r in rows_to_plot]
    )
    if dup_abbr:
        print(
            "Duplicate abbreviations in grid (using full names for those): {}".format(
                ", ".join(sorted(dup_abbr))
            )
        )

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx >= len(rows_to_plot):
            continue

        row = rows_to_plot[idx]
        name = row.get("molecule_name", "")
        cid = row.get("cid", "")
        smiles = row.get("smiles", "")

        img = smiles_to_image(smiles, size=mol_size)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.56,
                "Invalid SMILES",
                ha="center",
                va="center",
                fontsize=15,
                color="crimson",
                transform=ax.transAxes,
            )

        # Put labels below each molecule panel.
        ax.text(
            0.5,
            -0.12,
            panel_label(name, cid, dup_abbr),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=26,
            linespacing=1.45,
            clip_on=False,
        )

    fig.suptitle("Passivator Molecules from CSV (5x6)", fontsize=22, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=6.8, w_pad=3.0)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot passivator molecules from CSV in a 5x6 grid.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to molecules_cid_smiles_passivator.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--mol-size", type=int, default=260, help="Molecule image size in pixels")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError("Input CSV not found: {}".format(input_path))

    rows = load_csv_rows(input_path)
    print("Loaded {} row(s) from {}".format(len(rows), input_path))
    plot_grid(rows=rows, output_path=args.output, mol_size=args.mol_size)
    print("Saved molecule grid to {}".format(args.output))


if __name__ == "__main__":
    main()

