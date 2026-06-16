"""
Extract molecular structure images from PubChem (by CID) for molecules in
extracted_results_mol_cleaned_v2.csv, filtered by journal. Group by journal and
create one combined figure per journal with subplots (one molecule image + name per subplot).
"""
import csv
import io
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Force white backgrounds for figures and axes
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "none"

# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = "extracted_results_mol_cleaned_v2.csv"
OUTPUT_DIR = "molecule_images_by_journal"
OUTPUT_CID_SMILES_CSV = "molecules_cid_smiles.csv"
PUBCHEM_IMAGE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
# Request multiple SMILES-related properties; API may return SMILES, ConnectivitySMILES, or CanonicalSMILES
PUBCHEM_SMILES_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,SMILES,ConnectivitySMILES/JSON"
IMAGE_SIZE = "300x300"
SLEEP_BETWEEN_REQUESTS = 0.25
REQUEST_TIMEOUT = 15

# Column indices (0-based): title=0, year=1, journal=2, impact_factor=3, molecule_name=4, molecule_cid=5
JOURNAL_COL = 2
MOLECULE_NAME_COL = 4
MOLECULE_CID_COL = 5

JOURNALS = [
    "NATURE ENERGY",
    "NATURE",
    "SCIENCE",
    "NATURE ELECTRONICS",
    "NATURE MATERIALS",
    "NATURE NANOTECHNOLOGY",
    "NATURE PHOTONICS",
    "NATURE CHEMISTRY",
    "NATURE SYNTHESIS",
    "NATURE COMMUNICATIONS",
    "SCIENCE ADVANCES",
    "JOULE",
]

# Extra molecules to include per journal (cid, name) not necessarily in the CSV
EXTRA_MOLECULES: Dict[str, List[Tuple[int, str]]] = {
    "SCIENCE": [(75919, "2-Pyy")],
}

# Molecule names to exclude per journal (case-insensitive match; include common variants e.g. C60, C-60)
EXCLUDE_MOLECULES: Dict[str, List[str]] = {
    "JOULE": ["C60", "C-60"],
    "NATURE COMMUNICATIONS": ["ortho-carborane", "Methylammonium iodide"],
}

SUBPLOT_COLS = 6
MAX_SUBPLOTS_PER_FIGURE = 60
FONT_SIZE_NAME = 11


def load_molecules_by_journal(csv_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Load CSV, filter by journal list, keep rows with valid CID in column 6.
    Returns: journal -> list of (cid, molecule_name), deduplicated by CID per journal.
    """
    journal_set = set(j.upper().strip() for j in JOURNALS)
    by_journal: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    seen_per_journal: Dict[str, set] = defaultdict(set)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return dict(by_journal)
        for row in reader:
            if len(row) <= max(JOURNAL_COL, MOLECULE_NAME_COL, MOLECULE_CID_COL):
                continue
            journal = (row[JOURNAL_COL] or "").strip().upper()
            if journal not in journal_set:
                continue
            cid_s = (row[MOLECULE_CID_COL] or "").strip()
            if not cid_s or not cid_s.isdigit():
                continue
            cid = int(cid_s)
            name = (row[MOLECULE_NAME_COL] or "").strip() or f"CID{cid}"
            if cid in seen_per_journal[journal]:
                continue
            seen_per_journal[journal].add(cid)
            by_journal[journal].append((cid, name))

    for journal, extra in EXTRA_MOLECULES.items():
        for cid, name in extra:
            if cid not in seen_per_journal[journal]:
                by_journal[journal].append((cid, name))
                seen_per_journal[journal].add(cid)

    for journal, exclude_names in EXCLUDE_MOLECULES.items():
        if journal not in by_journal:
            continue
        exclude_set = {n.strip().lower() for n in exclude_names}
        by_journal[journal] = [
            (cid, name) for cid, name in by_journal[journal]
            if name.strip().lower() not in exclude_set
        ]

    return dict(by_journal)


def _image_white_background(img: Image.Image, grey_min: int = 200, grey_tolerance: int = 25) -> Image.Image:
    """Replace light grey background with white. PubChem PNGs often have a grey background (~232)."""
    img = img.convert("RGB")
    data = list(img.getdata())
    out = []
    for (r, g, b) in data:
        if (
            r >= grey_min and g >= grey_min and b >= grey_min
            and abs(r - g) <= grey_tolerance and abs(g - b) <= grey_tolerance and abs(r - b) <= grey_tolerance
        ):
            out.append((255, 255, 255))
        else:
            out.append((r, g, b))
    img.putdata(out)
    return img


def fetch_structure_image(cid: int) -> Optional[Image.Image]:
    """Download PubChem structure PNG for CID; return PIL Image or None."""
    url = PUBCHEM_IMAGE_URL.format(cid=cid)
    if IMAGE_SIZE:
        url += f"?image_size={IMAGE_SIZE}"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return _image_white_background(img)
    except Exception:
        return None


def fetch_smiles(cid: int) -> Optional[str]:
    """Fetch SMILES for CID from PubChem (CanonicalSMILES or fallback to SMILES/ConnectivitySMILES); return None on failure."""
    url = PUBCHEM_SMILES_URL.format(cid=cid)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        first = props[0]
        for key in ("CanonicalSMILES", "SMILES", "IsomericSMILES", "ConnectivitySMILES"):
            if key in first and first[key]:
                return first[key]
        return None
    except Exception:
        return None


def download_all_images_and_smiles(
    by_journal: Dict[str, List[Tuple[int, str]]],
) -> Tuple[Dict[int, Optional[Image.Image]], Dict[int, Optional[str]]]:
    """Download structure image and SMILES for each unique CID; return (cid->image, cid->smiles)."""
    all_cids = set()
    for items in by_journal.values():
        for cid, _ in items:
            all_cids.add(cid)
    cid_to_image: Dict[int, Optional[Image.Image]] = {}
    cid_to_smiles: Dict[int, Optional[str]] = {}
    for cid in tqdm(sorted(all_cids), desc="Downloading structure images and SMILES"):
        cid_to_image[cid] = fetch_structure_image(cid)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        cid_to_smiles[cid] = fetch_smiles(cid)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return cid_to_image, cid_to_smiles


def save_cid_smiles_csv(
    by_journal: Dict[str, List[Tuple[int, str]]],
    cid_to_smiles: Dict[int, Optional[str]],
    out_path: str,
) -> None:
    """Write CSV of selected molecules: molecule_name, cid, smiles, journal. Quote all fields so names with commas stay one column."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["molecule_name", "cid", "smiles", "journal"])
        for journal in JOURNALS:
            if journal not in by_journal:
                continue
            for cid, name in by_journal[journal]:
                smiles = cid_to_smiles.get(cid) or ""
                w.writerow([name, str(cid), smiles, journal])


def _display_label(molecule_name: str) -> str:
    """Use abbreviation in trailing parentheses if present, else full name."""
    if not molecule_name or not molecule_name.strip():
        return molecule_name or ""
    name = molecule_name.strip()
    # Trailing " (abbreviation)" with a space before the opening paren
    m = re.match(r"^(.+)\s+\(([^)]*)\)\s*$", name)
    if m:
        return m.group(2).strip() or name
    return name


def _truncate_label(name: str, max_chars: int = 35) -> str:
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 3] + "..."


def plot_journal_figure(
    journal: str,
    items: List[Tuple[int, str]],
    cid_to_image: Dict[int, Optional[Image.Image]],
    out_dir: str,
) -> None:
    """Create one or more figures per journal: grid of subplots, each showing molecule image + name."""
    safe_name = journal.replace(" ", "_")
    start = 0
    part = 0
    while start < len(items):
        chunk = items[start : start + MAX_SUBPLOTS_PER_FIGURE]
        n = len(chunk)
        ncols = min(SUBPLOT_COLS, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), facecolor="white")
        for ax in (axes.flatten() if n > 1 else [axes]):
            ax.set_facecolor("white")
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for idx, (cid, name) in enumerate(chunk):
            ax = axes[idx]
            ax.set_facecolor("white")
            ax.set_axis_off()
            img = cid_to_image.get(cid)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=10)
            label = _display_label(name)
            ax.set_title(_truncate_label(label), fontsize=FONT_SIZE_NAME, wrap=True)
        for j in range(idx + 1, len(axes)):
            axes[j].set_facecolor("white")
            axes[j].set_axis_off()
        title = journal if part == 0 else f"{journal} (part {part + 1})"
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        fname = f"molecules_{safe_name}.png" if part == 0 else f"molecules_{safe_name}_{part + 1}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        start += MAX_SUBPLOTS_PER_FIGURE
        part += 1


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_CSV)
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading CSV and grouping by journal...")
    by_journal = load_molecules_by_journal(input_path)
    # Restrict to requested journals in order
    by_journal = {j: by_journal[j] for j in JOURNALS if j in by_journal}
    total = sum(len(v) for v in by_journal.values())
    print(f"Found {total} unique molecules across {len(by_journal)} journals.")

    print("Downloading structure images and SMILES from PubChem...")
    cid_to_image, cid_to_smiles = download_all_images_and_smiles(by_journal)

    csv_path = os.path.join(out_dir, OUTPUT_CID_SMILES_CSV)
    save_cid_smiles_csv(by_journal, cid_to_smiles, csv_path)
    print(f"Saved CID/SMILES CSV: {csv_path}")

    print("Creating combined figures per journal...")
    for journal in JOURNALS:
        if journal not in by_journal:
            continue
        items = by_journal[journal]
        plot_journal_figure(journal, items, cid_to_image, out_dir)
        print(f"  {journal}: {len(items)} molecules")

    print(f"Done. Figures and CSV saved in {out_dir}")


if __name__ == "__main__":
    main()
