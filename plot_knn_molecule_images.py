"""
Extract 2D molecule images from knn_embedding_results.csv and combine them
in a layout per query: the query molecule sits on the left (spanning both rows)
with its abbreviation label; 20 reference neighbors fill a 2x10 grid to its right.

Usage:
    python plot_knn_molecule_images.py --input knn_embedding_results.csv --output_dir ./knn_images
"""
import argparse
import csv
import json
import os
import re
import time
from urllib.request import Request, urlopen
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

# Default paths
DEFAULT_INPUT_CSV = "knn_finetunedembedding_results.csv"
DEFAULT_OUTPUT_DIR = "knn_images_finetuned"
MOL_IMAGE_SIZE = (280, 280)
GRID_ROWS, GRID_COLS = 2, 10
NUM_REFERENCE = 20  # 1 query + 20 refs = 21 total
GAP_COL = 24  # Horizontal gap between adjacent ref columns
GAP_QUERY = 48  # Horizontal gap between the query column and the ref grid
PANELS_PER_IMAGE = 3
PANEL_GAP = 80

try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception:
    sascorer = None


def abbreviation_from_name(query_name: str) -> str:
    """Extract abbreviation from end of query_name: text in the last parentheses, e.g. '... (BAE)' -> 'BAE'."""
    if not query_name or not isinstance(query_name, str):
        return ""
    s = query_name.strip()
    idx = s.rfind("(")
    if idx != -1 and ")" in s[idx:]:
        end = s.index(")", idx)
        return s[idx + 1 : end].strip()
    return s


def sanitize_filename(s: str, max_len: int = 80) -> str:
    """Replace invalid filename characters; truncate if needed."""
    s = re.sub(r'[<>:"/\\|?*\n\r]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s or "unnamed"


def smiles_to_image(smiles: str, size: Tuple[int, int] = MOL_IMAGE_SIZE) -> Optional[Image.Image]:
    """Draw 2D structure from SMILES; return PIL Image or None."""
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        return img
    except Exception:
        return None


def smiles_to_sa_score(smiles: str) -> Optional[float]:
    """Compute RDKit synthetic accessibility (SA) score for a SMILES."""
    if not smiles or sascorer is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def _normalize_row(row: dict) -> dict:
    """Strip keys (and BOM on first key) and strip string values for reliable column access."""
    out = {}
    for k, v in row.items():
        key = k.strip().lstrip("\ufeff")
        out[key] = (v.strip() if isinstance(v, str) else v) if v is not None else ""
    return out


def load_and_group_csv(csv_path: str) -> List[dict]:
    """
    Load knn_embedding_results.csv and group rows by query.
    Returns list of groups; each group is a dict:
      query_name, query_cid, query_smiles, query_journal,
      refs: list of {rank, ref_cid, ref_smiles, ref_status, distance}
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [_normalize_row(r) for r in reader]

    # Group by query_cid (unique per query)
    groups_by_cid = defaultdict(list)
    for row in rows:
        cid = (row.get("query_cid") or "").strip()
        if not cid:
            continue
        groups_by_cid[cid].append(row)

    out = []
    for cid, group_rows in groups_by_cid.items():
        # Sort by rank (integer so 10 comes after 9)
        def rank_key(r):
            try:
                return int((r.get("rank") or "0").strip() or 0)
            except (ValueError, TypeError):
                return 0

        group_rows = sorted(group_rows, key=rank_key)
        first = group_rows[0]
        query_smiles = (first.get("query_smiles") or "").strip()
        query_cid = cid
        refs = []
        for r in group_rows:
            refs.append({
                "rank": (r.get("rank") or "").strip(),
                "ref_cid": (r.get("ref_cid") or "").strip(),
                "ref_smiles": (r.get("ref_smiles") or "").strip(),
                "ref_status": (r.get("ref_status") or "").strip(),
                "distance": (r.get("distance") or "").strip(),
            })
        out.append({
            "query_name": (first.get("query_name") or "").strip(),
            "query_cid": query_cid,
            "query_smiles": query_smiles,
            "query_journal": (first.get("query_journal") or "").strip(),
            "refs": refs,
        })
    return out


def _load_font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _safe_float_text(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fetch_vendor_map(cids: List[str], timeout_s: float = 6.0, sleep_s: float = 0.10) -> Dict[str, bool]:
    """
    Check if each CID has vendor information in PubChem PUG-View.
    Returns dict: cid -> bool.
    """
    out: Dict[str, bool] = {}
    unique_cids = sorted({c.strip() for c in cids if c and str(c).strip().isdigit()})
    if not unique_cids:
        return out

    for cid in tqdm(unique_cids, desc="Checking PubChem vendors"):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        has_vendor = False
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            text_blob = json.dumps(payload)
            has_vendor = ("Chemical Vendors" in text_blob) or ("Vendors" in text_blob)
        except Exception:
            has_vendor = False
        out[cid] = has_vendor
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


def make_grid_image(
    query_name: str,
    query_smiles: str,
    query_cid: str,
    ref_smiles_list: List[str],
    ref_cid_list: List[str],
    ref_vendor_flags: Optional[List[bool]] = None,
    mol_size: Tuple[int, int] = MOL_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """
    Build layout: query on the left (spanning both rows) with abbreviation + CID + SA;
    refs 1..20 fill a GRID_ROWS x GRID_COLS (2x10) grid to the right, each labeled with CID + SA.
    Refs with vendor info receive a green "V" marker.
    Returns single PIL Image.
    """
    from PIL import ImageDraw

    w, h = mol_size
    # Font sizes for labels
    query_name_font_size = 44
    cid_font_size = 32
    sa_font_size = 32
    # Label area below each molecule image (query gets 3 lines, refs 2 lines)
    ref_label_height = cid_font_size + sa_font_size + 20
    query_label_height = query_name_font_size + cid_font_size + sa_font_size + 24
    label_height = max(ref_label_height, query_label_height)
    cell_w, cell_h = w, h + label_height
    ref_grid_w = GRID_COLS * cell_w + (GRID_COLS - 1) * GAP_COL
    ref_grid_h = GRID_ROWS * cell_h
    query_col_w = cell_w  # query column has the same width as a ref cell
    grid_w = query_col_w + GAP_QUERY + ref_grid_w
    grid_h = ref_grid_h

    # White background
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font_query = _load_font(query_name_font_size)
    font_ref = _load_font(cid_font_size)
    font_sa = _load_font(sa_font_size)

    # Query centered vertically in the left column
    query_img = smiles_to_image(query_smiles, mol_size)
    query_sa = smiles_to_sa_score(query_smiles)
    if query_img is None:
        query_img = Image.new("RGB", mol_size, (240, 240, 240))
    query_y = (grid_h - cell_h) // 2
    grid.paste(query_img, (0, query_y))
    try:
        label = abbreviation_from_name(query_name) or query_name
        if len(label) > 20:
            label = label[:17] + "..."
        draw.text((5, query_y + h + 4), label, fill=(0, 0, 0), font=font_query)
        if query_cid:
            draw.text((5, query_y + h + 4 + query_name_font_size + 4), f"CID {query_cid}", fill=(60, 60, 60), font=font_ref)
        draw.text(
            (5, query_y + h + 4 + query_name_font_size + cid_font_size + 10),
            f"SA {_safe_float_text(query_sa)}",
            fill=(60, 60, 60),
            font=font_sa,
        )
    except Exception:
        pass

    # Refs fill the 2x10 grid to the right of the query, row-major
    ref_x0 = query_col_w + GAP_QUERY
    col_x = lambda c: ref_x0 + c * (cell_w + GAP_COL)
    for idx, smi in enumerate(ref_smiles_list[:NUM_REFERENCE]):
        ref_img = smiles_to_image(smi, mol_size)
        if ref_img is None:
            ref_img = Image.new("RGB", mol_size, (240, 240, 240))
        r, c = divmod(idx, GRID_COLS)
        if r >= GRID_ROWS:
            break
        x = col_x(c)
        y = r * cell_h
        grid.paste(ref_img, (x, y))
        ref_cid = ref_cid_list[idx] if idx < len(ref_cid_list) else ""
        ref_sa = smiles_to_sa_score(smi)
        try:
            if ref_cid:
                draw.text((x + 5, y + h + 2), f"CID {ref_cid}", fill=(0, 0, 0), font=font_ref)
            draw.text((x + 5, y + h + 2 + cid_font_size + 4), f"SA {_safe_float_text(ref_sa)}", fill=(50, 50, 50), font=font_sa)
        except Exception:
            pass
        if ref_vendor_flags and idx < len(ref_vendor_flags) and ref_vendor_flags[idx]:
            # Green "V" marker for refs with vendor info in PubChem.
            marker_x0, marker_y0 = x + 8, y + 8
            marker_x1, marker_y1 = marker_x0 + 28, marker_y0 + 28
            draw.ellipse([marker_x0, marker_y0, marker_x1, marker_y1], fill=(46, 160, 67), outline=(0, 90, 24), width=2)
            draw.text((marker_x0 + 8, marker_y0 + 3), "V", fill=(255, 255, 255), font=_load_font(18))

    return grid


def combine_panels(panel_images: List[Image.Image], gap: int = PANEL_GAP) -> Image.Image:
    """Stack multiple query+ref panels vertically into one image."""
    if not panel_images:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    width = max(img.width for img in panel_images)
    height = sum(img.height for img in panel_images) + gap * (len(panel_images) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for img in panel_images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Plot query + 2x10 ref molecule grids from k-NN result CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Path to knn_embedding_results.csv")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for images")
    parser.add_argument("--size", type=int, default=280, help="Molecule image width/height in pixels")
    parser.add_argument("--panels_per_image", type=int, default=PANELS_PER_IMAGE, help="How many query+ref panels to combine in one output image")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    print(f"Loading {args.input}...")
    groups = load_and_group_csv(args.input)
    print(f"Found {len(groups)} query molecule(s)")

    os.makedirs(args.output_dir, exist_ok=True)
    size = (args.size, args.size)

    # Pre-check vendor availability for all potential reference CIDs.
    all_ref_cids: List[str] = []
    for g in groups:
        for r in g["refs"]:
            all_ref_cids.append((r.get("ref_cid") or "").strip())
    vendor_map = fetch_vendor_map(all_ref_cids)

    panel_images: List[Image.Image] = []
    panel_names: List[str] = []
    image_counter = 1

    for g in tqdm(groups, desc="Plotting"):
        query_name = g["query_name"]
        query_smiles = g["query_smiles"]
        query_cid = g["query_cid"]
        # Use refs with rank 2..21 (rank 1 is the query itself, distance 0)
        def _rank_int(r):
            try:
                return int((r.get("rank") or "").strip())
            except (ValueError, TypeError):
                return -1

        selected_refs = [
            r for r in g["refs"]
            if 2 <= _rank_int(r) <= 1 + NUM_REFERENCE
        ]
        ref_smiles_list = [r["ref_smiles"] for r in selected_refs]
        ref_cid_list = [r["ref_cid"] for r in selected_refs]
        ref_vendor_flags = [vendor_map.get(cid.strip(), False) for cid in ref_cid_list]

        try:
            grid_img = make_grid_image(
                query_name, query_smiles, query_cid,
                ref_smiles_list, ref_cid_list, ref_vendor_flags=ref_vendor_flags, mol_size=size,
            )
        except Exception as e:
            print(f"  Error plotting {query_name or query_cid}: {e}")
            continue
        if grid_img is None:
            print(f"  Skip (no image): {query_name or query_cid}")
            continue

        panel_images.append(grid_img)
        panel_names.append(sanitize_filename(f"{query_name}_{query_cid}", max_len=30))

        if len(panel_images) >= max(1, int(args.panels_per_image)):
            combined = combine_panels(panel_images)
            tag = "__".join(panel_names[:3]) if panel_names else f"set_{image_counter:03d}"
            out_path = os.path.join(args.output_dir, f"combined_{image_counter:03d}_{sanitize_filename(tag, max_len=90)}.png")
            try:
                combined.save(out_path)
                print(f"  Saved {out_path}")
            except Exception as e:
                print(f"  Error saving {out_path}: {e}")
            image_counter += 1
            panel_images = []
            panel_names = []

    if panel_images:
        combined = combine_panels(panel_images)
        tag = "__".join(panel_names[:3]) if panel_names else f"set_{image_counter:03d}"
        out_path = os.path.join(args.output_dir, f"combined_{image_counter:03d}_{sanitize_filename(tag, max_len=90)}.png")
        try:
            combined.save(out_path)
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  Error saving {out_path}: {e}")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
