"""
Statistics for the perovskite-molecule binding-energy dataset.

Per-row layout (merged downstream CSV):
    cid, SMILES, functional_group, formula,
    pb_bond_encoding (list 0/1, one entry per atom),
    adsorption_energy,
    surface_tag,
    adsorbate_structure (JSON with coords.3d and elements.number).

Two outputs:

(1) Violin plot of binding energy grouped by the SET of anchoring atom elements
    restricted to {N, O, S, P}. A molecule contributes to the group named by the
    sorted concatenation of its anchoring elements in that set, e.g. "N", "O",
    "S", "P", "N+O", "N+O+S", etc.

(2) Four violin plots — one per element in {N, O, S, P}. Each plot shows the
    binding energy distribution across functional-group categories. ONLY
    molecules with EXACTLY one anchoring atom in {N, O, S, P} are considered.
    Other anchors outside this set (e.g. H, C, Cl) are allowed and do not
    disqualify the molecule — for example a molecule whose pb_bond_encoding
    flags one N and one H still qualifies. But molecules with two or more
    anchors in {N, O, S, P} (e.g. N+O, N+N, O+S) are dropped from plots 2-5.

    The functional group is NOT taken from the merged CSV `functional_group`
    column. Instead, RDKit matches SMARTS patterns from
    ``dataset/prediction/funct_group.csv`` (columns: functional group, chemical
    structure, SMILES, SMARTS) so labels match that table. The anchor atom must
    lie in the matched subgraph.

    Pipeline:
      - Parse SMILES into a heavy-atom mol.
      - Map the DFT anchor index onto the mol (trivial element match or Hungarian
        geometric match using adsorbate_structure 3D coords).
      - Find SMARTS hits from funct_group.csv that include the anchor atom;
        prefer longer SMARTS / larger matches to resolve overlaps.

    Labels ``other`` (no SMARTS match) and ``ambiguous_*`` (unresolved anchor /
    conflicting labels) are excluded from violin plots 2-5 and their per-group
    stats CSVs.

Usage:
    python analyze_binding_anchors.py
    python analyze_binding_anchors.py --include_extra
    python analyze_binding_anchors.py --input <csv> --output_dir <dir>
    python analyze_binding_anchors.py --funct_group_csv dataset/prediction/funct_group.csv
    python analyze_binding_anchors.py --min_group_size 5
    python analyze_binding_anchors.py --publication

With ``--include_extra``, rows from ``config.downstream_extra_csv`` are appended
(energy < -1.3 eV or in [-0.6, 0] eV). Anchors are inferred from SMILES as all
heavy N/O/S/P atoms (no DFT ``pb_bond_encoding`` in the extra file).
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from config import Config  # type: ignore
    _cfg = Config()
    _DEFAULT_INPUT = (_cfg.downstream_csv or "").strip()
    _DEFAULT_EXTRA_CSV = (_cfg.downstream_extra_csv or "").strip()
except Exception:
    _DEFAULT_INPUT = ""
    _DEFAULT_EXTRA_CSV = ""

# Extra CSV (train_downstream): best_adsorption_energy + SMILES only
EXTRA_ENERGY_COL_CANDIDATES = [
    "best_adsorption_energy",
    "adsorption_energy",
    "binding_energy",
]
EXTRA_STRONG_THRESHOLD_EV = -1.3
EXTRA_WEAK_LOW_EV = -0.6
EXTRA_WEAK_HIGH_EV = 0.0

# Optional dependencies for anchor-atom functional-group detection.
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import RDLogger  # type: ignore
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    _RDKIT_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    linear_sum_assignment = None  # type: ignore
    _SCIPY_AVAILABLE = False

ANCHOR_ELEMENTS: Dict[int, str] = {7: "N", 8: "O", 15: "P", 16: "S"}
ANCHOR_ORDER: List[str] = ["N", "O", "S", "P"]

# Violin plot typography (plots 1 and 2-5)
VIOLIN_TITLE_FONTSIZE = 15
VIOLIN_AXIS_LABEL_FONTSIZE = 14
VIOLIN_TICK_LABEL_FONTSIZE = 13

# Muted, publication-friendly violin palette (fills + edges)
VIOLIN_FILL_ANCHOR_COMBINATIONS = "#6B9AA8"  # dusty blue-teal
VIOLIN_FILL_DEFAULT = "#7A8FA3"  # blue-gray fallback
VIOLIN_FILL_BY_ELEMENT: Dict[str, str] = {
    "N": "#3D5A80",  # deep slate blue
    "O": "#B87A6E",  # muted terracotta
    "S": "#7D6B8C",  # dusty purple
    "P": "#B89B4A",  # antique gold
}
VIOLIN_BODY_EDGE = "#2A3238"  # blue-black outline
VIOLIN_REF_LINE = "#A8B0B8"  # soft neutral for y=0


def publication_rcparams() -> Dict[str, object]:
    """Matplotlib rc settings for print / journal figures (used with ``plt.rc_context``)."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "sans-serif",
        ],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    }


def save_violin_figure(fig, path_png: str, publication: bool) -> List[str]:
    """Save PNG (dpi 200 or 300); if publication, also save a vector PDF next to it."""
    out: List[str] = []
    dpi = 300 if publication else 200
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    out.append(path_png)
    if publication:
        pdf_path = os.path.splitext(path_png)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
        out.append(pdf_path)
    return out


def parse_tags(s: str) -> Optional[List[int]]:
    """Parse the pb_bond_encoding list (e.g. '[0,0,1,0,1]') into a list of 0/1."""
    if not s:
        return None
    try:
        out = ast.literal_eval(str(s).strip())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(out, list):
        return None
    try:
        out = [int(x) for x in out]
    except (TypeError, ValueError):
        return None
    if any(x not in (0, 1) for x in out):
        return None
    return out


def parse_adsorbate_structure(s: str) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
    """Parse adsorbate_structure JSON into (atomic_numbers, coords[n,3]).

    ``coords['3d']`` may be either a flat ``[x0,y0,z0, …]`` list (length ``3*n``)
    or a nested list/array of shape ``(n, 3)`` — both match the merged downstream
    format used elsewhere in this repo. Either component may be ``None`` if missing
    or malformed.
    """
    if not s:
        return None, None
    text = str(s).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None, None
    if not isinstance(obj, dict):
        return None, None

    numbers: Optional[List[int]] = None
    elements = obj.get("elements")
    if isinstance(elements, dict):
        raw_numbers = elements.get("number")
        if isinstance(raw_numbers, list):
            try:
                numbers = [int(z) for z in raw_numbers]
            except (TypeError, ValueError):
                numbers = None

    coords: Optional[np.ndarray] = None
    coords_block = obj.get("coords")
    if isinstance(coords_block, dict):
        raw_coords = coords_block.get("3d")
        if isinstance(raw_coords, list) and raw_coords:
            try:
                arr = np.asarray(raw_coords, dtype=np.float64)
            except (TypeError, ValueError):
                arr = None
            if arr is not None:
                if arr.ndim == 2 and arr.shape[1] == 3:
                    coords = arr
                elif arr.ndim == 1 and numbers is not None:
                    flat = np.ravel(arr)
                    n = len(numbers)
                    if flat.size == 3 * n:
                        coords = flat.reshape(n, 3)
    return numbers, coords


def parse_atomic_numbers(s: str) -> Optional[List[int]]:
    """Backwards-compatible wrapper that returns just elements.number."""
    numbers, _ = parse_adsorbate_structure(s)
    return numbers


def extra_energy_in_training_ranges(energy: float) -> bool:
    """Match ``train_downstream.load_extra_smiles_csv`` energy windows."""
    return energy < EXTRA_STRONG_THRESHOLD_EV or (
        EXTRA_WEAK_LOW_EV <= energy <= EXTRA_WEAK_HIGH_EV
    )


def anchor_combo_label(symbols: List[str]) -> str:
    """Stable label for a set of anchor element symbols, ordered N, O, S, P."""
    present = [e for e in ANCHOR_ORDER if e in symbols]
    return "+".join(present)


# ---------------------------------------------------------------------------
# Anchor-atom functional-group detection (RDKit + DFT-coord assisted mapping)
# ---------------------------------------------------------------------------

# Module-level cache to avoid re-parsing / re-embedding the same SMILES.
_HEAVY_MOL_CACHE: Dict[str, "object"] = {}
_EMBED_MOL_CACHE: Dict[str, "object"] = {}


def _get_heavy_mol(smiles: str):
    """Return an RDKit heavy-atom mol for ``smiles``, cached."""
    if not _RDKIT_AVAILABLE or not smiles:
        return None
    if smiles in _HEAVY_MOL_CACHE:
        return _HEAVY_MOL_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _HEAVY_MOL_CACHE[smiles] = None
        return None
    mol_heavy = Chem.RemoveHs(mol)
    _HEAVY_MOL_CACHE[smiles] = mol_heavy
    return mol_heavy


def _get_embedded_mol(smiles: str):
    """Build mol from SMILES, add explicit Hs, embed 3D coords. Cached.

    Returns ``None`` if RDKit is unavailable or embedding fails.
    """
    if not _RDKIT_AVAILABLE or not smiles:
        return None
    if smiles in _EMBED_MOL_CACHE:
        return _EMBED_MOL_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _EMBED_MOL_CACHE[smiles] = None
        return None
    mol_h = Chem.AddHs(mol)
    embed_rc = -1
    strategies = (
        lambda: AllChem.EmbedMolecule(mol_h, randomSeed=42),
        lambda: AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=500),
        lambda: AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()),
        lambda: AllChem.EmbedMolecule(
            mol_h, useRandomCoords=True, randomSeed=42, maxAttempts=500
        ),
    )
    for fn in strategies:
        try:
            embed_rc = int(fn())
            if embed_rc == 0 and mol_h.GetNumConformers() > 0:
                break
        except Exception:
            embed_rc = -1
    if mol_h.GetNumConformers() == 0:
        _EMBED_MOL_CACHE[smiles] = None
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
    except Exception:
        pass
    _EMBED_MOL_CACHE[smiles] = mol_h
    return mol_h


def _hungarian_assign(cost: np.ndarray):
    """Minimum-cost assignment for a square cost matrix.

    Uses SciPy's Hungarian when available; otherwise brute force over
    permutations (only for k <= 8, typical for duplicate O/N/S/P counts).
    """
    cost = np.asarray(cost, dtype=np.float64)
    k = int(cost.shape[0])
    if cost.ndim != 2 or cost.shape[1] != k:
        return None
    if k == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return np.asarray(r, dtype=int), np.asarray(c, dtype=int)
    if k > 8:
        return None
    best_c = np.inf
    best_perm: Optional[Tuple[int, ...]] = None
    for perm in itertools.permutations(range(k)):
        s = float(sum(cost[i, perm[i]] for i in range(k)))
        if s < best_c:
            best_c = s
            best_perm = perm
    if best_perm is None:
        return None
    r = np.arange(k, dtype=int)
    c = np.asarray(best_perm, dtype=int)
    return r, c


def _hungarian_dft_to_canon(
    atomic_numbers: List[int], coords: np.ndarray, canon_mol
) -> Optional[Dict[int, int]]:
    """Hungarian per-element assignment mapping DFT idx -> canonical mol idx."""
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords - coords.mean(axis=0)
    conf = canon_mol.GetConformer()
    can_coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(canon_mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )
    can_coords = can_coords - can_coords.mean(axis=0)
    canon_zs = [a.GetAtomicNum() for a in canon_mol.GetAtoms()]
    mapping: Dict[int, int] = {}
    for z in set(atomic_numbers):
        dft_inds = [i for i, zz in enumerate(atomic_numbers) if zz == z]
        can_inds = [i for i, zz in enumerate(canon_zs) if zz == z]
        if len(dft_inds) != len(can_inds):
            return None
        d = np.linalg.norm(
            coords[dft_inds][:, None, :] - can_coords[can_inds][None, :, :], axis=2
        )
        sol = _hungarian_assign(d)
        if sol is None:
            return None
        r, c = sol
        for ri, ci in zip(r, c):
            mapping[int(dft_inds[int(ri)])] = int(can_inds[int(ci)])
    return mapping


def _canon_to_heavy_idx_map(canon_mol) -> Dict[int, int]:
    out: Dict[int, int] = {}
    counter = 0
    for atom in canon_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            out[atom.GetIdx()] = counter
            counter += 1
    return out


def _resolve_anchor_heavy_idx(
    smiles: str,
    dft_numbers: List[int],
    dft_coords: Optional[np.ndarray],
    dft_anchor_idx: int,
    anchor_z: int,
) -> Tuple[Optional["object"], Optional[int], str]:
    """Find the heavy-atom index in the SMILES mol that matches the DFT anchor.

    When the SMILES has several atoms of the anchor element, DFT 3D coordinates
    from ``adsorbate_structure`` (``coords['3d']``, flat or ``(n,3)``) are aligned
    to an embedded SMILES+Hs conformer via per-element Hungarian matching, then
    the DFT anchor index is mapped to a heavy-atom index.

    Returns ``(mol_heavy, heavy_idx, status)``. Status is one of:
        "trivial"  -- mol has exactly one heavy atom of this element.
        "geom"     -- disambiguated via Hungarian geometric mapping.
        "ambig"    -- multiple candidates and geometric mapping unavailable.
        "missing"  -- mol could not be built or has no atom of this element.
    """
    mol_heavy = _get_heavy_mol(smiles)
    if mol_heavy is None:
        return None, None, "missing"
    candidates = [
        a.GetIdx() for a in mol_heavy.GetAtoms() if a.GetAtomicNum() == anchor_z
    ]
    if not candidates:
        return None, None, "missing"
    if len(candidates) == 1:
        return mol_heavy, candidates[0], "trivial"

    if dft_coords is None or dft_numbers is None:
        return mol_heavy, None, "ambig"
    canon = _get_embedded_mol(smiles)
    if canon is None:
        return mol_heavy, None, "ambig"
    mapping = _hungarian_dft_to_canon(dft_numbers, dft_coords, canon)
    if mapping is None:
        return mol_heavy, None, "ambig"
    canon_idx = mapping.get(dft_anchor_idx)
    if canon_idx is None:
        return mol_heavy, None, "ambig"
    heavy_idx = _canon_to_heavy_idx_map(canon).get(canon_idx)
    if heavy_idx is None:
        return mol_heavy, None, "ambig"
    # Sanity: heavy atom in mol_heavy at heavy_idx should match anchor_z.
    if mol_heavy.GetAtomWithIdx(heavy_idx).GetAtomicNum() != anchor_z:
        return mol_heavy, None, "ambig"
    return mol_heavy, heavy_idx, "geom"


# Default path to the project's functional-group SMARTS table.
DEFAULT_FUNCT_GROUP_CSV = os.path.join(
    SCRIPT_DIR, "dataset", "prediction", "funct_group.csv"
)


def _normalize_fieldnames(fieldnames: Optional[List[str]]) -> Dict[str, str]:
    """Map stripped lower-case header -> original header string."""
    out: Dict[str, str] = {}
    for h in fieldnames or []:
        key = (h or "").strip().lstrip("\ufeff").lower()
        if key and key not in out:
            out[key] = h
    return out


def load_funct_group_patterns(csv_path: str) -> Tuple[List[Tuple[str, "object", str, int]], int, int]:
    """Load SMARTS rows from ``funct_group.csv``.

    Returns ``(patterns, n_ok, n_skip)`` where each pattern is
    ``(display_name, query_mol, smarts_str, source_row_1based)``.
    Patterns are sorted for matching: longer SMARTS first, then larger
    substructure size (see ``match_anchor_functional_group``).
    """
    if not _RDKIT_AVAILABLE or not csv_path or not os.path.isfile(csv_path):
        return [], 0, 0

    raw_rows: List[Tuple[str, str, int]] = []  # (name, smarts, row_1based)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()

    col_fg: Optional[str] = None
    col_smarts: Optional[str] = None
    for delim in ("\t", ","):
        buf = io.StringIO(text)
        reader = csv.DictReader(buf, delimiter=delim)
        fn_map = _normalize_fieldnames(reader.fieldnames)
        cand_fg = fn_map.get("functional group") or fn_map.get("functional_group")
        cand_smarts = fn_map.get("smarts")
        if cand_fg and cand_smarts:
            col_fg, col_smarts = cand_fg, cand_smarts
            for row_idx, row in enumerate(reader, start=2):
                if not row:
                    continue
                smarts = (row.get(col_smarts) or "").strip()
                if not smarts:
                    continue
                name = (row.get(col_fg) or "").strip()
                if not name:
                    name = smarts
                raw_rows.append((name, smarts, row_idx))
            break

    if not col_fg or not col_smarts or not raw_rows:
        print(
            "WARNING: could not read funct_group CSV (need tab or comma delimiter and "
            "'functional group' + 'SMARTS' columns): {}".format(csv_path)
        )
        return [], 0, 0

    patterns: List[Tuple[str, "object", str, int]] = []
    n_skip = 0
    for name, smarts, row_1based in raw_rows:
        try:
            qmol = Chem.MolFromSmarts(smarts)
        except Exception:
            qmol = None
        if qmol is None:
            print(
                "WARNING: skipping SMARTS (parse failed) row {}: {}".format(
                    row_1based, smarts[:80]
                )
            )
            n_skip += 1
            continue
        patterns.append((name, qmol, smarts, row_1based))

    n_ok = len(patterns)
    return patterns, n_ok, n_skip


def match_anchor_functional_group(
    mol,
    heavy_idx: int,
    patterns: List[Tuple[str, "object", str, int]],
) -> str:
    """Return the functional-group label from ``funct_group.csv`` for ``heavy_idx``.

    Considers every SMARTS hit whose atom index set contains ``heavy_idx`` and
    picks the hit with the largest match size, then longest SMARTS string, then
    highest source row (later rows beat ties — generics tend to appear earlier).
    """
    if not patterns or mol is None:
        return "other"

    best: Optional[Tuple[int, int, int, str]] = None
    # tuple: (match_size, smarts_len, row, label) — maximize lexicographically

    for label, qmol, smarts, row_1based in patterns:
        if qmol is None:
            continue
        try:
            matches = mol.GetSubstructMatches(qmol, uniquify=True, useChirality=False)
        except Exception:
            continue
        slen = len(smarts)
        for match in matches:
            if heavy_idx not in match:
                continue
            key = (len(match), slen, row_1based, label)
            if best is None or key > best:
                best = key

    return best[3] if best else "other"


def detect_anchor_functional_group(
    smiles: str,
    anchor_symbol: str,
    dft_numbers: Optional[List[int]],
    dft_coords: Optional[np.ndarray],
    dft_anchor_idx: Optional[int],
    funct_patterns: List[Tuple[str, "object", str, int]],
) -> Tuple[str, str]:
    """Detect the functional group at the anchoring atom via SMARTS table.

    Returns ``(label, status)`` where status is one of
    ``"trivial" | "geom" | "consensus" | "ambig" | "missing" | "no_rdkit" | "no_patterns"``.
    """
    if not _RDKIT_AVAILABLE:
        return "no_rdkit", "no_rdkit"
    if not funct_patterns:
        return "no_patterns", "no_patterns"
    if not smiles:
        return "missing", "missing"
    anchor_z = {"N": 7, "O": 8, "S": 16, "P": 15}.get(anchor_symbol)
    if anchor_z is None:
        return "missing", "missing"

    mol_heavy, heavy_idx, status = _resolve_anchor_heavy_idx(
        smiles,
        dft_numbers or [],
        dft_coords,
        dft_anchor_idx if dft_anchor_idx is not None else -1,
        anchor_z,
    )
    if mol_heavy is None:
        return "missing", status
    if heavy_idx is not None:
        return match_anchor_functional_group(mol_heavy, heavy_idx, funct_patterns), status

    candidates = [a for a in mol_heavy.GetAtoms() if a.GetAtomicNum() == anchor_z]
    raw_labels = {
        match_anchor_functional_group(mol_heavy, a.GetIdx(), funct_patterns) for a in candidates
    }
    if len(raw_labels) == 1:
        return raw_labels.pop(), "consensus"
    non_other = raw_labels - {"other"}
    if len(non_other) == 1:
        return next(iter(non_other)), "consensus"
    if not non_other:
        return "other", "consensus"
    return "ambiguous_{}".format(anchor_symbol), "ambig"


def load_rows(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lstrip("\ufeff") for h in (reader.fieldnames or [])]
        rows = list(reader)
    return rows, fieldnames


def resolve_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first matching column from candidates (case-insensitive)."""
    lookup = {h.lower(): h for h in fieldnames}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def collect_records(
    rows: List[Dict[str, str]], fieldnames: List[str]
) -> Tuple[List[dict], Dict[str, int]]:
    """Build per-molecule records and a count of skip reasons."""
    skipped: Counter = Counter()

    col_cid = resolve_column(fieldnames, ["cid", "CID"])
    col_smiles = resolve_column(fieldnames, ["SMILES", "smiles", "canonical_smiles"])
    col_energy = resolve_column(fieldnames, ["adsorption_energy", "binding_energy", "energy"])
    col_tags = resolve_column(fieldnames, ["pb_bond_encoding", "binding_tag", "binding_tags"])
    col_struct = resolve_column(fieldnames, ["adsorbate_structure", "structure"])
    col_func = resolve_column(fieldnames, ["functional_group", "func_group", "function_group"])

    missing = [
        name
        for name, col in [
            ("energy", col_energy),
            ("tags", col_tags),
            ("structure", col_struct),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            "Missing required column(s) {}. Found columns: {}".format(missing, fieldnames)
        )

    records: List[dict] = []
    for row in rows:
        try:
            energy = float((row.get(col_energy or "") or "").strip())
        except (TypeError, ValueError):
            skipped["bad_energy"] += 1
            continue
        if not np.isfinite(energy):
            skipped["bad_energy"] += 1
            continue

        tags = parse_tags(row.get(col_tags or "", ""))
        if tags is None:
            skipped["bad_tags"] += 1
            continue
        numbers, coords = parse_adsorbate_structure(row.get(col_struct or "", ""))
        if numbers is None:
            skipped["bad_structure"] += 1
            continue
        if len(tags) != len(numbers):
            skipped["tag_length_mismatch"] += 1
            continue
        if coords is not None and coords.shape[0] != len(numbers):
            coords = None

        anchor_indices_dft = [i for i, t in enumerate(tags) if t == 1]
        anchor_z = [numbers[i] for i in anchor_indices_dft]
        anchor_symbols = [ANCHOR_ELEMENTS[z] for z in anchor_z if z in ANCHOR_ELEMENTS]
        anchor_indices_in_set = [
            i for i, z in zip(anchor_indices_dft, anchor_z) if z in ANCHOR_ELEMENTS
        ]

        func_group = ""
        if col_func is not None:
            func_group = (row.get(col_func) or "").strip()

        smiles = ""
        if col_smiles is not None:
            smiles = (row.get(col_smiles) or "").strip()

        records.append(
            {
                "cid": (row.get(col_cid or "") or "").strip() if col_cid else "",
                "smiles": smiles,
                "energy": energy,
                "n_total_anchors": int(sum(tags)),
                "n_anchors_in_set": len(anchor_symbols),
                "anchor_symbols": anchor_symbols,
                "anchor_indices_dft": anchor_indices_in_set,
                "dft_numbers": numbers,
                "dft_coords": coords,
                "functional_group_csv": func_group,
            }
        )

    return records, dict(skipped)


def _heavy_mol_atomic_numbers_and_coords(
    smiles: str,
) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
    """Atomic numbers and 3D coords (heavy atoms) from an embedded SMILES conformer."""
    if not _RDKIT_AVAILABLE or not smiles.strip():
        return None, None
    canon = _get_embedded_mol(smiles)
    if canon is None:
        mol_heavy = _get_heavy_mol(smiles)
        if mol_heavy is None:
            return None, None
        numbers = [a.GetAtomicNum() for a in mol_heavy.GetAtoms()]
        return numbers, None
    heavy_map = _canon_to_heavy_idx_map(canon)
    if not heavy_map:
        return None, None
    conf = canon.GetConformer()
    heavy_idx_sorted = sorted(heavy_map.keys())
    numbers = [canon.GetAtomWithIdx(ci).GetAtomicNum() for ci in heavy_idx_sorted]
    coords = np.array(
        [conf.GetAtomPosition(ci) for ci in heavy_idx_sorted],
        dtype=np.float64,
    )
    return numbers, coords


def collect_records_from_extra_csv(csv_path: str) -> Tuple[List[dict], Dict[str, int]]:
    """Build records from extra SMILES CSV (no DFT tags / adsorbate_structure).

    Anchoring elements are all heavy N/O/S/P atoms in the SMILES graph, matching the
    training-set extension in ``train_downstream`` (binding tags absent on extra rows).
    """
    skipped: Counter = Counter()
    records: List[dict] = []

    if not _RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required to load extra SMILES CSV records.")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lstrip("\ufeff") for h in (reader.fieldnames or [])]
        col_energy = resolve_column(fieldnames, EXTRA_ENERGY_COL_CANDIDATES)
        col_smiles = resolve_column(fieldnames, ["SMILES", "smiles", "canonical_smiles"])
        if col_energy is None:
            raise ValueError(
                "Extra CSV missing energy column (tried {}). Columns: {}".format(
                    EXTRA_ENERGY_COL_CANDIDATES, fieldnames
                )
            )
        if col_smiles is None:
            raise ValueError("Extra CSV missing SMILES column. Columns: {}".format(fieldnames))

        for row in reader:
            smiles = (row.get(col_smiles) or "").strip()
            if not smiles:
                skipped["missing_smiles"] += 1
                continue
            raw_e = (row.get(col_energy) or "").strip()
            if not raw_e:
                skipped["bad_energy"] += 1
                continue
            try:
                energy = float(raw_e)
            except (TypeError, ValueError):
                skipped["bad_energy"] += 1
                continue
            if not np.isfinite(energy):
                skipped["bad_energy"] += 1
                continue
            if not extra_energy_in_training_ranges(energy):
                skipped["energy_window"] += 1
                continue

            mol_heavy = _get_heavy_mol(smiles)
            if mol_heavy is None or mol_heavy.GetNumAtoms() < 1:
                skipped["bad_smiles"] += 1
                continue

            numbers, coords = _heavy_mol_atomic_numbers_and_coords(smiles)
            if numbers is None:
                skipped["bad_smiles"] += 1
                continue

            anchor_symbols = [
                ANCHOR_ELEMENTS[z] for z in numbers if z in ANCHOR_ELEMENTS
            ]

            records.append(
                {
                    "cid": "",
                    "smiles": smiles,
                    "energy": energy,
                    "n_total_anchors": len(anchor_symbols),
                    "n_anchors_in_set": len(anchor_symbols),
                    "anchor_symbols": anchor_symbols,
                    "anchor_indices_dft": [],
                    "dft_numbers": numbers,
                    "dft_coords": coords,
                    "functional_group_csv": "",
                    "source": "extra",
                }
            )

    return records, dict(skipped)


def _set_violin_style(parts, face_color: str, edge_color: str = VIOLIN_BODY_EDGE) -> None:
    for body in parts["bodies"]:
        body.set_facecolor(face_color)
        body.set_edgecolor(edge_color)
        body.set_alpha(0.88)
        body.set_linewidth(0.8)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        if key in parts:
            parts[key].set_color(edge_color)
            parts[key].set_linewidth(1.0)
    if "cmedians" in parts:
        med = parts["cmedians"]
        med.set_color(edge_color)
        med.set_linewidth(1.2)
        try:
            med.set_linestyles("--")
        except Exception:
            try:
                med.set_linestyle("--")
            except Exception:
                pass


def draw_violin(
    ax,
    groups: List[Tuple[str, List[float], int]],
    title: str,
    xlabel: str,
    color: str = VIOLIN_FILL_DEFAULT,
    *,
    show_median: bool = True,
    publication: bool = False,
) -> None:
    if not groups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = [vals for _, vals, _ in groups]
    labels = ["{}\n(n={})".format(name, n) for name, _, n in groups]
    positions = list(range(1, len(groups) + 1))
    parts = ax.violinplot(
        data,
        positions=positions,
        showmedians=show_median,
        showextrema=False,
        widths=0.8,
    )
    _set_violin_style(parts, color)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=VIOLIN_TICK_LABEL_FONTSIZE)
    ax.set_ylabel(
        "Adsorption energy (eV)",
        fontsize=VIOLIN_AXIS_LABEL_FONTSIZE,
    )
    ax.set_xlabel(xlabel, fontsize=VIOLIN_AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=VIOLIN_TITLE_FONTSIZE)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        labelsize=VIOLIN_TICK_LABEL_FONTSIZE,
    )
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(VIOLIN_TICK_LABEL_FONTSIZE)
    if publication:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.axhline(
        0.0,
        color=VIOLIN_REF_LINE,
        linewidth=0.55,
        linestyle="--",
        alpha=0.42 if publication else 0.5,
        zorder=0,
    )
    ax.grid(True, axis="y", alpha=0.16 if publication else 0.25, linewidth=0.55, zorder=0)


def write_group_stats_csv(
    path: str, groups: List[Tuple[str, List[float], int]], group_label: str
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([group_label, "n", "mean_eV", "median_eV", "std_eV", "min_eV", "max_eV"])
        for name, vals, n in groups:
            arr = np.asarray(vals, dtype=float)
            writer.writerow(
                [
                    name,
                    n,
                    f"{arr.mean():.6f}",
                    f"{np.median(arr):.6f}",
                    f"{arr.std(ddof=0):.6f}",
                    f"{arr.min():.6f}",
                    f"{arr.max():.6f}",
                ]
            )


def plot_anchor_combinations(
    records: List[dict],
    output_dir: str,
    min_group_size: int,
    publication: bool = False,
) -> Optional[str]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        if rec["n_anchors_in_set"] == 0:
            continue
        label = anchor_combo_label(rec["anchor_symbols"])
        if not label:
            continue
        grouped[label].append(rec["energy"])

    items = [
        (label, vals, len(vals))
        for label, vals in grouped.items()
        if len(vals) >= min_group_size
    ]
    items.sort(key=lambda x: (-x[2], x[0]))
    if not items:
        print("No anchor combinations met --min_group_size threshold.")
        return None

    fig_h = 7.0 if publication else 6.0
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(items)), fig_h))
    draw_violin(
        ax,
        items,
        title="Binding energy by N/O/S/P anchoring atom combination",
        xlabel="Anchoring atom set",
        color=VIOLIN_FILL_ANCHOR_COMBINATIONS,
        publication=publication,
    )
    fig.tight_layout()
    out_png = os.path.join(output_dir, "violin_anchor_combinations.png")
    for p in save_violin_figure(fig, out_png, publication):
        print("  Saved {}".format(p))
    plt.close(fig)
    write_group_stats_csv(
        os.path.join(output_dir, "stats_anchor_combinations.csv"),
        items,
        group_label="anchor_set",
    )
    return out_png


def _exclude_from_single_anchor_violin_fg(fg: str) -> bool:
    """Catch-all and unresolved labels are not shown in violin plots 2-5."""
    if not fg:
        return True
    if fg == "other":
        return True
    return fg.startswith("ambiguous_")


def _is_single_NOSP_anchor(rec: dict) -> bool:
    """Plots 2-5 are restricted to molecules with EXACTLY one anchoring atom
    whose element is in {N, O, S, P}.

    Other anchoring atoms outside this set (H, C, Cl, ...) are allowed and do
    NOT disqualify the molecule. Two or more anchors that are themselves in
    {N, O, S, P} (e.g. N+O, N+N, O+S) DO disqualify the molecule.
    """
    return rec.get("n_anchors_in_set") == 1


def plot_single_anchor_by_functional(
    records: List[dict],
    output_dir: str,
    min_group_size: int,
    funct_patterns: List[Tuple[str, "object", str, int]],
    publication: bool = False,
) -> List[str]:
    if not _RDKIT_AVAILABLE:
        print(
            "WARNING: RDKit is not available; cannot detect anchor functional groups."
            " Skipping plots 2-5."
        )
        return []
    if not funct_patterns:
        print(
            "WARNING: no SMARTS patterns loaded from funct_group.csv; skipping plots 2-5."
        )
        return []

    eligible = [r for r in records if _is_single_NOSP_anchor(r)]
    by_nosp = Counter(int(r.get("n_anchors_in_set", 0)) for r in records)
    print(
        "  Plot 2-5 filter: keeping molecules with exactly one N/O/S/P anchor "
        "(other anchors like H/Cl OK): {} of {} (N/O/S/P-anchor distribution: {}).".format(
            len(eligible),
            len(records),
            dict(sorted(by_nosp.items())),
        )
    )

    by_element: Dict[str, Dict[str, List[float]]] = {
        elem: defaultdict(list) for elem in ANCHOR_ORDER
    }
    status_counts: Counter = Counter()
    n_considered = 0
    for rec in eligible:
        n_considered += 1
        elem = rec["anchor_symbols"][0]
        smiles = rec.get("smiles") or ""
        anchor_dft_indices = rec.get("anchor_indices_dft") or []
        anchor_dft_idx = anchor_dft_indices[0] if anchor_dft_indices else None
        fg, status = detect_anchor_functional_group(
            smiles=smiles,
            anchor_symbol=elem,
            dft_numbers=rec.get("dft_numbers"),
            dft_coords=rec.get("dft_coords"),
            dft_anchor_idx=anchor_dft_idx,
            funct_patterns=funct_patterns,
        )
        status_counts[(elem, status)] += 1
        if fg in {"missing", "no_rdkit", "no_patterns"}:
            continue
        if _exclude_from_single_anchor_violin_fg(fg):
            continue
        by_element[elem][fg].append(rec["energy"])

    if n_considered:
        print(
            "  Anchor functional-group detection on {} single-anchor molecules:".format(
                n_considered
            )
        )
        for (elem, status), n in sorted(status_counts.items()):
            print("    {} / {}: {}".format(elem, status, n))

    palette = VIOLIN_FILL_BY_ELEMENT
    saved: List[str] = []
    for elem in ANCHOR_ORDER:
        groups = by_element[elem]
        items = [
            (fg, vals, len(vals))
            for fg, vals in groups.items()
            if len(vals) >= min_group_size
            and not _exclude_from_single_anchor_violin_fg(fg)
        ]
        items.sort(key=lambda x: (float(np.median(x[1])), x[0]))
        if not items:
            print(
                "Single-anchor {}: no functional groups met --min_group_size={}.".format(
                    elem, min_group_size
                )
            )
            continue

        fig_h = 7.0 if publication else 6.0
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(items)), fig_h))
        draw_violin(
            ax,
            items,
            title="Single-{} anchor: binding energy by functional group (funct_group.csv SMARTS)".format(
                elem
            ),
            xlabel="Functional group (SMARTS match at anchor)",
            color=palette.get(elem, VIOLIN_FILL_DEFAULT),
            publication=publication,
        )
        fig.tight_layout()
        out_png = os.path.join(
            output_dir, "violin_single_anchor_{}_by_functional_group.png".format(elem)
        )
        for p in save_violin_figure(fig, out_png, publication):
            print("  Saved {}".format(p))
        plt.close(fig)
        write_group_stats_csv(
            os.path.join(output_dir, "stats_single_anchor_{}.csv".format(elem)),
            items,
            group_label="functional_group",
        )
        saved.append(out_png)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Violin plots: binding energy vs anchoring atoms / functional groups."
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        help="Merged downstream CSV path. Defaults to config.downstream_csv if available.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SCRIPT_DIR, "logs", "binding_anchor_stats"),
        help="Directory to save plots and per-group CSV stats.",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=3,
        help="Drop categories with fewer than this many molecules (default: 3).",
    )
    parser.add_argument(
        "--energy_min",
        type=float,
        default=None,
        help="Optional lower bound on adsorption energy (eV).",
    )
    parser.add_argument(
        "--energy_max",
        type=float,
        default=None,
        help="Optional upper bound on adsorption energy (eV).",
    )
    parser.add_argument(
        "--funct_group_csv",
        default=DEFAULT_FUNCT_GROUP_CSV,
        help=(
            "Tab- or comma-delimited table of functional groups with SMARTS "
            "(default: dataset/prediction/funct_group.csv). Used for plots 2-5."
        ),
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help=(
            "Publication-oriented styling: rc_context fonts/spines, despine top/right, "
            "PNG at 300 dpi plus matching PDF, subtle grid."
        ),
    )
    parser.add_argument(
        "--include_extra",
        action="store_true",
        help=(
            "Append config.downstream_extra_csv (best_adsorption_energy; "
            "E < -1.3 eV or E in [-0.6, 0] eV; anchors from SMILES N/O/S/P)."
        ),
    )
    parser.add_argument(
        "--extra_csv",
        default=None,
        help="Override extra CSV path (used with --include_extra).",
    )
    args = parser.parse_args()

    if not args.input:
        raise SystemExit(
            "No --input provided and config.downstream_csv is empty. Pass --input <csv>."
        )
    if not os.path.isfile(args.input):
        raise SystemExit("Input CSV not found: {}".format(args.input))

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading {}".format(args.input))
    rows, fieldnames = load_rows(args.input)
    print("  {} row(s); columns: {}".format(len(rows), fieldnames))

    records, skipped = collect_records(rows, fieldnames)
    if skipped:
        print("  [primary] Skipped (per reason): {}".format(skipped))

    if args.include_extra:
        extra_path = (args.extra_csv or _DEFAULT_EXTRA_CSV or "").strip()
        if not extra_path:
            print(
                "Warning: --include_extra set but config.downstream_extra_csv is empty; "
                "primary dataset only."
            )
        elif not os.path.isfile(extra_path):
            print("Warning: extra CSV not found: {}; primary dataset only.".format(extra_path))
        else:
            print("Loading extra: {}".format(extra_path))
            try:
                extra_records, extra_skipped = collect_records_from_extra_csv(extra_path)
            except (RuntimeError, ValueError) as exc:
                raise SystemExit("Failed to load extra CSV: {}".format(exc)) from exc
            if extra_skipped:
                print("  [extra] Skipped (per reason): {}".format(extra_skipped))
            print("  [extra] {} molecule(s) appended (SMILES-inferred N/O/S/P anchors)".format(
                len(extra_records)
            ))
            records = records + extra_records

    if args.energy_min is not None or args.energy_max is not None:
        before = len(records)
        lo = args.energy_min if args.energy_min is not None else -np.inf
        hi = args.energy_max if args.energy_max is not None else np.inf
        records = [r for r in records if lo <= r["energy"] <= hi]
        print("  Energy filter [{}, {}] eV: {} -> {}".format(lo, hi, before, len(records)))

    if not records:
        raise SystemExit("No usable molecules after parsing/filtering.")

    n_with_anchor_in_set = sum(1 for r in records if r["n_anchors_in_set"] > 0)
    n_single_nosp = sum(1 for r in records if r["n_anchors_in_set"] == 1)
    print(
        "  Usable molecules: {} total, {} have at least one N/O/S/P anchor, "
        "{} have exactly one N/O/S/P anchor (plots 2-5 pool; other anchors like H/Cl allowed)".format(
            len(records), n_with_anchor_in_set, n_single_nosp
        )
    )

    pub = bool(getattr(args, "publication", False))
    rc_extra = publication_rcparams() if pub else {}
    with plt.rc_context(rc_extra):
        plot_anchor_combinations(
            records, args.output_dir, args.min_group_size, publication=pub
        )

    fg_csv = (args.funct_group_csv or "").strip() or DEFAULT_FUNCT_GROUP_CSV
    funct_patterns, n_fg_ok, n_fg_skip = load_funct_group_patterns(fg_csv)
    print(
        "  funct_group SMARTS: {} pattern(s) from {} ({} skipped / invalid)".format(
            n_fg_ok, fg_csv, n_fg_skip
        )
    )

    with plt.rc_context(rc_extra):
        single_pngs = plot_single_anchor_by_functional(
            records,
            args.output_dir,
            args.min_group_size,
            funct_patterns,
            publication=pub,
        )

    print("Done. Outputs in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
