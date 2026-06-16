"""
Bar-chart distribution of adsorption / binding energy from the merged downstream CSV.

Uses the primary dataset (``config.downstream_csv`` or ``--input``). The bar chart compares
initial-only vs. initial+extended (``config.downstream_extra_csv``, ``best_adsorption_energy``,
filtered like train_downstream) side by side in each energy bin. Bins span [-3, 0] eV with
width 0.1, with Gaussian fits per series.

Also writes a violin plot of binding energy vs. heavy-atom count bins [0, 30] step 5
(energies in [-3, 0] eV; initial dataset only).

Usage:
    python plot_binding_energy_histogram.py
    python plot_binding_energy_histogram.py --no_extra
    python plot_binding_energy_histogram.py --input <merged.csv> --output_dir logs/binding_energy_hist
    python plot_binding_energy_histogram.py --publication
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
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

try:
    from scipy.optimize import curve_fit  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    curve_fit = None  # type: ignore
    _SCIPY_AVAILABLE = False

try:
    from rdkit import Chem  # type: ignore

    _RDKIT_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    _RDKIT_AVAILABLE = False

ENERGY_COL_CANDIDATES = [
    "adsorption_energy",
    "binding_energy",
    "energy",
]
EXTRA_ENERGY_COL_CANDIDATES = [
    "best_adsorption_energy",
    "adsorption_energy",
    "binding_energy",
    "energy",
]
SMILES_COL_CANDIDATES = ["SMILES", "smiles", "canonical_smiles"]
# Extra CSV: match train_downstream.load_extra_smiles_csv energy windows
EXTRA_STRONG_THRESHOLD_EV = -1.3
EXTRA_WEAK_LOW_EV = -0.6
EXTRA_WEAK_HIGH_EV = 0.0


def extra_energy_in_training_ranges(energy: float) -> bool:
    """Strong binders (< -1.3 eV) or weak binders ([-0.6, 0] eV)."""
    return energy < EXTRA_STRONG_THRESHOLD_EV or (
        EXTRA_WEAK_LOW_EV <= energy <= EXTRA_WEAK_HIGH_EV
    )

DEFAULT_BIN_MIN = -3.0
DEFAULT_BIN_MAX = 0.0
DEFAULT_BIN_WIDTH = 0.1

SIZE_BIN_MIN = 0
SIZE_BIN_MAX = 30
SIZE_BIN_WIDTH = 5
SIZE_BIN_LABELS = [
    "[0, 5)",
    "[5, 10)",
    "[10, 15)",
    "[15, 20)",
    "[20, 25)",
    "[25, 30]",
]

BAR_FILL = "#6B9AA8"
BAR_FILL_COMBINED = "#B87A6E"
BAR_EDGE = "#2A3238"
FIT_LINE = "#3D5A80"
FIT_LINE_COMBINED = "#B87A6E"
LABEL_INITIAL = "Initial"
LABEL_COMBINED = "Initial + extended"
VIOLIN_FILL = "#6B9AA8"
VIOLIN_EDGE = "#2A3238"
VIOLIN_REF_LINE = "#A8B0B8"


class MolRecord(NamedTuple):
    energy: float
    n_heavy: Optional[int]


def resolve_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lookup = {h.lower(): h for h in fieldnames}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def heavy_atom_count_from_smiles(smiles: str) -> Optional[int]:
    """Heavy-atom count after RemoveHs (same notion as train_downstream graphs)."""
    if not _RDKIT_AVAILABLE or Chem is None:
        return None
    s = (smiles or "").strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return int(Chem.RemoveHs(mol).GetNumAtoms())


def size_bin_label(n_heavy: int) -> Optional[str]:
    """Map heavy-atom count to a [0, 30] bin of width 5."""
    if n_heavy < SIZE_BIN_MIN or n_heavy > SIZE_BIN_MAX:
        return None
    idx = min(n_heavy // SIZE_BIN_WIDTH, len(SIZE_BIN_LABELS) - 1)
    return SIZE_BIN_LABELS[idx]


def load_mol_records(
    csv_path: str,
    *,
    energy_col_candidates: Optional[List[str]] = None,
    source_label: str = "",
    energy_filter: Optional[Callable[[float], bool]] = None,
) -> Tuple[List[MolRecord], str]:
    """Parse binding energy and heavy-atom count (from SMILES) per row."""
    candidates = energy_col_candidates or ENERGY_COL_CANDIDATES
    prefix = "  [{}] ".format(source_label) if source_label else "  "
    filtered_range = 0
    skipped = 0
    skipped_smiles = 0
    records: List[MolRecord] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lstrip("\ufeff") for h in (reader.fieldnames or [])]
        col_energy = resolve_column(fieldnames, candidates)
        if col_energy is None:
            raise ValueError(
                "No energy column found (tried {}). Columns: {}".format(
                    ", ".join(candidates), fieldnames
                )
            )
        col_smiles = resolve_column(fieldnames, SMILES_COL_CANDIDATES)

        for row in reader:
            raw = (row.get(col_energy) or "").strip()
            if not raw or raw.upper() == "N/A":
                skipped += 1
                continue
            try:
                e = float(raw)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not np.isfinite(e):
                skipped += 1
                continue
            if energy_filter is not None and not energy_filter(e):
                filtered_range += 1
                continue

            n_heavy: Optional[int] = None
            if col_smiles is not None:
                smiles = (row.get(col_smiles) or "").strip()
                n_heavy = heavy_atom_count_from_smiles(smiles)
                if n_heavy is None and smiles:
                    skipped_smiles += 1

            records.append(MolRecord(energy=e, n_heavy=n_heavy))

    if skipped:
        print("{}Skipped {} row(s) with missing or invalid energy.".format(prefix, skipped))
    if filtered_range:
        print(
            "{}Filtered {} row(s) outside extra energy windows "
            "(< {:.1f} eV or [{:.1f}, {:.1f}] eV).".format(
                prefix,
                filtered_range,
                EXTRA_STRONG_THRESHOLD_EV,
                EXTRA_WEAK_LOW_EV,
                EXTRA_WEAK_HIGH_EV,
            )
        )
    if skipped_smiles:
        print("{}Could not parse SMILES for {} row(s) (no size for violin).".format(prefix, skipped_smiles))
    if col_smiles is None:
        print("{}No SMILES column; energies loaded but size violins will be empty.".format(prefix))
    elif not _RDKIT_AVAILABLE:
        print("{}RDKit unavailable; energies loaded but size violins will be empty.".format(prefix))

    return records, col_energy


def make_bin_edges(
    bin_min: float, bin_max: float, bin_width: float
) -> np.ndarray:
    n = int(round((bin_max - bin_min) / bin_width))
    edges = bin_min + np.arange(n + 1, dtype=np.float64) * bin_width
    edges[-1] = bin_max
    return edges


def histogram_in_range(
    energies: np.ndarray,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Count values in [edges[0], edges[-1]]; return counts, bin centers, n_outside."""
    lo, hi = float(edges[0]), float(edges[-1])
    in_range = energies[(energies >= lo) & (energies <= hi)]
    n_outside = int(energies.size - in_range.size)
    counts, _ = np.histogram(in_range, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(np.int64), centers, n_outside


def _gaussian_counts(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian evaluated at bin centers; ``amp`` scales peak height."""
    if sigma <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_bins(
    bin_centers: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Least-squares Gaussian fit to (center, count) histogram points."""
    counts_f = counts.astype(np.float64)
    if counts_f.sum() < 1:
        return np.zeros_like(bin_centers, dtype=np.float64), {}

    mu0 = float(np.average(bin_centers, weights=counts_f + 1e-12))
    var0 = float(np.average((bin_centers - mu0) ** 2, weights=counts_f + 1e-12))
    sigma0 = max(np.sqrt(var0), bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1)
    amp0 = float(counts_f.max()) if counts_f.max() > 0 else 1.0

    if _SCIPY_AVAILABLE and curve_fit is not None:
        try:
            popt, _ = curve_fit(
                _gaussian_counts,
                bin_centers,
                counts_f,
                p0=[amp0, mu0, sigma0],
                bounds=(
                    [0.0, float(bin_centers.min()), 1e-6],
                    [np.inf, float(bin_centers.max()), np.inf],
                ),
                maxfev=20000,
            )
            amp, mu, sigma = popt
            fit_y = _gaussian_counts(bin_centers, amp, mu, sigma)
            return fit_y, {"amp": float(amp), "mu": float(mu), "sigma": float(sigma)}
        except Exception as exc:
            print("  Gaussian fit failed ({}); using moment-based curve.".format(exc))

    mu = mu0
    sigma = sigma0
    amp = amp0
    fit_y = _gaussian_counts(bin_centers, amp, mu, sigma)
    return fit_y, {"amp": amp, "mu": mu, "sigma": sigma}


def plot_histogram_comparison(
    initial_energies: List[float],
    combined_energies: Optional[List[float]],
    output_dir: str,
    *,
    bin_min: float = DEFAULT_BIN_MIN,
    bin_max: float = DEFAULT_BIN_MAX,
    bin_width: float = DEFAULT_BIN_WIDTH,
    publication: bool = False,
) -> str:
    """Grouped bar chart: initial-only vs. initial+extended (same energy bins)."""
    os.makedirs(output_dir, exist_ok=True)
    edges = make_bin_edges(bin_min, bin_max, bin_width)
    width = float(edges[1] - edges[0])

    arr_initial = np.asarray(initial_energies, dtype=np.float64)
    counts_initial, centers, n_out_initial = histogram_in_range(arr_initial, edges)
    n_in_initial = int(counts_initial.sum())
    print(
        "  [{}] in [{:.1f}, {:.1f}] eV: {} / {} ({} outside)".format(
            LABEL_INITIAL,
            bin_min,
            bin_max,
            n_in_initial,
            arr_initial.size,
            n_out_initial,
        )
    )

    has_combined = combined_energies is not None and len(combined_energies) > 0
    if has_combined:
        arr_combined = np.asarray(combined_energies, dtype=np.float64)
        counts_combined, _, n_out_combined = histogram_in_range(arr_combined, edges)
        n_in_combined = int(counts_combined.sum())
        print(
            "  [{}] in [{:.1f}, {:.1f}] eV: {} / {} ({} outside)".format(
                LABEL_COMBINED,
                bin_min,
                bin_max,
                n_in_combined,
                arr_combined.size,
                n_out_combined,
            )
        )
    else:
        counts_combined = None
        n_in_combined = 0

    fit_initial, params_initial = fit_gaussian_to_bins(centers, counts_initial)
    fit_combined = np.zeros_like(fit_initial)
    params_combined: dict = {}
    if has_combined and counts_combined is not None:
        fit_combined, params_combined = fit_gaussian_to_bins(centers, counts_combined)

    x_curve = np.linspace(bin_min, bin_max, 400)
    curve_initial = (
        _gaussian_counts(
            x_curve, params_initial["amp"], params_initial["mu"], params_initial["sigma"]
        )
        if params_initial
        else np.zeros_like(x_curve)
    )
    curve_combined = (
        _gaussian_counts(
            x_curve,
            params_combined["amp"],
            params_combined["mu"],
            params_combined["sigma"],
        )
        if params_combined
        else np.zeros_like(x_curve)
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.0), facecolor="white")
    bar_w = width * (0.38 if has_combined else 0.92)
    offset = width * 0.22 if has_combined else 0.0

    ax.bar(
        centers - offset,
        counts_initial,
        width=bar_w,
        align="center",
        color=BAR_FILL,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        alpha=0.9,
        label=LABEL_INITIAL,
        zorder=2,
    )
    if has_combined and counts_combined is not None:
        ax.bar(
            centers + offset,
            counts_combined,
            width=bar_w,
            align="center",
            color=BAR_FILL_COMBINED,
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            alpha=0.9,
            label=LABEL_COMBINED,
            zorder=2,
        )

    ax.plot(
        x_curve,
        curve_initial,
        color=FIT_LINE,
        linewidth=2.0,
        label="Gaussian fit ({})".format(LABEL_INITIAL),
        zorder=3,
    )
    if has_combined and params_combined:
        ax.plot(
            x_curve,
            curve_combined,
            color=FIT_LINE_COMBINED,
            linewidth=2.0,
            label="Gaussian fit ({})".format(LABEL_COMBINED),
            zorder=3,
        )

    ax.set_xlim(bin_min - 0.05 * width, bin_max + 0.05 * width)
    ax.set_xlabel("Binding energy (eV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    title = "Binding energy distribution (bin width = {:.1f} eV)".format(bin_width)
    if has_combined:
        title += "\ninitial vs. initial + extended"
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", direction="in", labelsize=10)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.55, zorder=0)

    stat = "{}: N = {}\nBins: [{:.1f}, {:.1f}], Δ = {:.1f} eV".format(
        LABEL_INITIAL, n_in_initial, bin_min, bin_max, bin_width
    )
    if params_initial:
        stat += "\nμ = {:.3f} eV, σ = {:.3f} eV".format(
            params_initial["mu"], params_initial["sigma"]
        )
    if has_combined:
        stat += "\n\n{}: N = {}".format(LABEL_COMBINED, n_in_combined)
        if params_combined:
            stat += "\nμ = {:.3f} eV, σ = {:.3f} eV".format(
                params_combined["mu"], params_combined["sigma"]
            )
    ax.text(
        0.98,
        0.97,
        stat,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    if publication:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png = os.path.join(output_dir, "binding_energy_histogram.png")
    dpi = 300 if publication else 200
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    if publication:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        print("  Saved {}".format(out_pdf))
    plt.close(fig)
    print("  Saved {}".format(out_png))

    bins_csv = os.path.join(output_dir, "binding_energy_histogram_bins.csv")
    with open(bins_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "bin_low_eV",
            "bin_high_eV",
            "bin_center_eV",
            "count_initial",
            "fit_initial",
        ]
        if has_combined:
            header.extend(["count_combined", "fit_combined"])
        writer.writerow(header)
        for i, c in enumerate(centers):
            row = [
                f"{edges[i]:.4f}",
                f"{edges[i + 1]:.4f}",
                f"{c:.4f}",
                int(counts_initial[i]),
                f"{fit_initial[i]:.4f}",
            ]
            if has_combined and counts_combined is not None:
                row.extend([int(counts_combined[i]), f"{fit_combined[i]:.4f}"])
            writer.writerow(row)
    print("  Saved {}".format(bins_csv))
    return out_png


def _set_violin_style(parts, face_color: str, edge_color: str = VIOLIN_EDGE) -> None:
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


def plot_energy_vs_size_violin(
    records: List[MolRecord],
    output_dir: str,
    *,
    energy_min: float = DEFAULT_BIN_MIN,
    energy_max: float = DEFAULT_BIN_MAX,
    publication: bool = False,
    min_group_size: int = 1,
) -> Optional[str]:
    """Violin plot: binding energy (y) vs. heavy-atom size bins [0, 30] step 5 (x).

    Only molecules with energy in [energy_min, energy_max] eV are included.
    """
    if not _RDKIT_AVAILABLE:
        print("  Size violin skipped: RDKit not available.")
        return None

    grouped: Dict[str, List[float]] = defaultdict(list)
    n_no_size = 0
    n_outside_size = 0
    n_outside_energy = 0
    for rec in records:
        if rec.energy < energy_min or rec.energy > energy_max:
            n_outside_energy += 1
            continue
        if rec.n_heavy is None:
            n_no_size += 1
            continue
        label = size_bin_label(rec.n_heavy)
        if label is None:
            n_outside_size += 1
            continue
        grouped[label].append(rec.energy)

    items: List[Tuple[str, List[float], int]] = []
    for label in SIZE_BIN_LABELS:
        vals = grouped.get(label, [])
        n = len(vals)
        if n >= min_group_size:
            items.append((label, vals, n))

    if n_no_size or n_outside_size or n_outside_energy:
        print(
            "  Size violin: {} outside [{:.1f}, {:.1f}] eV, {} without size, "
            "{} outside [{}, {}] heavy atoms.".format(
                n_outside_energy,
                energy_min,
                energy_max,
                n_no_size,
                n_outside_size,
                SIZE_BIN_MIN,
                SIZE_BIN_MAX,
            )
        )

    if not items:
        print("  Size violin skipped: no molecules with size in [{}, {}].".format(
            SIZE_BIN_MIN, SIZE_BIN_MAX
        ))
        return None

    os.makedirs(output_dir, exist_ok=True)
    data = [vals for _, vals, _ in items]
    labels = ["{}\n(n={})".format(name, n) for name, _, n in items]
    positions = list(range(1, len(items) + 1))

    fig_h = 6.5 if publication else 6.0
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(items)), fig_h), facecolor="white")
    parts = ax.violinplot(
        data,
        positions=positions,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    _set_violin_style(parts, VIOLIN_FILL)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Binding energy (eV)", fontsize=12)
    ax.set_xlabel("Heavy-atom count (SMILES, no H)", fontsize=12)
    ax.set_title(
        "Binding energy vs. molecule size "
        "({:.1f}–{:.1f} eV; size bins width = {} atoms)".format(
            energy_min, energy_max, SIZE_BIN_WIDTH
        ),
        fontsize=13,
    )
    ax.set_ylim(energy_min, energy_max)
    ax.tick_params(axis="both", direction="in", labelsize=10)
    ax.axhline(
        0.0,
        color=VIOLIN_REF_LINE,
        linewidth=0.55,
        linestyle="--",
        alpha=0.45,
        zorder=0,
    )
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.55, zorder=0)
    if publication:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png = os.path.join(output_dir, "violin_binding_energy_vs_size.png")
    dpi = 300 if publication else 200
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    if publication:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        print("  Saved {}".format(out_pdf))
    plt.close(fig)
    print("  Saved {}".format(out_png))

    stats_csv = os.path.join(output_dir, "violin_binding_energy_vs_size_stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["size_bin", "n", "mean_eV", "median_eV", "std_eV", "min_eV", "max_eV"]
        )
        for name, vals, n in items:
            arr = np.asarray(vals, dtype=np.float64)
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
    print("  Saved {}".format(stats_csv))
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binding-energy histogram ([-3, 0] eV) and size-binned violin "
            "(heavy atoms [0, 30], step 5)."
        )
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        help="Merged downstream CSV (default: config.downstream_csv).",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SCRIPT_DIR, "logs", "binding_energy_histogram"),
        help="Directory for PNG/PDF and per-bin CSV.",
    )
    parser.add_argument(
        "--bin_min",
        type=float,
        default=DEFAULT_BIN_MIN,
        help="Histogram lower edge (eV, default: -3).",
    )
    parser.add_argument(
        "--bin_max",
        type=float,
        default=DEFAULT_BIN_MAX,
        help="Histogram upper edge (eV, default: 0).",
    )
    parser.add_argument(
        "--bin_width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help="Bin width in eV (default: 0.1).",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Despine and save 300 dpi PNG plus PDF.",
    )
    parser.add_argument(
        "--no_extra",
        action="store_true",
        help=(
            "Do not load config.downstream_extra_csv; bar chart shows initial dataset only."
        ),
    )
    parser.add_argument(
        "--extra_csv",
        default=None,
        help="Path to extended CSV (default: config.downstream_extra_csv).",
    )
    # Backward compatibility
    parser.add_argument(
        "--include_extra",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not args.input:
        raise SystemExit(
            "No --input provided and config.downstream_csv is empty. Pass --input <csv>."
        )
    if not os.path.isfile(args.input):
        raise SystemExit("Input CSV not found: {}".format(args.input))
    if args.bin_width <= 0:
        raise SystemExit("--bin_width must be positive.")
    if args.bin_max <= args.bin_min:
        raise SystemExit("--bin_max must be greater than --bin_min.")

    print("Loading primary: {}".format(args.input))
    primary_records, col_energy = load_mol_records(args.input, source_label="primary")
    if not primary_records:
        raise SystemExit("No valid binding energies found in primary CSV.")

    initial_energies = [r.energy for r in primary_records]
    arr = np.asarray(initial_energies, dtype=np.float64)
    print(
        "  [primary] {} molecule(s); column {!r}; mean={:.3f} eV, median={:.3f} eV".format(
            len(primary_records), col_energy, arr.mean(), np.median(arr)
        )
    )

    combined_energies: Optional[List[float]] = None
    load_extra = not args.no_extra
    extra_path = (args.extra_csv or _DEFAULT_EXTRA_CSV or "").strip()
    if load_extra and extra_path:
        if not os.path.isfile(extra_path):
            print("Warning: extra CSV not found: {}; comparison uses initial only.".format(
                extra_path
            ))
        else:
            print("Loading extra: {}".format(extra_path))
            extra_records, extra_col = load_mol_records(
                extra_path,
                energy_col_candidates=EXTRA_ENERGY_COL_CANDIDATES,
                source_label="extra",
                energy_filter=extra_energy_in_training_ranges,
            )
            if not extra_records:
                print("Warning: no valid energies in extra CSV; comparison uses initial only.")
            else:
                extra_arr = np.asarray([r.energy for r in extra_records], dtype=np.float64)
                print(
                    "  [extra] {} molecule(s); column {!r}; mean={:.3f} eV, median={:.3f} eV".format(
                        len(extra_records),
                        extra_col,
                        extra_arr.mean(),
                        np.median(extra_arr),
                    )
                )
                combined_energies = initial_energies + [r.energy for r in extra_records]
                comb_arr = np.asarray(combined_energies, dtype=np.float64)
                print(
                    "  [combined] {} molecule(s); mean={:.3f} eV, median={:.3f} eV".format(
                        len(combined_energies), comb_arr.mean(), np.median(comb_arr)
                    )
                )
    elif load_extra and not extra_path:
        print("Warning: downstream_extra_csv not set; comparison uses initial only.")

    plot_histogram_comparison(
        initial_energies,
        combined_energies,
        args.output_dir,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        bin_width=args.bin_width,
        publication=bool(args.publication),
    )
    print(
        "  Size violin: {} molecule(s) from initial dataset only.".format(
            len(primary_records)
        )
    )
    plot_energy_vs_size_violin(
        primary_records,
        args.output_dir,
        energy_min=args.bin_min,
        energy_max=args.bin_max,
        publication=bool(args.publication),
    )
    print("Done. Outputs in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
