"""
Violin plot of test-set prediction error vs. binned DFT adsorption energy.

Reads ``test_predictions.csv`` from downstream training (columns:
``target_binding_energy``, ``predicted_binding_energy``, ``error``).
X-axis: bins of ``target_binding_energy`` (same as ``adsorption_energy`` in the merged CSV).
Y-axis: absolute error = |column 2 − column 1| (eV), computed from the first two
CSV columns (not the ``error`` column).

Usage:
    python plot_test_error_violin.py
    python plot_test_error_violin.py --input /path/to/test_predictions.csv
    python plot_test_error_violin.py --publication
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_DEFAULT_INPUT = (
    "/kfs3/scratch/yeming/ai4m/prediction/logs/downstream_notag_03222026/"
    "predictions/test_predictions.csv"
)

DEFAULT_BIN_MIN = -3.0
DEFAULT_BIN_MAX = 0.0
DEFAULT_BIN_WIDTH = 0.1
DEFAULT_MIN_GROUP_SIZE = 3

VIOLIN_FILL = "#6B9AA8"
VIOLIN_EDGE = "#2A3238"
VIOLIN_REF_LINE = "#A8B0B8"

def load_test_predictions(csv_path: str) -> Tuple[List[float], List[float], str, str]:
    """Return (col1 values, |col2−col1|, col1 name, col2 name) per row."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lstrip("\ufeff") for h in (reader.fieldnames or [])]
        if len(fieldnames) < 2:
            raise ValueError("CSV needs at least two columns; found: {}".format(fieldnames))

        col_first = fieldnames[0]
        col_second = fieldnames[1]

        col1_vals: List[float] = []
        abs_errors: List[float] = []
        skipped = 0
        for row in reader:
            try:
                v1 = float((row.get(col_first) or "").strip())
                v2 = float((row.get(col_second) or "").strip())
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not np.isfinite(v1) or not np.isfinite(v2):
                skipped += 1
                continue
            col1_vals.append(v1)
            abs_errors.append(abs(v2 - v1))

    if skipped:
        print("  Skipped {} row(s) with missing or invalid values.".format(skipped))
    if not col1_vals:
        raise ValueError("No valid rows in {}".format(csv_path))
    return col1_vals, abs_errors, col_first, col_second


def make_bin_edges(bin_min: float, bin_max: float, bin_width: float) -> np.ndarray:
    n = int(round((bin_max - bin_min) / bin_width))
    edges = bin_min + np.arange(n + 1, dtype=np.float64) * bin_width
    edges[-1] = bin_max
    return edges


def bin_label(lo: float, hi: float, is_last: bool) -> str:
    if is_last:
        return "[{:.1f}, {:.1f}]".format(lo, hi)
    return "[{:.1f}, {:.1f})".format(lo, hi)


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


def plot_error_violin(
    targets: List[float],
    abs_errors: List[float],
    output_dir: str,
    *,
    col_first_name: str = "column 1",
    col_second_name: str = "column 2",
    bin_min: float = DEFAULT_BIN_MIN,
    bin_max: float = DEFAULT_BIN_MAX,
    bin_width: float = DEFAULT_BIN_WIDTH,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    publication: bool = False,
) -> str:
    edges = make_bin_edges(bin_min, bin_max, bin_width)
    n_bins = len(edges) - 1
    grouped: Dict[int, List[float]] = defaultdict(list)
    n_outside = 0

    for tgt, abs_err in zip(targets, abs_errors):
        if tgt < bin_min or tgt > bin_max:
            n_outside += 1
            continue
        idx = min(int((tgt - bin_min) / bin_width), n_bins - 1)
        grouped[idx].append(abs_err)

    if n_outside:
        print(
            "  {} sample(s) with target_binding_energy outside [{:.1f}, {:.1f}] eV.".format(
                n_outside, bin_min, bin_max
            )
        )

    items: List[Tuple[str, List[float], int]] = []
    for i in range(n_bins):
        vals = grouped.get(i, [])
        if len(vals) < min_group_size:
            continue
        lo, hi = float(edges[i]), float(edges[i + 1])
        items.append((bin_label(lo, hi, i == n_bins - 1), vals, len(vals)))

    if not items:
        raise SystemExit(
            "No bins with >= {} samples in [{:.1f}, {:.1f}] eV (width {:.1f}).".format(
                min_group_size, bin_min, bin_max, bin_width
            )
        )

    os.makedirs(output_dir, exist_ok=True)
    data = [vals for _, vals, _ in items]
    labels = ["{}\n(n={})".format(name, n) for name, _, n in items]
    positions = list(range(1, len(items) + 1))

    fig, ax = plt.subplots(
        figsize=(max(9.0, 1.1 * len(items)), 6.5 if publication else 6.0),
        facecolor="white",
    )
    parts = ax.violinplot(
        data,
        positions=positions,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    _set_violin_style(parts, VIOLIN_FILL)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11)
    ax.set_ylabel(
        "Absolute error |{} − {}| (eV)".format(col_second_name, col_first_name),
        fontsize=12,
    )
    ax.set_xlabel("{} (binning on x)".format(col_first_name), fontsize=12)
    ax.set_title(
        "Test-set absolute prediction error vs. {} (bin width = {:.1f} eV)".format(
            col_first_name, bin_width
        ),
        fontsize=13,
    )
    ax.axhline(0.0, color=VIOLIN_REF_LINE, linewidth=0.55, linestyle="--", alpha=0.45, zorder=0)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", direction="in", labelsize=10)
    if publication:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_in = sum(n for _, _, n in items)
    ax.text(
        0.98,
        0.97,
        "N = {}\nBins: [{:.1f}, {:.1f}], Δ = {:.1f} eV".format(
            n_in, bin_min, bin_max, bin_width
        ),
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    fig.tight_layout()
    out_png = os.path.join(output_dir, "test_error_vs_adsorption_energy_violin.png")
    dpi = 300 if publication else 200
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    if publication:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        print("  Saved {}".format(out_pdf))
    plt.close(fig)
    print("  Saved {}".format(out_png))

    stats_csv = os.path.join(output_dir, "test_error_vs_adsorption_energy_violin_stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "adsorption_energy_bin",
                "n",
                "mean_abs_error_eV",
                "median_abs_error_eV",
                "std_abs_error_eV",
                "min_abs_error_eV",
                "max_abs_error_eV",
            ]
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
        description="Violin plot: |col2−col1| vs. binned column-1 (adsorption energy)."
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        help="test_predictions.csv from train_downstream (default: log_dir/predictions/).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same folder as --input).",
    )
    parser.add_argument("--bin_min", type=float, default=DEFAULT_BIN_MIN)
    parser.add_argument("--bin_max", type=float, default=DEFAULT_BIN_MAX)
    parser.add_argument("--bin_width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help="Minimum test molecules per bin to draw a violin (default: 3).",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Despine and save 300 dpi PNG plus PDF.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit("Input CSV not found: {}".format(args.input))
    if args.bin_width <= 0:
        raise SystemExit("--bin_width must be positive.")
    if args.bin_max <= args.bin_min:
        raise SystemExit("--bin_max must be greater than --bin_min.")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    print("Loading {}".format(args.input))
    col1, abs_errors, name1, name2 = load_test_predictions(args.input)
    print(
        "  {} test sample(s); y = |{} − {}|; mean abs error = {:.4f} eV".format(
            len(col1), name2, name1, float(np.mean(abs_errors))
        )
    )

    plot_error_violin(
        col1,
        abs_errors,
        output_dir,
        col_first_name=name1,
        col_second_name=name2,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        bin_width=args.bin_width,
        min_group_size=args.min_group_size,
        publication=bool(args.publication),
    )
    print("Done.")


if __name__ == "__main__":
    main()
