"""
Extract validation losses from saved checkpoint files and plot loss vs. epoch.

Example:
    python comparison_2feat/plot_checkpoint_val_loss.py

    python comparison_2feat/plot_checkpoint_val_loss.py \
        --checkpoint-dir /kfs3/scratch/yeming/ai4m/prediction/comparison_2feat/checkpoints \
        --recursive \
        --include-best
"""
import argparse
import csv
import math
import re
import sys

if sys.version_info < (3, 6):
    sys.exit("Please run this script with Python 3.6+ from your PyTorch training environment.")

from pathlib import Path

import torch


DEFAULT_CHECKPOINT_DIR = Path(
    "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/downstream_notag_03222026/downstream"
)
EXCLUDED_CHECKPOINT_FILENAMES = {
    "downstream_best_model.pt",
    "gin_e_finetuned.pt",
}


def _to_float(value):
    """Convert scalar checkpoint values to a Python float."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value):
    """Convert scalar checkpoint epoch values to an int."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _epoch_from_name(path):
    match = re.search(r"(?:^|[_-])epoch[_-]?(\d+)(?:\D|$)", path.stem)
    return int(match.group(1)) if match else None


def _loss_from_checkpoint(checkpoint):
    for key in ("val_loss", "validation_loss", "valid_loss", "loss"):
        loss = _to_float(checkpoint.get(key))
        if loss is not None:
            return loss, key
    return None, None


def find_checkpoint_files(
    checkpoint_dir,
    pattern,
    recursive,
    include_best,
):
    globber = checkpoint_dir.rglob if recursive else checkpoint_dir.glob
    paths = sorted(
        p
        for p in globber(pattern)
        if p.is_file() and p.name.lower() not in EXCLUDED_CHECKPOINT_FILENAMES
    )
    if include_best:
        return paths
    return [p for p in paths if "best" not in p.name.lower()]


def extract_losses(paths):
    rows = []
    skipped = []

    for path in paths:
        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:  # Keep scanning if one file is stale/corrupt.
            skipped.append((path, "load failed: {}".format(exc)))
            continue

        if not isinstance(checkpoint, dict):
            skipped.append((path, "checkpoint is not a dict"))
            continue

        epoch = _to_int(checkpoint.get("epoch")) or _epoch_from_name(path)
        val_loss, loss_key = _loss_from_checkpoint(checkpoint)
        train_loss = _to_float(checkpoint.get("train_loss"))

        if epoch is None:
            skipped.append((path, "missing epoch"))
            continue
        if val_loss is None:
            skipped.append((path, "missing validation loss"))
            continue

        rows.append(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "loss_key": loss_key,
                "checkpoint": str(path),
            }
        )

    rows.sort(key=lambda row: (row["epoch"], row["checkpoint"]))

    if skipped:
        print("Skipped {} checkpoint(s):".format(len(skipped)))
        for path, reason in skipped:
            print("  {}: {}".format(path, reason))

    return rows


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "val_loss", "train_loss", "loss_key", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(rows, output_png):
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    ax.plot(epochs, val_losses, marker="o", linewidth=1.8, markersize=4, label="Validation loss")

    train_rows = [row for row in rows if row["train_loss"] is not None]
    if train_rows:
        ax.plot(
            [row["epoch"] for row in train_rows],
            [row["train_loss"] for row in train_rows],
            marker="s",
            linewidth=1.3,
            markersize=3,
            alpha=0.75,
            label="Train loss",
        )

    best_idx = min(range(len(rows)), key=lambda idx: rows[idx]["val_loss"])
    best = rows[best_idx]
    ax.scatter([best["epoch"]], [best["val_loss"]], s=70, zorder=5, label="Best validation")
    ax.annotate(
        "best: epoch {}\nval={:.4g}".format(best["epoch"], best["val_loss"]),
        xy=(best["epoch"], best["val_loss"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_title("Validation Loss Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract epoch/loss values from PyTorch checkpoints and plot validation loss."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint files. Default: {}".format(DEFAULT_CHECKPOINT_DIR),
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob pattern for checkpoint files. Default: *.pt",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search checkpoint subdirectories too.",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        help="Include files with 'best' in the filename. Default excludes them to avoid duplicate epochs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Default: <checkpoint-dir>/validation_loss_by_epoch.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Plot output path. Default: <checkpoint-dir>/validation_loss_by_epoch.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("Checkpoint directory not found: {}".format(checkpoint_dir))

    output_csv = args.output_csv or checkpoint_dir / "validation_loss_by_epoch.csv"
    output_png = args.output_png or checkpoint_dir / "validation_loss_by_epoch.png"

    checkpoint_files = find_checkpoint_files(
        checkpoint_dir=checkpoint_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        include_best=args.include_best,
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            "No checkpoint files found in {} with pattern {}".format(checkpoint_dir, args.pattern)
        )

    rows = extract_losses(checkpoint_files)
    if not rows:
        raise RuntimeError("No epoch/loss pairs could be extracted from the checkpoint files.")

    write_csv(rows, output_csv)
    plot_losses(rows, output_png)

    best = min(rows, key=lambda row: row["val_loss"])
    print("Extracted {} checkpoint(s).".format(len(rows)))
    print("Best validation loss: {:.6g} at epoch {}".format(best["val_loss"], best["epoch"]))
    print("CSV saved to: {}".format(output_csv))
    print("Plot saved to: {}".format(output_png))


if __name__ == "__main__":
    main()
