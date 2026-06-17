#!/usr/bin/env python3
"""
Plot mean Pearson correlation per source from per-source ablation results.

Usage:
    python3 plot_per_source_results.py \
        --per_source_run_id 21 \
        [--combined_run_id 20] \
        [--results_dir results] \
        [--output per_source_pearson.png]

Looks for:
    results/<per_source_run_id>/src_*/src_<per_source_run_id>_*/eval_best.ckpt/k562_results.csv
    results/<combined_run_id>/qc_emb_*/qc_emb_*/eval_best.ckpt/k562_results.csv   (reference lines)
    results/<combined_run_id>/baseline_*/baseline_*/eval_best.ckpt/k562_results.csv (reference lines)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PEARSON_COLS = [
    "mean_pearson",
    "pearson",
    "pearson_mean",
    "mean pearson",
    "Pearson",
    "mean_pearson_delta",
]


def read_mean_pearson(csv_path: Path) -> float | None:
    """Return the mean Pearson value from a k562_results.csv, or None on failure."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  WARNING: could not read {csv_path}: {e}", file=sys.stderr)
        return None

    # Try known column names first
    for col in PEARSON_COLS:
        if col in df.columns:
            val = df[col].mean()
            return float(val)

    # Fallback: any column whose name contains "pearson" (case-insensitive)
    pearson_like = [c for c in df.columns if "pearson" in c.lower()]
    if pearson_like:
        col = pearson_like[0]
        print(f"  INFO: using column '{col}' from {csv_path.parent.parent.name}")
        return float(df[col].mean())

    print(
        f"  WARNING: no Pearson column found in {csv_path}\n"
        f"           Available columns: {list(df.columns)}",
        file=sys.stderr,
    )
    return None


def collect_per_source(results_dir: Path, run_id: str) -> dict[str, float]:
    """
    Scan results/<run_id>/src_*/src_<run_id>_*/eval_best.ckpt/k562_results.csv
    and return {source_name: mean_pearson}.
    """
    pattern = f"{run_id}/src_*/src_{run_id}_*/eval_best.ckpt/k562_results.csv"
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        # Also try without the eval_best.ckpt level (flat structure)
        pattern2 = f"{run_id}/src_*/src_{run_id}_*/k562_results.csv"
        csvs = sorted(results_dir.glob(pattern2))

    data = {}
    for csv in csvs:
        # Extract source name from path: src_<run_id>_<SRC_NAME>
        inner_dir = csv.parent if "eval_best.ckpt" not in csv.parent.name else csv.parent.parent
        dir_name = inner_dir.name  # e.g. "src_21_BioGRID"
        # Strip the "src_<run_id>_" prefix
        prefix = f"src_{run_id}_"
        source = dir_name[len(prefix):] if dir_name.startswith(prefix) else dir_name
        # Restore hyphens that were replaced with underscores in bash (ESM_2 → ESM-2)
        # Check if original name with hyphen exists in a typical set; otherwise keep underscore
        val = read_mean_pearson(csv)
        if val is not None:
            data[source] = val
            print(f"  {source:<25s}  mean_pearson = {val:.4f}")

    return data


def collect_reference(results_dir: Path, run_id: str, label_prefix: str) -> dict[str, float]:
    """
    Collect reference runs (combined QC / baseline) from run_id directory.
    Returns {label: mean_pearson} for each run found.
    """
    pattern = f"{run_id}/{label_prefix}*/*/eval_best.ckpt/k562_results.csv"
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        pattern2 = f"{run_id}/{label_prefix}*/*/k562_results.csv"
        csvs = sorted(results_dir.glob(pattern2))

    data = {}
    for csv in csvs:
        # Inner dir name is the run name, e.g. "qc_emb_20_lr1e-4"
        inner_dir = csv.parent if "eval_best.ckpt" not in csv.parent.name else csv.parent.parent
        label = inner_dir.name
        val = read_mean_pearson(csv)
        if val is not None:
            data[label] = val
            print(f"  [ref] {label:<30s}  mean_pearson = {val:.4f}")

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    per_source: dict[str, float],
    combined_ref: dict[str, float],
    baseline_ref: dict[str, float],
    output: Path,
) -> None:
    # Sort sources by value descending
    sources = sorted(per_source, key=per_source.__getitem__, reverse=True)
    values = [per_source[s] for s in sources]

    # Replace underscores back to hyphens for display (bash-safe names)
    display_names = [s.replace("_", "-") if s not in per_source else s for s in sources]
    # Actually just use the original keys for display
    display_names = sources

    n = len(sources)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5))

    # Color bars by value (blue gradient)
    norm = plt.Normalize(min(values) - 0.01, max(values) + 0.01)
    cmap = plt.cm.Blues
    colors = [cmap(norm(v)) for v in values]

    bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=0.5)

    # Reference lines
    ref_styles = [
        ("combined QC",  combined_ref,  "tab:orange",  "--"),
        ("baseline",     baseline_ref,  "tab:red",     ":"),
    ]
    added_labels = set()
    for group_label, ref_dict, color, ls in ref_styles:
        for run_name, ref_val in ref_dict.items():
            lbl = f"{group_label}: {run_name} ({ref_val:.4f})"
            ax.axhline(ref_val, color=color, linestyle=ls, linewidth=1.5, alpha=0.85, label=lbl)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) - min(values)) * 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=6, rotation=90,
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Pearson Correlation")
    ax.set_title("Per-Source QuantumCell Embedding Ablation — Mean Pearson on K562")

    if combined_ref or baseline_ref:
        ax.legend(fontsize=8, loc="lower right")

    y_min = min(values + list(combined_ref.values()) + list(baseline_ref.values())) if values else 0
    y_max = max(values + list(combined_ref.values()) + list(baseline_ref.values())) if values else 1
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin * 2)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"\nSaved plot to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot per-source mean Pearson barplot.")
    parser.add_argument("--per_source_run_id", default="21", help="Run ID for per-source results")
    parser.add_argument("--combined_run_id",   default=None, help="Run ID with combined+baseline runs (optional)")
    parser.add_argument("--results_dir",       default="results", help="Root results directory")
    parser.add_argument("--output",            default="per_source_pearson.png", help="Output PNG path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Per-source runs (run_id={args.per_source_run_id}) ---")
    per_source = collect_per_source(results_dir, args.per_source_run_id)

    combined_ref, baseline_ref = {}, {}
    if args.combined_run_id:
        print(f"\n--- Combined QC reference (run_id={args.combined_run_id}) ---")
        combined_ref = collect_reference(results_dir, args.combined_run_id, "qc_emb")
        print(f"\n--- Baseline reference (run_id={args.combined_run_id}) ---")
        baseline_ref = collect_reference(results_dir, args.combined_run_id, "baseline")

    if not per_source:
        print("\nERROR: no per-source results found. Check the paths.", file=sys.stderr)
        sys.exit(1)

    plot(per_source, combined_ref, baseline_ref, Path(args.output))


if __name__ == "__main__":
    main()
