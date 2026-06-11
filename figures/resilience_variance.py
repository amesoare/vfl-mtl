"""
figures/resilience_variance.py — DP Resilience Variance Plot.

Single panel:
  x-axis: ε
  y-axis: std(AUC) across seeds — variance inflation index
  One line per task (IHM, Decomp, Pheno)

Usage:
    python figures/resilience_variance.py \
        --input results/privacy_utility_combined.csv \
        --output figures/resilience_variance.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         15,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":    17,
    "axes.titleweight":  "normal",
    "axes.labelsize":    15,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "legend.fontsize":   12,
})

# Brand palette — matches plot_results_summary.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

TASK_COLS = {
    "IHM":    ("val_ihm_auroc",         _C[1]),   # purple
    "Decomp": ("val_decomp_auroc",      _C[2]),   # dark purple/navy
    "Pheno":  ("val_pheno_macro_auroc", _C[3]),   # dark red
}
TASK_COLS_TEST = {
    "IHM":    ("ihm_auroc",         _C[1]),
    "Decomp": ("decomp_auroc",      _C[2]),
    "Pheno":  ("pheno_macro_auroc", _C[3]),
}
EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", "∞"]


CLINICAL_FLOORS_RV = {"IHM": 0.75, "Decomp": 0.70, "Pheno": 0.65}


def _plot_resilience_std(df: pd.DataFrame, output: Path,
                         task_cols: dict | None = None) -> None:
    """std(AUC) across seeds per task as a function of ε."""
    if task_cols is None:
        task_cols = TASK_COLS_TEST

    fig, ax = plt.subplots(figsize=(14, 8))

    for task_name, (task_col, color) in task_cols.items():
        x_plot, y_std = [], []
        for xi, eps in enumerate(EPS_ORDER):
            sub = df[df["epsilon_level"] == eps][task_col].dropna()
            if sub.empty:
                continue
            x_plot.append(xi)
            y_std.append(float(sub.std()))
        ax.plot(x_plot, y_std, color=color, marker="o", ms=8,
                linewidth=2.5, label=task_name)

    ax.set_xticks(range(len(EPS_ORDER)))
    ax.set_xticklabels(EPS_LABELS, fontsize=26)
    ax.set_xlabel("Privacy budget ε", fontsize=28)
    ax.set_ylabel("Std(AUC-ROC) across seeds", fontsize=28)
    ax.set_title(
        "AUC-ROC standard deviation across seeds as a function of privacy budget ε",
        fontsize=32,
    )
    ax.tick_params(axis="y", labelsize=26)
    ax.legend(fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/privacy_utility_combined.csv")
    parser.add_argument("--output", default="figures/resilience_variance.png")
    parser.add_argument("--mode",   default="uniform",
                        help="Which DP mode to plot (uniform or stratified)")
    args = parser.parse_args()

    out = Path(args.output)

    test_dp = Path("results/test_results_dp.csv")
    if test_dp.exists():
        df = pd.read_csv(test_dp)
        df["epsilon_level"] = pd.to_numeric(
            df["epsilon_level"], errors="coerce").fillna(float("inf"))
        _plot_resilience_std(df, out, task_cols=TASK_COLS_TEST)
    else:
        df = pd.read_csv(args.input)
        df["epsilon_level"] = pd.to_numeric(
            df["epsilon_level"], errors="coerce").fillna(float("inf"))
        df = df.groupby(["mode", "epsilon_level", "seed"]).last().reset_index()
        _plot_resilience_std(df[df["mode"] == args.mode], out)


if __name__ == "__main__":
    main()
