"""
figures/resilience_variance.py — Figure 2: DP Resilience Variance Plot.

Answers SRQ1: how much does DP stochasticity destabilise training?

Single panel:
  x-axis: ε (log scale)
  y-axis: std(AUC) across seeds — variance inflation index
  One line per task (IHM, Decomp, Pheno)

Usage:
    python figures/resilience_variance.py \
        --input results/privacy_utility.csv \
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

# Brand palette — matches plot_results_summary.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

TASK_COLS = {
    "IHM":    ("val_ihm_auroc",         _C[1]),   # purple
    "Decomp": ("val_decomp_auroc",      _C[2]),   # dark purple/navy
    "Pheno":  ("val_pheno_macro_auroc", _C[3]),   # dark red
}
EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", "∞"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/privacy_utility.csv")
    parser.add_argument("--output", default="figures/resilience_variance.png")
    parser.add_argument("--mode",   default="uniform",
                        help="Which DP mode to plot (uniform or stratified)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))

    df = df.groupby(["mode", "epsilon_level", "seed"]).last().reset_index()
    df = df[df["mode"] == args.mode]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for task_name, (task_col, color) in TASK_COLS.items():
        x_plot, y_std = [], []
        for eps in EPS_ORDER:
            sub = df[df["epsilon_level"] == eps][task_col]
            if sub.empty:
                continue
            x_plot.append(eps if eps != float("inf") else 20.0)
            y_std.append(float(sub.std()))

        ax.plot(x_plot, y_std, color=color, marker="o", ms=4, linewidth=1.4,
                label=task_name)

    x_ticks = [e if e != float("inf") else 20.0 for e in EPS_ORDER]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(EPS_LABELS, fontsize=8)
    ax.set_xlabel("Privacy budget ε", fontsize=8)
    ax.set_ylabel("Std(AUC-ROC) across seeds", fontsize=8)
    ax.set_title(f"Training Variance Under DP Noise ({args.mode} σ)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
