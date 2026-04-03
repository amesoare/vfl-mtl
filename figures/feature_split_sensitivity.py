"""
figures/feature_split_sensitivity.py

Grouped bar chart of final-round task loss per feature split configuration (Exp 2).
Three bars per configuration (IHM / LOS / Pheno), with error bars across seeds.

Usage:
    python figures/feature_split_sensitivity.py \
        --exp2 results/exp2.csv --output figures/feature_split_sensitivity.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

TASKS  = ["ihm_loss", "los_loss", "pheno_loss"]
LABELS = ["IHM",      "LOS",      "Pheno"]
COLORS = ["#fbb45e",  "#afc4d5",  "#353b56"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2",   default="results/exp2.csv")
    parser.add_argument("--output", default="figures/feature_split_sensitivity.png")
    args = parser.parse_args()

    df    = pd.read_csv(args.exp2)
    final = df.groupby(["split_config", "seed"]).last().reset_index()
    agg   = final.groupby("split_config")[TASKS].agg(["mean", "std"])

    configs = agg.index.tolist()
    x       = np.arange(len(configs))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (task, label, color) in enumerate(zip(TASKS, LABELS, COLORS)):
        means = agg[task]["mean"].values
        stds  = agg[task]["std"].fillna(0).values
        ax.bar(x + i * width, means, width,
               yerr=stds, label=label, capsize=4,
               color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_xlabel("Feature Split Configuration")
    ax.set_ylabel("Training Loss (lower = better)")
    ax.set_title("Task Loss Sensitivity to Feature Split Configuration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
