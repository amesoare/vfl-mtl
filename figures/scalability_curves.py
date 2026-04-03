"""
figures/scalability_curves.py

Two-panel figure from Exp 4:
  Left : rounds-to-convergence bar chart (mean ± std across seeds) per n_sites
  Right: loss curves per n_sites (mean across seeds)

Usage:
    python figures/scalability_curves.py \
        --exp4 results/exp4.csv --output figures/scalability.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

COLORS = ["#afc4d5", "#7d7585"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp4",   default="results/exp4.csv")
    parser.add_argument("--output", default="figures/scalability.png")
    args = parser.parse_args()

    df = pd.read_csv(args.exp4)

    # One convergence_round value per (n_sites, seed)
    summary = df.groupby(["n_sites", "seed"])["convergence_round"].first().reset_index()
    agg = summary.groupby("n_sites")["convergence_round"].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: convergence round bar chart
    x_labels = agg.index.astype(str).tolist()
    axes[0].bar(
        x_labels, agg["mean"],
        yerr=agg["std"], capsize=5,
        color=COLORS[:len(x_labels)], alpha=0.85,
    )
    axes[0].set_xlabel("Number of Institutions")
    axes[0].set_ylabel("Rounds to Convergence")
    axes[0].set_title("Convergence Speed vs. n_sites")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: loss curves per n_sites
    for (n_sites, grp), color in zip(df.groupby("n_sites"), COLORS):
        mean_loss = grp.groupby("round")["train_loss"].mean()
        std_loss  = grp.groupby("round")["train_loss"].std().fillna(0)
        axes[1].plot(mean_loss.index, mean_loss.values,
                     label=f"{n_sites} sites", color=color)
        axes[1].fill_between(mean_loss.index,
                             mean_loss - std_loss, mean_loss + std_loss,
                             alpha=0.2, color=color)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Loss Curves by n_sites")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
