"""
figures/negative_transfer_heatmap.py

Task × model heatmap of loss reduction vs. IHM-only single-task baseline.
Positive = MTL reduces loss (helps); negative = MTL increases loss (negative transfer).

Usage:
    python figures/negative_transfer_heatmap.py \
        --exp3 results/exp3.csv --output figures/negative_transfer.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_delta_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each task_config, compute mean loss delta vs. 'ihm_only' baseline.
    Delta = baseline_loss - model_loss  (positive = model has lower loss = better).
    Uses final-round values per seed, then averages across seeds.
    """
    final    = df.groupby(["task_config", "seed"]).last().reset_index()
    baseline = final[final["task_config"] == "ihm_only"].set_index("seed")
    metrics  = ["ihm_loss", "decomp_loss", "pheno_loss"]
    configs  = [c for c in final["task_config"].unique() if c != "ihm_only"]

    rows = []
    for cfg in configs:
        grp = final[final["task_config"] == cfg].set_index("seed")
        row = {"task_config": cfg}
        for m in metrics:
            delta = (baseline[m] - grp[m]).mean()  # positive = MTL reduces loss
            row[m] = round(float(delta), 4)
        rows.append(row)

    return pd.DataFrame(rows).set_index("task_config")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3",   default="results/exp3.csv")
    parser.add_argument("--output", default="figures/negative_transfer.png")
    args = parser.parse_args()

    df = pd.read_csv(args.exp3)
    delta_df = build_delta_matrix(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        delta_df,
        annot=True, fmt=".3f", center=0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        xticklabels=["IHM Loss", "Decomp Loss", "Pheno Loss"],
    )
    ax.set_title("Loss Reduction vs. IHM-only Baseline\n(positive = MTL helps, negative = transfer hurts)")
    ax.set_xlabel("Task Loss")
    ax.set_ylabel("Model Configuration")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
