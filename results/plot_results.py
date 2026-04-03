"""
results/plot_results.py — Shared utilities for loading and plotting experiment results.

Usage:
    from results.plot_results import load_results, summary_table, loss_curves

    df = load_results("results/exp1.csv")
    tbl = summary_table(df, group_cols=["model"], metric_cols=["train_loss", "val_ihm_auroc"])
    print(tbl)
    loss_curves(df, group_col="model", output_path="figures/exp1_loss.png")
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/HPC use
import matplotlib.pyplot as plt


def load_results(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summary_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Return mean ± std over seeds, grouped by group_cols."""
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    return agg


def loss_curves(
    df: pd.DataFrame,
    group_col: str,
    loss_col: str = "train_loss",
    output_path: str | None = None,
) -> None:
    """Plot mean training loss curves per group (mean ± std across seeds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby(group_col):
        mean = grp.groupby("round")[loss_col].mean()
        std  = grp.groupby("round")[loss_col].std().fillna(0)
        ax.plot(mean.index, mean.values, label=name)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def val_metric_curves(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "val_ihm_auroc",
    output_path: str | None = None,
) -> None:
    """Plot mean val metric curves per group (mean ± std across seeds)."""
    if metric_col not in df.columns:
        print(f"Column '{metric_col}' not found in dataframe — skipping.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby(group_col):
        sub = grp.dropna(subset=[metric_col])
        mean = sub.groupby("round")[metric_col].mean()
        std  = sub.groupby("round")[metric_col].std().fillna(0)
        ax.plot(mean.index, mean.values, label=name)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def comparison_table(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """
    Final-round mean ± std per group, formatted as 'mean ± std' strings.
    """
    final = df.groupby([group_col, "seed"]).last().reset_index()
    agg   = final.groupby(group_col)[metric_cols].agg(["mean", "std"]).round(4)
    result = pd.DataFrame(index=agg.index)
    for col in metric_cols:
        result[col] = agg[col].apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
        )
    return result
