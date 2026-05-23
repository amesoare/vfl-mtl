"""
figures/plot_results_summary.py

Summary figure across all 4 experiments.
Plots validation metrics (AUROC / kappa) from the final training round,
mean ± std across seeds.

Usage:
    python figures/plot_results_summary.py \
        --results_dir results/ --output figures/results_summary.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Brand palette — 7 HEX codes
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

_DISP = {"VFL-MTL": "PRISM"}  # model-name → display label

PALETTE = {
    # Exp 1 — model variants
    "VFL-MTL":   _C[0],
    "PRISM":     _C[0],   # display alias
    "ST-IHM":    _C[1],
    "ST-Decomp": _C[2],
    "ST-Pheno":  _C[3],
    # Exp 2 — split configs
    "default":   _C[4],
    "balanced":  _C[5],
    "skewed":    _C[6],
    # Exp 3 — task configs
    "all_tasks": _C[0],
    "ihm_only":  _C[1],
    "ihm_decomp": _C[2],
    "ihm_pheno": _C[3],
    # Exp 4 — n_sites
    2:           _C[4],
    3:           _C[5],
}
_FALLBACK = _C[6]


def last_round(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final training round per (model/config, seed) pair."""
    group_cols = [c for c in ("model", "split_config", "task_config", "n_sites") if c in df.columns]
    if not group_cols:
        return df
    idx = df.groupby(group_cols + ["seed"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def bar_group(ax, groups, values, errors, ylabel, title, rotation=0):
    x = np.arange(len(groups))
    colors = [PALETTE.get(g, _FALLBACK) for g in groups]
    ax.bar(x, values, yerr=errors, capsize=5,
           color=colors, alpha=0.90, width=0.55, ecolor="#555555", error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotation, ha="right" if rotation else "center", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def agg_metric(df: pd.DataFrame, groupby_col: str, metric: str):
    """Mean and std of metric across seeds, for available rows only."""
    sub = df[df[metric].notna()]
    g = sub.groupby(groupby_col)[metric]
    return g.mean(), g.std().fillna(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output",      default="figures/round1_summary.png")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    exp1      = last_round(pd.read_csv(rd / "exp1.csv"))
    exp2_split = last_round(pd.read_csv(rd / "exp2_split.csv")) if (rd / "exp2_split.csv").exists() else pd.DataFrame()
    exp2      = last_round(pd.read_csv(rd / "exp2.csv"))
    exp3      = last_round(pd.read_csv(rd / "exp3.csv"))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "PRISM Preliminary Results — Validation Metrics, Final Round (mean ± std, 3 seeds)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Exp 1 · IHM AUROC — only models that train IHM ──────────────────────
    # ST-Decomp and ST-Pheno have IHM loss weight = 0; their IHM head is
    # untrained (random init), so val_ihm_auroc is noise — excluded here.
    ax = axes[0, 0]
    ihm_models = exp1[exp1["model"].isin(["VFL-MTL", "ST-IHM"])]
    mu, sd = agg_metric(ihm_models, "model", "val_ihm_auroc")
    bar_group(ax, [_DISP.get(g, g) for g in mu.index.tolist()], mu.values, sd.values,
              "IHM AUROC", "Exp1 · IHM AUROC\nPRISM vs ST-IHM")
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="chance")
    ax.legend(fontsize=7)

    # ── Exp 1 · Pheno Macro-AUC — only models that train Pheno ──────────────
    ax = axes[0, 1]
    pheno_models = exp1[exp1["model"].isin(["VFL-MTL", "ST-Pheno"])]
    mu, sd = agg_metric(pheno_models, "model", "val_pheno_macro_auroc")
    bar_group(ax, [_DISP.get(g, g) for g in mu.index.tolist()], mu.values, sd.values,
              "Pheno Macro-AUC", "Exp1 · Pheno Macro-AUC\nPRISM vs ST-Pheno")
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="chance")
    ax.legend(fontsize=7)

    # ── Exp 1 · Decomp AUC-ROC — only models that train Decomp ─────────────
    ax = axes[1, 0]
    decomp_models = exp1[exp1["model"].isin(["VFL-MTL", "ST-Decomp"])]
    mu, sd = agg_metric(decomp_models, "model", "val_decomp_auroc")
    bar_group(ax, [_DISP.get(g, g) for g in mu.index.tolist()], mu.values, sd.values,
              "Decomp AUC-ROC", "Exp1 · Decomp AUC-ROC\nPRISM vs ST-Decomp")

    # ── Exp 2 (omitted) · IHM AUROC ──────────────────────────────────────────
    ax = axes[0, 2]
    if not exp2_split.empty and "val_ihm_auroc" in exp2_split.columns:
        mu, sd = agg_metric(exp2_split, "split_config", "val_ihm_auroc")
        bar_group(ax, mu.index.tolist(), mu.values, sd.values,
                  "IHM AUROC", "Exp2 · IHM AUROC\nFeature Split Sensitivity", rotation=15)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    else:
        ax.set_title("Exp2 · IHM AUROC\n(omitted)"); ax.set_visible(False)

    # ── Exp 2 (omitted) · Pheno Macro-AUC ────────────────────────────────────
    ax = axes[1, 1]
    if not exp2_split.empty and "val_pheno_macro_auroc" in exp2_split.columns:
        mu, sd = agg_metric(exp2_split, "split_config", "val_pheno_macro_auroc")
        bar_group(ax, mu.index.tolist(), mu.values, sd.values,
                  "Pheno Macro-AUC", "Exp2 · Pheno Macro-AUC\nFeature Split Sensitivity", rotation=15)
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
    else:
        ax.set_visible(False)

    # ── Exp 2 · IHM AUROC ────────────────────────────────────────────────────
    ax = axes[0, 3]
    mu, sd = agg_metric(exp2, "task_config", "val_ihm_auroc")
    bar_group(ax, mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp2 · IHM AUROC\nTask Relatedness", rotation=20)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    # ── Exp 2 · Pheno Macro-AUC ──────────────────────────────────────────────
    ax = axes[1, 2]
    mu, sd = agg_metric(exp2, "task_config", "val_pheno_macro_auroc")
    bar_group(ax, mu.index.tolist(), mu.values, sd.values,
              "Pheno Macro-AUC", "Exp2 · Pheno Macro-AUC\nTask Relatedness", rotation=20)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    # ── Exp 3 · IHM AUROC + elapsed time ─────────────────────────────────────
    ax = axes[1, 3]
    mu, sd = agg_metric(exp3, "n_sites", "val_ihm_auroc")
    bar_group(ax, mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp3 · IHM AUROC\nScalability (n_sites)")
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    mu_t, _ = agg_metric(exp3, "n_sites", "elapsed_s")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(mu_t)), mu_t.values, color=_C[6],
             linestyle="--", marker="o", markersize=4, linewidth=1.2)
    ax2.set_ylabel("Elapsed (s)", fontsize=7, color=_C[6])
    ax2.tick_params(axis="y", labelsize=7, colors=_C[6])

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
