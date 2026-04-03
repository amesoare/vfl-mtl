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
_C = ["#fbb45e", "#5b3b3e", "#ad8d86", "#c99379", "#afc4d5", "#7d7585", "#353b56"]

PALETTE = {
    # Exp 1 — model variants
    "VFL-MTL":   _C[0],
    "ST-IHM":    _C[1],
    "ST-LOS":    _C[2],
    "ST-Pheno":  _C[3],
    # Exp 2 — split configs
    "default":   _C[4],
    "balanced":  _C[5],
    "skewed":    _C[6],
    # Exp 3 — task configs
    "all_tasks": _C[0],
    "ihm_only":  _C[1],
    "ihm_los":   _C[2],
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
    parser.add_argument("--output",      default="figures/results_summary.png")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    exp1 = last_round(pd.read_csv(rd / "exp1.csv"))
    exp2 = last_round(pd.read_csv(rd / "exp2.csv"))
    exp3 = last_round(pd.read_csv(rd / "exp3.csv"))
    exp4 = last_round(pd.read_csv(rd / "exp4.csv"))

    # 3×4 grid: one row per experiment, columns = IHM / LOS / Pheno / spare
    # Exp3 only uses col 0 (IHM AUROC — the only metric present in all configs);
    # Exp4 goes in col 1 of row 2. Unused cells are hidden.
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        "VFL-MTL Preliminary Results — Validation Metrics, Final Round (mean ± std, 3 seeds)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Row 0 · Exp 1 ────────────────────────────────────────────────────────
    # Only show models that actually train the task on each subplot.
    # ST-LOS/ST-Pheno IHM heads are untrained (weight=0) → excluded from IHM plot.
    ihm_models   = exp1[exp1["model"].isin(["VFL-MTL", "ST-IHM"])]
    los_models   = exp1[exp1["model"].isin(["VFL-MTL", "ST-LOS"])]
    pheno_models = exp1[exp1["model"].isin(["VFL-MTL", "ST-Pheno"])]

    mu, sd = agg_metric(ihm_models, "model", "val_ihm_auroc")
    bar_group(axes[0, 0], mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp1 · IHM AUROC\nVFL-MTL vs ST-IHM")
    axes[0, 0].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="chance")
    axes[0, 0].legend(fontsize=7)

    mu, sd = agg_metric(los_models, "model", "val_los_kappa")
    bar_group(axes[0, 1], mu.index.tolist(), mu.values, sd.values,
              "LOS Cohen's κ", "Exp1 · LOS Kappa\nVFL-MTL vs ST-LOS")
    axes[0, 1].axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, label="chance (κ=0)")
    axes[0, 1].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01),
                      borderaxespad=0, frameon=False)

    mu, sd = agg_metric(pheno_models, "model", "val_pheno_macro_auroc")
    bar_group(axes[0, 2], mu.index.tolist(), mu.values, sd.values,
              "Pheno Macro-AUC", "Exp1 · Pheno Macro-AUC\nVFL-MTL vs ST-Pheno")
    axes[0, 2].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, label="chance")
    axes[0, 2].legend(fontsize=7)

    axes[0, 3].set_visible(False)  # col 3 unused for Exp1

    # ── Row 1 · Exp 2 ────────────────────────────────────────────────────────
    mu, sd = agg_metric(exp2, "split_config", "val_ihm_auroc")
    bar_group(axes[1, 0], mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp2 · IHM AUROC\nFeature Split Sensitivity", rotation=15)
    axes[1, 0].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    mu, sd = agg_metric(exp2, "split_config", "val_los_kappa")
    bar_group(axes[1, 1], mu.index.tolist(), mu.values, sd.values,
              "LOS Cohen's κ", "Exp2 · LOS Kappa\nFeature Split Sensitivity", rotation=15)
    axes[1, 1].axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, label="chance (κ=0)")
    axes[1, 1].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01),
                      borderaxespad=0, frameon=False)

    mu, sd = agg_metric(exp2, "split_config", "val_pheno_macro_auroc")
    bar_group(axes[1, 2], mu.index.tolist(), mu.values, sd.values,
              "Pheno Macro-AUC", "Exp2 · Pheno Macro-AUC\nFeature Split Sensitivity", rotation=15)
    axes[1, 2].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    axes[1, 3].set_visible(False)  # col 3 unused for Exp2

    # ── Row 2 · Exp 3 (col 0 only) + Exp 4 (col 1) ──────────────────────────
    # Exp3's question is whether task pairing affects IHM — all 4 configs train IHM,
    # so IHM AUROC is the single sufficient metric. LOS/Pheno subplots are omitted:
    # only 2 configs train each, there's no LOS-only/Pheno-only baseline in Exp3,
    # and the question is answered entirely by the IHM comparison.
    mu, sd = agg_metric(exp3, "task_config", "val_ihm_auroc")
    bar_group(axes[2, 0], mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp3 · IHM AUROC\nTask Relatedness", rotation=20)
    axes[2, 0].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    # Exp4: IHM AUROC by n_sites + elapsed time on twin axis
    ax = axes[2, 1]
    mu, sd = agg_metric(exp4, "n_sites", "val_ihm_auroc")
    bar_group(ax, mu.index.tolist(), mu.values, sd.values,
              "IHM AUROC", "Exp4 · IHM AUROC\nScalability (n_sites)")
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)

    mu_t, _ = agg_metric(exp4, "n_sites", "elapsed_s")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(mu_t)), mu_t.values, color=_C[6],
             linestyle="--", marker="o", markersize=4, linewidth=1.2)
    ax2.set_ylim(0, max(mu_t.values) * 1.3)
    ax2.set_ylabel("Elapsed (s)", fontsize=7, color=_C[6])
    ax2.tick_params(axis="y", labelsize=7, colors=_C[6])
    for x_pos, val in enumerate(mu_t.values):
        ax2.annotate(f"{val:.0f}s", xy=(x_pos, val),
                     xytext=(6, 4), textcoords="offset points",
                     fontsize=7, color=_C[6])

    axes[2, 2].set_visible(False)
    axes[2, 3].set_visible(False)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
