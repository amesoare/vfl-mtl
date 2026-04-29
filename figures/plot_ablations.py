"""
figures/plot_ablations.py

Ablation study figure for Paper 1 (Week 4).
Grouped bar chart: final-round validation metrics, mean ± std across 3 seeds.
VFL-MTL (full system) shown as gold reference bar; Δ delta annotations vs. VFL-MTL.

Layout: 1 row × 3 columns — IHM AUROC | Decomp AUROC | Pheno Macro-AUC

Usage:
    python figures/plot_ablations.py \
        --input results/ablations.csv --output figures/ablations.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Display names and ordering
# ---------------------------------------------------------------------------
MODEL_ORDER = [
    "VFL-MTL",
    "abl_no_mmoe",
    "abl_uniform_gating",
    "abl_experts_2",
    "abl_experts_8",
    "abl_embed_32",
    "abl_embed_128",
]

LABELS = {
    "VFL-MTL":           "VFL-MTL\n(full)",
    "abl_no_mmoe":       "No MMoE\n(MLP)",
    "abl_uniform_gating":"Uniform\nGating",
    "abl_experts_2":     "Experts\n= 2",
    "abl_experts_8":     "Experts\n= 8",
    "abl_embed_32":      "Embed\n= 32",
    "abl_embed_128":     "Embed\n= 128",
}

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

COLORS = {
    "VFL-MTL":           _C[0],
    "abl_no_mmoe":       _C[1],
    "abl_uniform_gating":_C[2],
    "abl_experts_2":     _C[3],
    "abl_experts_8":     _C[4],
    "abl_embed_32":      _C[5],
    "abl_embed_128":     _C[6],
}

# Group separators: draw a faint vertical divider after these x-positions
# groups: [VFL-MTL] | [no_mmoe, uniform_gating] | [experts_2, experts_8] | [embed_32, embed_128]
GROUP_DIVIDERS = [0.5, 2.5, 4.5]
GROUP_LABELS   = ["Reference", "MMoE ablations", "Expert count", "Embed dim"]
GROUP_SPANS    = [(0, 0), (1, 2), (3, 4), (5, 6)]  # (x_start, x_end) indices

METRICS = [
    ("val_ihm_auroc",         "IHM AUC-ROC",         "IHM AUC-ROC"),
    ("val_decomp_auroc",      "Decomp AUC-ROC",       "Decomp AUC-ROC"),
    ("val_pheno_macro_auroc", "Pheno Macro-AUC",      "Pheno Macro-AUC"),
]


def last_round(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["model", "seed"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def agg(df: pd.DataFrame, model_order: list[str], metric: str):
    g = df.groupby("model")[metric]
    mu = g.mean().reindex(model_order)
    sd = g.std().reindex(model_order).fillna(0)
    return mu, sd


def draw_panel(ax, mu, sd, metric, ylabel, title, ref_mu):
    n = len(MODEL_ORDER)
    x = np.arange(n)
    bar_w = 0.62

    bars = ax.bar(
        x, mu.values, yerr=sd.values,
        width=bar_w, capsize=4, alpha=0.88,
        color=[COLORS[m] for m in MODEL_ORDER],
        ecolor="#444444", error_kw={"linewidth": 1.0, "capthick": 1.0},
        zorder=3,
    )

    # Reference line at VFL-MTL mean
    ax.axhline(ref_mu, color=_C[0], linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=2)

    # Group separator lines
    for xd in GROUP_DIVIDERS:
        ax.axvline(xd, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)

    # Δ delta annotations above each non-reference bar
    for i, model in enumerate(MODEL_ORDER):
        if model == "VFL-MTL":
            ax.text(i, mu.values[i] + sd.values[i] + 0.003,
                    f"{mu.values[i]:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color="#444444")
        else:
            delta = mu.values[i] - ref_mu
            sign  = "+" if delta >= 0 else "−"
            ax.text(i, mu.values[i] + sd.values[i] + 0.003,
                    f"{sign}{abs(delta):.3f}", ha="center", va="bottom",
                    fontsize=6.5,
                    color=_C[4] if delta >= 0 else _C[1])

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODEL_ORDER],
                       fontsize=7.5, linespacing=1.3)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_xlim(-0.55, n - 0.45)

    # y-limits: leave room for delta annotations
    ymin = max(0, mu.values.min() - sd.values.max() - 0.05)
    ymax = mu.values.max() + sd.values.max() + 0.05
    ax.set_ylim(ymin, ymax)

    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Group bracket labels along the top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(s + e) / 2 for s, e in GROUP_SPANS])
    ax2.set_xticklabels(GROUP_LABELS, fontsize=7, color="#666666")
    ax2.tick_params(top=False, length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/ablations.csv")
    parser.add_argument("--output", default="figures/ablations.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = last_round(df)
    df = df[df["model"].isin(MODEL_ORDER)]

    ref_row = df[df["model"] == "VFL-MTL"]

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        9,
        "axes.linewidth":   0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi":       150,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    fig.suptitle(
        "Ablation Study — Final-Round Validation Metrics (mean ± std, 3 seeds)\n"
        "Δ shown relative to VFL-MTL full system (dashed reference line)",
        fontsize=10, fontweight="bold", y=1.04,
    )

    for ax, (metric, ylabel, title) in zip(axes, METRICS):
        mu, sd   = agg(df, MODEL_ORDER, metric)
        ref_mu   = ref_row[metric].mean()
        draw_panel(ax, mu, sd, metric, ylabel, title, ref_mu)

    fig.tight_layout(rect=[0, 0, 1, 1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
