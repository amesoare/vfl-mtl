"""
figures/plot_results_summary.py

Exp 2: task relatedness violin plot (test set).

Produces:
  mimic_task_relatedness.png — test-set AUC by task configuration

Usage:
    python figures/plot_results_summary.py \
        --results_dir results/ --output figures/mimic_task_relatedness.png
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

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
    "axes.linewidth":    0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
})

CONFIG_ORDER = ["all_tasks", "ihm_only", "ihm_decomp", "ihm_pheno"]
CONFIG_LABELS = {
    "all_tasks":  "All tasks",
    "ihm_only":   "IHM only",
    "ihm_decomp": "IHM+Decomp",
    "ihm_pheno":  "IHM+Pheno",
}
CONFIG_COLORS = {
    "all_tasks":  _C[0],
    "ihm_only":   _C[1],
    "ihm_decomp": _C[2],
    "ihm_pheno":  _C[3],
}

FLOORS = {"ihm": 0.75, "decomp": 0.70, "pheno": 0.65}


def plot_task_config_violin(rd: Path, output: Path) -> None:
    """Exp 2 figure: grouped violin plot of test AUC by task configuration.

    Three panels (one per task); x = task config; violin body per config;
    individual seed dots overlaid; mean ± std diamond marker on top.
    """
    test2 = pd.read_csv(rd / "test_exp2.csv")

    _TASK_COLOR = {
        "ihm_auc_roc":     _C[1],
        "decomp_auc_roc":  _C[2],
        "pheno_macro_auc": _C[3],
    }

    PANELS = [
        ("ihm_auc_roc",    "IHM AUC-ROC",
         ["ihm_only", "ihm_decomp", "ihm_pheno", "all_tasks"], FLOORS["ihm"]),
        ("decomp_auc_roc", "Decomp AUC-ROC",
         ["ihm_decomp", "all_tasks"],                          FLOORS["decomp"]),
        ("pheno_macro_auc", "Pheno Macro-AUC",
         ["ihm_pheno", "all_tasks"],                           FLOORS["pheno"]),
    ]

    jitter_offsets = [-0.11, 0.0, 0.11]

    fig, axes = plt.subplots(3, 1, figsize=(15, 22))
    fig.suptitle(
        "Test-set AUC by Task Configuration",
        fontsize=35, fontweight="normal",
    )

    for ax, (metric, title, configs, floor) in zip(axes, PANELS):
        color   = _TASK_COLOR[metric]
        present = [c for c in configs if c in test2["task_config"].values]
        data_per_cfg = [
            test2[test2["task_config"] == c][metric].dropna().values
            for c in present
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parts = ax.violinplot(
                data_per_cfg,
                positions=range(len(present)),
                showmedians=False,
                showextrema=False,
                widths=0.65,
            )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_edgecolor(color)
            body.set_linewidth(1.2)

        for xi, vals in enumerate(data_per_cfg):
            for vi, v in enumerate(vals):
                jx = jitter_offsets[vi] if vi < len(jitter_offsets) else 0.0
                ax.scatter(xi + jx, v, color=color, s=70, zorder=4,
                           linewidths=0, alpha=0.85)

        means = [float(np.mean(v)) if len(v) else np.nan for v in data_per_cfg]
        stds  = [
            float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            for v in data_per_cfg
        ]
        ax.errorbar(
            range(len(present)), means, yerr=stds,
            fmt="D", color=color,
            capsize=6, capthick=1.8, elinewidth=1.8,
            markersize=9, markeredgecolor="white",
            markeredgewidth=1.0, zorder=5,
        )
        for xi, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                ax.text(xi, m + s + 0.010, f"{m:.3f}",
                        ha="center", va="bottom", fontsize=31,
                        fontweight="bold", color="#222222")

        ax.axhline(floor, color="#888888", lw=1.0, ls=":",
                   label=f"Clinical floor ({floor:.2f})")

        all_vals = np.concatenate(data_per_cfg) if any(len(v) for v in data_per_cfg) else np.array([])
        if len(all_vals):
            lo = max(0.0, float(all_vals.min()) - 0.06)
            hi = float(all_vals.max()) + 0.10
            ax.set_ylim(min(lo, floor - 0.04), max(hi, floor + 0.05))

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(
            [CONFIG_LABELS.get(c, c) for c in present],
            rotation=10, ha="right", fontsize=26,
        )
        ax.set_ylabel(title, fontsize=28)
        ax.set_title(title, fontsize=32, fontweight="bold", pad=6)
        ax.legend(fontsize=22, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=26)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(left=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output",      default="figures/mimic_task_relatedness.png")
    args = parser.parse_args()

    plot_task_config_violin(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
