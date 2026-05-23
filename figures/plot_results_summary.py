"""
figures/plot_results_summary.py

Summary figures for Paper 1 Experiments 2 and 3.
  Exp 2 (paper): task relatedness and negative transfer  — source: exp2.csv / test_exp2.csv
  Exp 3 (paper): scalability — n_sites                  — source: exp3.csv / test_exp3.csv

Produces three files:
  task_config_test.png — standalone Exp 2 figure, test metrics        (main text Fig 4)
  exp2_exp3_test.png   — combined Exp 2 + Exp 3, test metrics         (archive / appendix)
  exp2_exp3_val.png    — combined Exp 2 + Exp 3, validation metrics   (appendix)

Standalone Exp 3 (scalability) figure is produced by scalability_curves.py.

Usage:
    python figures/plot_results_summary.py \
        --results_dir results/ --output figures/exp2_exp3_val.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":    12,
    "axes.titleweight":  "normal",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
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
SITE_COLORS = {2: _C[6], 3: _C[3]}  # matches scalability_curves.py

FLOORS = {"ihm": 0.75, "decomp": 0.70, "pheno": 0.65}


def last_round(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ("model", "task_config", "n_sites") if c in df.columns]
    if not group_cols:
        return df
    idx = df.groupby(group_cols + ["seed"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def bar_panel(ax, groups, colors, values, errors, ylabel, title,
              floor=None, xlabel=None, rotation=0):
    x = np.arange(len(groups))
    ax.bar(x, values, yerr=errors, capsize=4,
           color=colors, alpha=0.88, width=0.62,
           ecolor="#444444", error_kw={"linewidth": 1.0, "capthick": 1.0},
           zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.grid(True, alpha=0.25, axis="y", zorder=0)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if floor is not None:
        ax.axhline(floor, color="#888888", linestyle="--", linewidth=0.8,
                   label=f"floor {floor:.2f}")
        ax.legend(fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel)


def main_val(rd: Path, output: Path) -> None:
    exp2 = last_round(pd.read_csv(rd / "exp2.csv"))   # paper Exp 2: task relatedness
    exp3 = last_round(pd.read_csv(rd / "exp3.csv"))   # paper Exp 3: scalability

    # 2×4 grid — col 3 unused in both rows
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # ── Row 0: Experiment 2 — task relatedness ───────────────────────────────
    present  = [c for c in CONFIG_ORDER if c in exp2["task_config"].values]
    labels2  = [CONFIG_LABELS[c] for c in present]
    colors2  = [CONFIG_COLORS[c] for c in present]

    task_panels_2 = [
        ("val_ihm_auroc",         "IHM AUC-ROC",    FLOORS["ihm"],   "IHM AUC-ROC by task configuration"),
        ("val_decomp_auroc",      "Decomp AUC-ROC",  FLOORS["decomp"],"Decompensation AUC-ROC by task configuration"),
        ("val_pheno_macro_auroc", "Pheno Macro-AUC", FLOORS["pheno"], "Phenotyping macro-AUC by task configuration"),
    ]
    for col, (metric, ylabel, floor, title) in enumerate(task_panels_2):
        mu = exp2.groupby("task_config")[metric].mean().reindex(present)
        sd = exp2.groupby("task_config")[metric].std().fillna(0).reindex(present)
        bar_panel(axes[0, col], labels2, colors2,
                  mu.values, sd.values, ylabel,
                  title, floor=floor, rotation=12)
    axes[0, 3].set_visible(False)

    # ── Row 1: Experiment 3 — scalability ────────────────────────────────────
    sites   = sorted(exp3["n_sites"].unique())
    labels3 = [str(s) for s in sites]
    colors3 = [SITE_COLORS.get(s, _C[6]) for s in sites]

    task_panels_3 = [
        ("val_ihm_auroc",    "IHM AUC-ROC",   FLOORS["ihm"],   "IHM AUC-ROC by number of sites"),
        ("val_decomp_auroc", "Decomp AUC-ROC", FLOORS["decomp"],"Decompensation AUC-ROC by number of sites"),
    ]
    for col, (metric, ylabel, floor, title) in enumerate(task_panels_3):
        mu = exp3.groupby("n_sites")[metric].mean().reindex(sites)
        sd = exp3.groupby("n_sites")[metric].std().fillna(0).reindex(sites)
        bar_panel(axes[1, col], labels3, colors3,
                  mu.values, sd.values, ylabel,
                  title, floor=floor, xlabel="n_sites")

    # wall-clock panel
    ax   = axes[1, 2]
    mu_t = exp3.groupby("n_sites")["elapsed_s"].mean().reindex(sites)
    sd_t = exp3.groupby("n_sites")["elapsed_s"].std().fillna(0).reindex(sites)
    ax.bar(labels3, mu_t.values, yerr=sd_t.values, capsize=4,
           color=colors3, alpha=0.88, width=0.62,
           ecolor="#444444", error_kw={"linewidth": 1.0, "capthick": 1.0},
           zorder=3)
    ax.set_xticks(range(len(labels3)))
    ax.set_xticklabels(labels3)
    ax.set_ylabel("Elapsed (s/round)")
    ax.set_xlabel("n_sites")
    ax.set_title("Per-round wall-clock time by number of sites", pad=6)
    ax.grid(True, alpha=0.25, axis="y", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (v, e) in enumerate(zip(mu_t.values, sd_t.values)):
        ax.annotate(f"{v:.1f}s", xy=(i, v + e),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11)

    axes[1, 3].set_visible(False)

    fig.tight_layout(h_pad=3.5)
    _add_row_titles(fig, axes)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def _add_row_titles(fig, axes) -> None:
    """Add centered bold row titles in figure coordinates (call after tight_layout)."""
    row_titles = [
        "Experiment 2: Task Relatedness and Negative Transfer",
        "Experiment 3: Scalability",
    ]
    for row, title in enumerate(row_titles):
        visible = [axes[row, col] for col in range(axes.shape[1])
                   if axes[row, col].get_visible()]
        if not visible:
            continue
        x_mid = (min(ax.get_position().x0 for ax in visible) +
                 max(ax.get_position().x1 for ax in visible)) / 2
        y_top = max(ax.get_position().y1 for ax in visible)
        fig.text(x_mid, y_top + 0.045, title,
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold",
                 transform=fig.transFigure)


def plot_test_summary(rd: Path, output: Path) -> None:
    """Same layout as the validation figure but using held-out test metrics.

    Row 1 (scalability) pulls training rounds from exp3.csv because
    test_exp3.csv is a flat test-metric file with no training-loop columns.
    """
    test2 = pd.read_csv(rd / "test_exp2.csv")   # paper Exp 2: task relatedness
    test3 = pd.read_csv(rd / "test_exp3.csv")   # paper Exp 3: scalability (test AUC)
    # actual last round lives in the training-loop CSV, not the test CSV
    conv3 = last_round(pd.read_csv(rd / "exp3.csv"))

    TEST_PANELS_2 = [
        ("ihm_auc_roc",    "IHM AUC-ROC",    FLOORS["ihm"],   "IHM AUC-ROC by task configuration"),
        ("decomp_auc_roc", "Decomp AUC-ROC",  FLOORS["decomp"],"Decompensation AUC-ROC by task configuration"),
        ("pheno_macro_auc","Pheno Macro-AUC", FLOORS["pheno"], "Phenotyping macro-AUC by task configuration"),
    ]

    TEST_PANELS_3 = [
        ("ihm_auc_roc",    "IHM AUC-ROC",    FLOORS["ihm"],   "IHM AUC-ROC by number of sites"),
        ("decomp_auc_roc", "Decomp AUC-ROC",  FLOORS["decomp"],"Decompensation AUC-ROC by number of sites"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # ── Row 0: Experiment 2 — task relatedness ───────────────────────────────
    present = [c for c in CONFIG_ORDER if c in test2["task_config"].values]
    labels2 = [CONFIG_LABELS[c] for c in present]
    colors2 = [CONFIG_COLORS[c] for c in present]

    for col, (metric, ylabel, floor, title) in enumerate(TEST_PANELS_2):
        mu = test2.groupby("task_config")[metric].mean().reindex(present)
        sd = test2.groupby("task_config")[metric].std().fillna(0).reindex(present)
        bar_panel(axes[0, col], labels2, colors2,
                  mu.values, sd.values, ylabel,
                  title, floor=floor, rotation=12)
    axes[0, 3].set_visible(False)

    # ── Row 1: Experiment 3 — scalability ────────────────────────────────────
    sites   = sorted(test3["n_sites"].unique())
    labels3 = [str(s) for s in sites]
    colors3 = [SITE_COLORS.get(s, _C[6]) for s in sites]

    # Panel col 0: actual training length (last round per n_sites/seed).
    # convergence_round is not comparable across n_sites configs — see scalability_curves.py.
    mu_c = conv3.groupby("n_sites")["round"].mean().reindex(sites)
    sd_c = conv3.groupby("n_sites")["round"].std().fillna(0).reindex(sites)
    bar_panel(axes[1, 0], labels3, colors3,
              mu_c.values, sd_c.values, "Round",
              "Training Rounds by number of sites", xlabel="n_sites")

    # Panels col 1–2: per-task test AUC
    for col, (metric, ylabel, floor, title) in enumerate(TEST_PANELS_3, start=1):
        mu = test3.groupby("n_sites")[metric].mean().reindex(sites)
        sd = test3.groupby("n_sites")[metric].std().fillna(0).reindex(sites)
        bar_panel(axes[1, col], labels3, colors3,
                  mu.values, sd.values, ylabel,
                  title, floor=floor, xlabel="n_sites")

    axes[1, 3].set_visible(False)

    fig.tight_layout(h_pad=3.5)
    _add_row_titles(fig, axes)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def plot_task_relatedness_test(rd: Path, output: Path) -> None:
    """Standalone Exp 2 figure: dot plot of test-set AUC-ROC by task configuration.

    One panel per task; x = config (only configs that train that task);
    grey jittered seed dots + task-coloured mean ± std marker;
    dashed reference at simplest config; dotted clinical floor line.
    """
    test2 = pd.read_csv(rd / "test_exp2.csv")

    _TASK_COLOR = {
        "ihm_auc_roc":     _C[1],   # purple
        "decomp_auc_roc":  _C[2],   # dark charcoal
        "pheno_macro_auc": _C[3],   # burgundy
    }

    # (metric, panel_title, configs ordered simple→complex, ref_config, floor)
    PANELS = [
        ("ihm_auc_roc",    "IHM AUC-ROC",
         ["ihm_only", "ihm_decomp", "ihm_pheno", "all_tasks"], "ihm_only",   FLOORS["ihm"]),
        ("decomp_auc_roc", "Decomp AUC-ROC",
         ["ihm_decomp", "all_tasks"],                           "ihm_decomp", FLOORS["decomp"]),
        ("pheno_macro_auc","Pheno Macro-AUC",
         ["ihm_pheno", "all_tasks"],                            "ihm_pheno",  FLOORS["pheno"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Test-set AUC by task configuration  (mean ± std, 3 seeds)",
        fontweight="normal", y=1.02,
    )

    jitter_offsets = [-0.10, 0.0, 0.10]

    for ax, (metric, title, configs, ref_cfg, floor) in zip(axes, PANELS):
        color   = _TASK_COLOR[metric]
        present = [c for c in configs if c in test2["task_config"].values]
        x_pos   = np.arange(len(present))

        # ── Seed dots (grey, jittered) ────────────────────────────────────────
        for xi, cfg in enumerate(present):
            vals = test2[test2["task_config"] == cfg][metric].dropna().values
            for vi, v in enumerate(vals):
                jx = jitter_offsets[vi] if vi < len(jitter_offsets) else 0.0
                ax.scatter(xi + jx, v, color="#bbbbbb", s=30, zorder=3,
                           linewidths=0, alpha=0.85)

        # ── Mean ± std markers ────────────────────────────────────────────────
        means, stds = [], []
        for cfg in present:
            vals = test2[test2["task_config"] == cfg][metric].dropna()
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

        ax.errorbar(x_pos, means, yerr=stds, fmt="o", color=color,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    markersize=9, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=4)

        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.006, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#333333")

        # ── Reference dashed line at simplest config ──────────────────────────
        if ref_cfg in present:
            ref_mean = means[present.index(ref_cfg)]
            ax.axhline(ref_mean, color=color, lw=0.9, ls="--", alpha=0.5,
                       label=f"ref: {CONFIG_LABELS.get(ref_cfg, ref_cfg)}")

        # ── Clinical floor ────────────────────────────────────────────────────
        ax.axhline(floor, color="#888888", lw=0.7, ls=":",
                   label=f"floor {floor:.2f}")

        # ── Axes styling ──────────────────────────────────────────────────────
        all_vals = test2[test2["task_config"].isin(present)][metric].dropna()
        lo = max(0.0, float(all_vals.min()) - 0.04)
        hi = float(all_vals.max()) + 0.06
        ax.set_ylim(min(lo, floor - 0.04), max(hi, floor + 0.04))

        ax.set_xticks(x_pos)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in present],
                           rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output",      default="figures/exp2_exp3_val.png")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    out = Path(args.output)
    main_val(rd, out)
    plot_test_summary(rd, out.parent / "exp2_exp3_test.png")
    plot_task_relatedness_test(rd, out.parent / "task_config_test.png")


if __name__ == "__main__":
    main()
