"""
figures/plot_baselines.py — Baseline comparison: local-only, centralized, VFL-MTL.

Reads (gracefully skips if missing):
  results/local_only_A.csv
  results/local_only_B.csv
  results/local_only_C.csv
  results/centralized.csv
  results/exp1.csv             (VFL-MTL rows only)
  results/test_results_nodp.csv

Produces:
  figures/baselines_learning_curves.png  — val AUC per epoch/round, mean ± std
  figures/baselines_final_metrics.png    — final-epoch bar chart per task
  figures/baselines_test_metrics.png     — test-set Cleveland lollipop per task
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Shared palette (matches plot_results_summary.py) ─────────────────────────
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

PALETTE = {
    # local-only site models: distinct warm/cool neutrals, no collision with ST-* task colors
    "local_A":             "#c5a77d",  # warm sand — contrasts with purple ST-IHM
    "local_B":             "#663139",  # sage green — contrasts with dark-navy ST-Decomp
    "local_C":             _C[5],      # light steel blue — contrasts with dark-red ST-Pheno
    "centralized_oracle":  _C[4],      # dark olive brown
    "VFL-MTL":             _C[0],      # mauve (PRISM)
    # single-task VFL baselines: canonical task colors, consistent with resilience_variance.py
    "ST-IHM":              _C[1],      # purple  (#6a4c7a)
    "ST-Decomp":           _C[2],      # dark navy (#2f283d) — was _C[6]; corrected to match task color
    "ST-Pheno":            _C[3],      # dark red (#8a3c48)
}

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
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RES = Path("results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f"  [SKIP] {path} not found")
    return None


def _final_stats(df: pd.DataFrame, metric: str, epoch_col: str = "epoch"):
    last = df.groupby("seed")[metric].last()
    return float(last.mean()), float(last.std(skipna=True))


def _curve(ax, df: pd.DataFrame, metric: str, label: str, color: str,
           epoch_col: str = "epoch"):
    grp  = df.groupby(epoch_col)[metric]
    mean = grp.mean()
    std  = grp.std().fillna(0)
    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=label)
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.18, color=color)


# ── Load data ─────────────────────────────────────────────────────────────────

local_a = _load(RES / "local_only_A.csv")
local_b = _load(RES / "local_only_B.csv")
local_c = _load(RES / "local_only_C.csv")
central = _load(RES / "centralized.csv")

vfl_mtl = st_ihm = st_decomp = st_pheno = None
exp1    = _load(RES / "exp1.csv")
if exp1 is not None and "model" in exp1.columns:
    def _subset(m):
        s = exp1[exp1["model"] == m].copy()
        return s if not s.empty else None
    vfl_mtl   = _subset("VFL-MTL")
    st_ihm    = _subset("ST-IHM")
    st_decomp = _subset("ST-Decomp")
    st_pheno  = _subset("ST-Pheno")


# ── Figure 1: Learning curves ─────────────────────────────────────────────────
# Three sub-plots, one per task. Each shows all available models for that task.

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
fig.suptitle("Validation AUC over training epochs and communication rounds", fontweight="normal", y=1.02)

task_panels = [
    # (ax, local_df, loc_metric, cen_metric, vfl_metric, st_df, st_key, title)
    (axes[0], local_a, "val_auc_roc",   "val_ihm_auc_roc",    "val_ihm_auroc",          st_ihm,   "ST-IHM",   "IHM AUC-ROC"),
    (axes[1], local_b, "val_auc_roc",   "val_decomp_auc_roc", "val_decomp_auroc",        st_decomp,"ST-Decomp","Decompensation AUC-ROC"),
    (axes[2], local_c, "val_macro_auc", "val_pheno_macro_auc","val_pheno_macro_auroc",   st_pheno, "ST-Pheno", "Phenotyping Macro-AUC"),
]

local_labels = ["local_A", "local_B", "local_C"]

for i, (ax, local_df, loc_metric, cen_metric, vfl_metric, st_df, st_key, title) in enumerate(task_panels):
    any_plotted = False

    if local_df is not None and loc_metric in local_df.columns:
        _curve(ax, local_df, loc_metric,
               label=f"local_{local_labels[i][-1]}",
               color=PALETTE[local_labels[i]])
        any_plotted = True

    if central is not None and cen_metric in central.columns:
        _curve(ax, central, cen_metric,
               label="centralized_oracle",
               color=PALETTE["centralized_oracle"])
        any_plotted = True

    if st_df is not None and vfl_metric in st_df.columns:
        _curve(ax, st_df, vfl_metric,
               label=st_key, color=PALETTE[st_key],
               epoch_col="round")
        any_plotted = True

    if vfl_mtl is not None and vfl_metric in vfl_mtl.columns:
        _curve(ax, vfl_mtl, vfl_metric,
               label="PRISM", color=PALETTE["VFL-MTL"],
               epoch_col="round")
        any_plotted = True

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_title(title, fontweight="normal")
    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Val AUC")
    ax.set_ylim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "baselines_learning_curves.png", dpi=150, bbox_inches="tight")
print("Saved: figures/baselines_learning_curves.png")
plt.close()


# ── Figure 2: Final-epoch validation lollipop ────────────────────────────────

bar_rows = []  # (task_label, model_label, mean, std, color)

task_defs = [
    ("IHM\nAUC-ROC",     local_a, "val_auc_roc",   central, "val_ihm_auc_roc",     st_ihm,   "val_ihm_auroc",          vfl_mtl, "val_ihm_auroc",          "local_A", "ST-IHM"),
    ("Decomp\nAUC-ROC",  local_b, "val_auc_roc",   central, "val_decomp_auc_roc",  st_decomp,"val_decomp_auroc",        vfl_mtl, "val_decomp_auroc",        "local_B", "ST-Decomp"),
    ("Pheno\nMacro-AUC", local_c, "val_macro_auc", central, "val_pheno_macro_auc", st_pheno, "val_pheno_macro_auroc",   vfl_mtl, "val_pheno_macro_auroc",   "local_C", "ST-Pheno"),
]

for task_label, loc_df, loc_m, cen_df, cen_m, st_df, st_m, vfl_df, vfl_m, loc_key, st_key in task_defs:
    if loc_df is not None and loc_m in loc_df.columns:
        mu, sd = _final_stats(loc_df, loc_m)
        bar_rows.append((task_label, f"local ({loc_key[-1]})", mu, sd, PALETTE[loc_key]))
    if cen_df is not None and cen_m in cen_df.columns:
        mu, sd = _final_stats(cen_df, cen_m)
        bar_rows.append((task_label, "centralized", mu, sd, PALETTE["centralized_oracle"]))
    if st_df is not None and st_m in st_df.columns:
        mu, sd = _final_stats(st_df, st_m, epoch_col="round")
        bar_rows.append((task_label, st_key, mu, sd, PALETTE[st_key]))
    if vfl_df is not None and vfl_m in vfl_df.columns:
        mu, sd = _final_stats(vfl_df, vfl_m, epoch_col="round")
        bar_rows.append((task_label, "PRISM", mu, sd, PALETTE["VFL-MTL"]))

if bar_rows:
    _VAL_LP_ORDER = [
        ("IHM\nAUC-ROC",     ["centralized", "PRISM", "ST-IHM",    "local (A)"]),
        ("Decomp\nAUC-ROC",  ["centralized", "PRISM", "ST-Decomp", "local (B)"]),
        ("Pheno\nMacro-AUC", ["centralized", "PRISM", "ST-Pheno",  "local (C)"]),
    ]
    _val_lkup = {(r[0], r[1]): r[2:] for r in bar_rows}

    fig_v, axes_v = plt.subplots(1, 3, figsize=(22, 10))
    fig_v.suptitle(
        "Validation Performance: Baseline Comparison",
        fontsize=35, fontweight="normal",
    )

    for ax, (task_label, model_order) in zip(axes_v, _VAL_LP_ORDER):
        lp_models, lp_means, lp_stds, lp_colors = [], [], [], []

        for disp in reversed(model_order):
            key = (task_label, disp)
            if key in _val_lkup:
                mu, sd, color = _val_lkup[key]
                lp_models.append(disp)
                lp_means.append(mu)
                lp_stds.append(sd)
                lp_colors.append(color)

        if not lp_models:
            continue

        y_pos   = np.arange(len(lp_models))
        n_items = len(lp_models)

        xlim_left  = max(0.0, min(m - s for m, s in zip(lp_means, lp_stds)) - 0.05)
        xlim_right = max(m + s for m, s in zip(lp_means, lp_stds)) + 0.09

        for yi, (mu, sd, color) in enumerate(zip(lp_means, lp_stds, lp_colors)):
            ax.hlines(yi, xlim_left, max(xlim_left, mu - sd), color=color,
                      linewidth=2.0, alpha=0.35, zorder=2)
            ax.errorbar(mu, yi, xerr=sd, fmt="none",
                        color=color, elinewidth=3.0, capsize=8, capthick=2.8,
                        alpha=0.90, zorder=3)
            ax.scatter(mu, yi, color=color, s=160, zorder=4, linewidths=0)
            ax.text(mu, yi + 0.22, f"{mu:.3f}",
                    va="bottom", ha="center", fontsize=31,
                    fontweight="bold", color="#111111")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lp_models, fontsize=28)
        ax.set_xlabel("Val AUC", fontsize=28)
        ax.set_title(task_label.replace("\n", " "), fontsize=32, pad=12, fontweight="normal")
        ax.set_xlim(xlim_left, min(xlim_right, 1.06))
        ax.set_ylim(-0.6, n_items - 0.2)
        ax.tick_params(axis="x", labelsize=26)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_v.savefig(OUT / "baselines_val_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/baselines_val_metrics.png")
    plt.close()
else:
    print("  [SKIP] No data available for validation lollipop")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Final metrics (mean ± std, 3 seeds) ──")
for task, model, mu, sd, _ in bar_rows:
    print(f"  {task.replace(chr(10),' '):20s}  {model:20s}  {mu:.4f} ± {sd:.4f}")


# ── Figure 3: baselines_test_metrics.png — Cleveland lollipop ────────────────
_trnodp = _load(RES / "test_results_nodp.csv")

if _trnodp is not None:
    _DISP_THESIS = {"VFL-MTL": "PRISM", "centralized_oracle": "centralized"}
    _trnodp_thesis = _trnodp.copy()
    if "model_disp" not in _trnodp_thesis.columns:
        _trnodp_thesis["model_disp"] = _trnodp_thesis["model"].map(
            lambda m: _DISP_THESIS.get(m, m)
        )

    _PAL_THESIS = {
        "local_A":     PALETTE["local_A"],
        "local_B":     PALETTE["local_B"],
        "local_C":     PALETTE["local_C"],
        "centralized": PALETTE["centralized_oracle"],
        "ST-IHM":      PALETTE.get("ST-IHM",   _C[5]),
        "ST-Decomp":   PALETTE.get("ST-Decomp", _C[6]),
        "ST-Pheno":    PALETTE.get("ST-Pheno",  _C[5]),
        "PRISM":       PALETTE["VFL-MTL"],
    }

    _LOLLIPOP_DEFS = [
        ("IHM AUC-ROC",     "ihm_auc_roc",    ["centralized", "PRISM", "ST-IHM",   "local_A"]),
        ("Decomp AUC-ROC",  "decomp_auc_roc", ["centralized", "PRISM", "ST-Decomp","local_B"]),
        ("Pheno Macro-AUC", "pheno_macro_auc", ["centralized", "PRISM", "ST-Pheno", "local_C"]),
    ]

    fig_lp, axes_lp = plt.subplots(1, 3, figsize=(22, 10))
    fig_lp.suptitle(
        "Held-Out Test Performance: Baseline Comparison",
        fontsize=35, fontweight="normal",
    )

    for ax, (task_label, metric, model_order) in zip(axes_lp, _LOLLIPOP_DEFS):
        lp_models: list[str]   = []
        lp_means:  list[float] = []
        lp_stds:   list[float] = []
        lp_colors: list[str]   = []

        for disp in reversed(model_order):
            vals = _trnodp_thesis[_trnodp_thesis["model_disp"] == disp][metric].dropna()
            if len(vals):
                lp_models.append(disp)
                lp_means.append(float(vals.mean()))
                lp_stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
                lp_colors.append(_PAL_THESIS.get(disp, _C[6]))

        y_pos   = np.arange(len(lp_models))
        n_items = len(lp_models)

        # Axis bounds: zoom into data range (Cleveland plot — zero baseline not needed)
        xlim_left  = max(0.0, min(mu - sd for mu, sd in zip(lp_means, lp_stds)) - 0.05) if lp_means else 0.0
        xlim_right = (max(mu + sd for mu, sd in zip(lp_means, lp_stds)) + 0.09) if lp_means else 1.0

        for yi, (mu, sd, color) in enumerate(zip(lp_means, lp_stds, lp_colors)):
            # Short reference line from left axis edge to the CI lower bound
            ax.hlines(yi, xlim_left, max(xlim_left, mu - sd), color=color,
                      linewidth=2.0, alpha=0.35, zorder=2)
            # Horizontal CI bar with caps
            ax.errorbar(mu, yi, xerr=sd, fmt="none",
                        color=color, elinewidth=3.0, capsize=8, capthick=2.8,
                        alpha=0.90, zorder=3)
            ax.scatter(mu, yi, color=color, s=160, zorder=4, linewidths=0)
            # Label floats just above the dot endpoint
            ax.text(mu, yi + 0.22, f"{mu:.3f}",
                    va="bottom", ha="center", fontsize=31,
                    fontweight="bold", color="#111111")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lp_models, fontsize=28)
        ax.set_xlabel("Test AUC", fontsize=28)
        ax.set_title(task_label, fontsize=32, pad=12, fontweight="normal")
        ax.set_xlim(xlim_left, min(xlim_right, 1.06))
        ax.set_ylim(-0.6, n_items - 0.2)
        ax.tick_params(axis="x", labelsize=26)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add random baseline only if 0.5 falls inside the finalised axis range
        x_lo, x_hi = ax.get_xlim()
        if x_lo <= 0.5 <= x_hi:
            ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.9,
                       label="Random (0.5)")
            ax.legend(fontsize=22, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_lp.savefig(OUT / "baselines_test_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/baselines_test_metrics.png")
    plt.close()
