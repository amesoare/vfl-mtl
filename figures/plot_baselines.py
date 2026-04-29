"""
figures/plot_baselines.py — Baseline comparison: local-only, centralized, VFL-MTL.

Reads (gracefully skips if missing):
  results/local_only_A.csv
  results/local_only_B.csv
  results/local_only_C.csv
  results/centralized.csv
  results/exp1.csv   (VFL-MTL rows only)

Produces:
  figures/baselines_learning_curves.png  — val AUC per epoch/round, mean ± std
  figures/baselines_final_metrics.png    — final-epoch bar chart per task
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
    "local_A":             _C[1],
    "local_B":             _C[2],
    "local_C":             _C[3],
    "centralized_oracle":  _C[4],
    "VFL-MTL":             _C[0],
    "ST-IHM":              _C[5],
    "ST-Decomp":           _C[6],
    "ST-Pheno":            _C[5],
}

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
    (axes[0], local_a, "val_auc_roc",   "val_ihm_auc_roc",    "val_ihm_auroc",          st_ihm,   "ST-IHM",   "In-hospital Mortality — AUC-ROC"),
    (axes[1], local_b, "val_auc_roc",   "val_decomp_auc_roc", "val_decomp_auroc",        st_decomp,"ST-Decomp","Decompensation — AUC-ROC"),
    (axes[2], local_c, "val_macro_auc", "val_pheno_macro_auc","val_pheno_macro_auroc",   st_pheno, "ST-Pheno", "Phenotyping — Macro-AUC"),
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
               label="VFL-MTL", color=PALETTE["VFL-MTL"],
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
        ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUT / "baselines_learning_curves.png", dpi=150, bbox_inches="tight")
print("Saved: figures/baselines_learning_curves.png")
plt.close()


# ── Figure 2: Final-epoch bar chart ──────────────────────────────────────────

bar_rows = []  # (task_label, model_label, mean, std, color)

task_defs = [
    ("IHM\nAUC-ROC",     local_a, "val_auc_roc",   central, "val_ihm_auc_roc",     st_ihm,   "val_ihm_auroc",          vfl_mtl, "val_ihm_auroc",          "local_A", "ST-IHM"),
    ("Decomp\nAUC-ROC",  local_b, "val_auc_roc",   central, "val_decomp_auc_roc",  st_decomp,"val_decomp_auroc",        vfl_mtl, "val_decomp_auroc",        "local_B", "ST-Decomp"),
    ("Pheno\nMacro-AUC", local_c, "val_macro_auc", central, "val_pheno_macro_auc", st_pheno, "val_pheno_macro_auroc",   vfl_mtl, "val_pheno_macro_auroc",   "local_C", "ST-Pheno"),
]

for task_label, loc_df, loc_m, cen_df, cen_m, st_df, st_m, vfl_df, vfl_m, loc_key, st_key in task_defs:
    if loc_df is not None and loc_m in loc_df.columns:
        mu, sd = _final_stats(loc_df, loc_m)
        bar_rows.append((task_label, f"local\n({loc_key[-1]})", mu, sd, PALETTE[loc_key]))
    if cen_df is not None and cen_m in cen_df.columns:
        mu, sd = _final_stats(cen_df, cen_m)
        bar_rows.append((task_label, "centralized", mu, sd, PALETTE["centralized_oracle"]))
    if st_df is not None and st_m in st_df.columns:
        mu, sd = _final_stats(st_df, st_m, epoch_col="round")
        bar_rows.append((task_label, st_key, mu, sd, PALETTE[st_key]))
    if vfl_df is not None and vfl_m in vfl_df.columns:
        mu, sd = _final_stats(vfl_df, vfl_m, epoch_col="round")
        bar_rows.append((task_label, "VFL-MTL", mu, sd, PALETTE["VFL-MTL"]))

if bar_rows:
    _order  = ["IHM", "Decomp", "Pheno"]
    tasks   = sorted(set(r[0] for r in bar_rows),
                     key=lambda t: next((i for i, o in enumerate(_order) if o in t), 99))
    models  = list(dict.fromkeys(r[1] for r in bar_rows))
    n_tasks = len(tasks)
    n_models= len(models)
    x       = np.arange(n_tasks)
    width   = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, model in enumerate(models):
        model_rows = {r[0]: r for r in bar_rows if r[1] == model}
        means = [model_rows[t][2] if t in model_rows else np.nan for t in tasks]
        stds  = [model_rows[t][3] if t in model_rows else 0.0     for t in tasks]
        color = model_rows[list(model_rows.keys())[0]][4] if model_rows else _C[6]
        offset = (j - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds, capsize=4,
                      color=color, alpha=0.88, edgecolor="white", label=model)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="normal")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Val AUC (mean ± std, 3 seeds)")
    ax.set_title("Final epoch performance per task across model configurations", fontweight="normal")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT / "baselines_final_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/baselines_final_metrics.png")
    plt.close()
else:
    print("  [SKIP] No data available for bar chart")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Final metrics (mean ± std, 3 seeds) ──")
for task, model, mu, sd, _ in bar_rows:
    print(f"  {task.replace(chr(10),' '):20s}  {model:20s}  {mu:.4f} ± {sd:.4f}")
