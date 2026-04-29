"""
figures/plot_supervisor.py  — Supervisor presentation figures.

Figure 1  (supervisor_exp1.png)
  — Exp 1 full: learning curves for all 4 models × 3 tasks + final-metric bar chart.

Figure 2  (supervisor_site_AC.png)
  — Site A: local_A vs ST-IHM vs VFL-MTL  (IHM AUC-ROC)
  — Site C: local_C vs ST-Pheno vs VFL-MTL  (Phenotyping macro-AUC)
  Each site: learning curve (left) + final bar (right).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Palette ───────────────────────────────────────────────────────────────────
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

PAL = {
    "VFL-MTL":   _C[0],
    "ST-IHM":    _C[1],
    "ST-Decomp": _C[2],
    "ST-Pheno":  _C[3],
    "local_A":   _C[1],
    "local_C":   _C[3],
}

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RES = Path("results")

# ── Load ──────────────────────────────────────────────────────────────────────
exp1    = pd.read_csv(RES / "exp1.csv")
local_a = pd.read_csv(RES / "local_only_A.csv")
local_c = pd.read_csv(RES / "local_only_C.csv")

n_rounds = int(exp1["round"].max())
n_seeds  = exp1["seed"].nunique()

print(f"exp1 : {sorted(exp1['model'].unique())}  |  {n_rounds} rounds  |  {n_seeds} seeds")
print(f"local_A epochs: {local_a.groupby('seed')['epoch'].max().to_dict()}")
print(f"local_C epochs: {local_c.groupby('seed')['epoch'].max().to_dict()}")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _curve(ax, df, x_col, y_col, label, color, ls="-"):
    grp  = df.groupby(x_col)[y_col]
    mean = grp.mean()
    std  = grp.std().fillna(0)
    ax.plot(mean.index, mean.values, color=color, lw=2,
            linestyle=ls, label=label)
    ax.fill_between(mean.index, mean - std, mean + std,
                    alpha=0.15, color=color)


def _final(df, y_col, x_col="epoch"):
    last = df.groupby("seed")[y_col].last()
    return float(last.mean()), float(last.std(skipna=True))


def _style(ax, title, xlabel, ylabel, ylim=(0.4, 1.0)):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="Random (0.5)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")


def _draw_bar(ax, entries, title):
    """entries: list of (label, mean, std).  Draws horizontal-labelled bars."""
    labels = [e[0] for e in entries]
    means  = [e[1] for e in entries]
    stds   = [e[2] for e in entries]
    colors = [PAL.get(e[0], _C[5]) for e in entries]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.55, yerr=stds, capsize=5,
                  color=colors, alpha=0.88, edgecolor="white")
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Val AUC (mean ± std, 3 seeds)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, axis="y")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Exp 1 full
# ══════════════════════════════════════════════════════════════════════════════

TASKS_E1 = [
    ("val_ihm_auroc",         "IHM — AUC-ROC",          "Val AUC-ROC"),
    ("val_decomp_auroc",      "Decompensation — AUC-ROC","Val AUC-ROC"),
    ("val_pheno_macro_auroc", "Phenotyping — Macro-AUC", "Val Macro-AUC"),
]

MODELS_E1 = ["VFL-MTL", "ST-IHM", "ST-Decomp", "ST-Pheno"]

fig1 = plt.figure(figsize=(18, 10))
gs   = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

# Row 0 — learning curves (one per task)
for col, (metric, title, ylabel) in enumerate(TASKS_E1):
    ax = fig1.add_subplot(gs[0, col])
    for model in MODELS_E1:
        sub = exp1[exp1["model"] == model]
        if sub.empty:
            continue
        _curve(ax, sub, "round", metric, model, PAL[model])
    _style(ax, title, f"FL Round  (total={n_rounds})", ylabel)

# Row 1 — grouped bar chart: final round × all models × all tasks
ax_bar = fig1.add_subplot(gs[1, :])

bar_rows = []
for metric, title, _ in TASKS_E1:
    task_label = title.split("—")[0].strip()
    for model in MODELS_E1:
        sub = exp1[exp1["model"] == model]
        if sub.empty:
            continue
        mu, sd = _final(sub, metric, x_col="round")
        bar_rows.append({"task": task_label, "model": model,
                         "mean": mu, "std": sd})

bdf      = pd.DataFrame(bar_rows)
tasks_u  = bdf["task"].unique().tolist()
n_t, n_m = len(tasks_u), len(MODELS_E1)
x        = np.arange(n_t)
w        = 0.8 / n_m

for j, model in enumerate(MODELS_E1):
    sub_m  = bdf[bdf["model"] == model]
    means  = [sub_m[sub_m["task"] == t]["mean"].values[0]
              if not sub_m[sub_m["task"] == t].empty else np.nan
              for t in tasks_u]
    stds   = [sub_m[sub_m["task"] == t]["std"].values[0]
              if not sub_m[sub_m["task"] == t].empty else 0.0
              for t in tasks_u]
    offset = (j - n_m / 2 + 0.5) * w
    bars   = ax_bar.bar(x + offset, means, w * 0.9,
                        yerr=stds, capsize=4,
                        color=PAL[model], alpha=0.88,
                        edgecolor="white", label=model)
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.013,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

ax_bar.axhline(0.5, color="grey", ls="--", lw=0.8, label="Random")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(tasks_u, fontsize=11)
ax_bar.set_ylim(0.4, 1.05)
ax_bar.set_ylabel("Val AUC (mean ± std, 3 seeds)")
ax_bar.set_title(f"Final-round performance  (round {n_rounds})",
                 fontweight="bold")
ax_bar.legend(fontsize=9, loc="upper right")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.grid(True, alpha=0.25, axis="y")

fig1.suptitle(
    "Experiment 1 — Task Heterogeneity vs. Single-Task Baselines",
    fontsize=14, fontweight="bold", y=1.01,
)
fig1.savefig(OUT / "supervisor_exp1.png", dpi=150, bbox_inches="tight")
print("Saved: figures/supervisor_exp1.png")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Site A & C comparison
# ══════════════════════════════════════════════════════════════════════════════

st_ihm   = exp1[exp1["model"] == "ST-IHM"]
st_pheno = exp1[exp1["model"] == "ST-Pheno"]
vfl_mtl  = exp1[exp1["model"] == "VFL-MTL"]

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.subplots_adjust(hspace=0.5, wspace=0.35)

# ── Site A: IHM ──────────────────────────────────────────────────────────────
ax_lc_a  = axes2[0, 0]
ax_bar_a = axes2[0, 1]

_curve(ax_lc_a, local_a, "epoch", "val_auc_roc",
       "local_A  (epoch-based)", PAL["local_A"])
_curve(ax_lc_a, st_ihm,  "round", "val_ihm_auroc",
       f"ST-IHM  (FL rounds, n={n_rounds})", PAL["ST-IHM"],  ls="--")
_curve(ax_lc_a, vfl_mtl, "round", "val_ihm_auroc",
       f"VFL-MTL  (FL rounds, n={n_rounds})", PAL["VFL-MTL"], ls="-.")

_style(ax_lc_a,
       "Site A — IHM AUC-ROC\n(local: epochs  |  VFL: rounds)",
       "Epoch / Round", "Val IHM AUC-ROC")

ihm_entries = [
    ("local_A",  *_final(local_a, "val_auc_roc")),
    ("ST-IHM",   *_final(st_ihm,  "val_ihm_auroc",  x_col="round")),
    ("VFL-MTL",  *_final(vfl_mtl, "val_ihm_auroc",  x_col="round")),
]
_draw_bar(ax_bar_a, ihm_entries,
          "Site A — Final IHM AUC-ROC\n(mean ± std, 3 seeds)")

# ── Site C: Phenotyping ───────────────────────────────────────────────────────
ax_lc_c  = axes2[1, 0]
ax_bar_c = axes2[1, 1]

_curve(ax_lc_c, local_c, "epoch", "val_macro_auc",
       "local_C  (epoch-based)", PAL["local_C"])
_curve(ax_lc_c, st_pheno, "round", "val_pheno_macro_auroc",
       f"ST-Pheno  (FL rounds, n={n_rounds})", PAL["ST-Pheno"], ls="--")
_curve(ax_lc_c, vfl_mtl,  "round", "val_pheno_macro_auroc",
       f"VFL-MTL  (FL rounds, n={n_rounds})", PAL["VFL-MTL"],  ls="-.")

_style(ax_lc_c,
       "Site C — Phenotyping Macro-AUC\n(local: epochs  |  VFL: rounds)",
       "Epoch / Round", "Val Phenotyping Macro-AUC")

pheno_entries = [
    ("local_C",   *_final(local_c,   "val_macro_auc")),
    ("ST-Pheno",  *_final(st_pheno,  "val_pheno_macro_auroc", x_col="round")),
    ("VFL-MTL",   *_final(vfl_mtl,   "val_pheno_macro_auroc", x_col="round")),
]
_draw_bar(ax_bar_c, pheno_entries,
          "Site C — Final Phenotyping Macro-AUC\n(mean ± std, 3 seeds)")

fig2.suptitle(
    "Site A & C: local-only vs. single-task VFL (ST) vs. VFL-MTL\n"
    f"Note: VFL models trained for {n_rounds} rounds only (preliminary results)",
    fontsize=13, fontweight="bold", y=1.02,
)
fig2.savefig(OUT / "supervisor_site_AC.png", dpi=150, bbox_inches="tight")
print("Saved: figures/supervisor_site_AC.png")
plt.close(fig2)


# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Site A — IHM (final mean ± std) ──")
for name, mu, sd in ihm_entries:
    print(f"  {name:12s}  {mu:.4f} ± {sd:.4f}")

print("\n── Site C — Phenotyping (final mean ± std) ──")
for name, mu, sd in pheno_entries:
    print(f"  {name:12s}  {mu:.4f} ± {sd:.4f}")
