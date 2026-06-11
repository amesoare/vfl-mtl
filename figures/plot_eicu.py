"""
figures/plot_eicu.py — eICU figures: baseline comparison + privacy-utility curves.

Reads:
  results/eicu_local_only_{A,B,C}.csv
  results/eicu_centralized.csv
  results/eicu_exp1.csv
  results/eicu_test_results.csv
  results/eicu_privacy_utility_combined.csv

Produces:
  figures/eicu_baselines_val_metrics.png
  figures/eicu_test_baselines_metrics.png
  figures/eicu_privacy_utility.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

PALETTE = {
    "local_A":     "#c5a77d",
    "local_B":     "#663139",
    "local_C":     _C[5],
    "centralized": _C[4],
    "PRISM":       _C[0],
    "ST-IHM":      _C[1],
    "ST-RLOS":     _C[2],
    "ST-Pheno":    _C[3],
}

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        15,
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":   21,
    "axes.titleweight": "normal",
    "axes.labelsize":   18,
    "xtick.labelsize":  19,
    "ytick.labelsize":  19,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RES = Path("results")


def _load(path):
    if path.exists():
        return pd.read_csv(path)
    print(f"  [SKIP] {path} not found")
    return None


def _final_stats(df, metric, epoch_col="epoch"):
    last = df.groupby("seed")[metric].last()
    return float(last.mean()), float(last.std(skipna=True))


# ---------------------------------------------------------------------------
# Load baselines data
# ---------------------------------------------------------------------------

local_a = _load(RES / "eicu_local_only_A.csv")
local_b = _load(RES / "eicu_local_only_B.csv")
local_c = _load(RES / "eicu_local_only_C.csv")
central = _load(RES / "eicu_centralized.csv")
exp1    = _load(RES / "eicu_exp1.csv")

vfl_mtl = st_ihm = st_rlos = st_pheno = None
if exp1 is not None:
    def _sub(m):
        s = exp1[exp1["model"] == m]
        return s if not s.empty else None
    vfl_mtl  = _sub("eICU_VFL-MTL")
    st_ihm   = _sub("eICU_ST-IHM")
    st_rlos  = _sub("eICU_ST-RLOS")
    st_pheno = _sub("eICU_ST-Pheno")

# ---------------------------------------------------------------------------
# Panel definitions
# invert=True → lower is better (MAE); hlines anchor to right (better) side
# ---------------------------------------------------------------------------

PANELS = [
    {
        "title":  "IHM AUC-ROC",
        "xlabel": "Val AUC",
        "invert": False,
        "data": [
            ("centralized", central,  "val_ihm_auc_roc",       "epoch"),
            ("PRISM",       vfl_mtl,  "val_ihm_auroc",         "round"),
            ("ST-IHM",      st_ihm,   "val_ihm_auroc",         "round"),
            ("local_A",     local_a,  "val_auc_roc",           "epoch"),
        ],
    },
    {
        "title":  "RLOS MAE",
        "xlabel": "Val MAE  (← lower is better)",
        "invert": True,
        "data": [
            ("centralized", central, "val_rlos_mae", "epoch"),
            ("PRISM",       vfl_mtl, "val_rlos_mae", "round"),
            ("ST-RLOS",     st_rlos, "val_rlos_mae", "round"),
            ("local_B",     local_b, "val_mae",      "epoch"),
        ],
    },
    {
        "title":  "Pheno Macro-AUC",
        "xlabel": "Val AUC",
        "invert": False,
        "data": [
            ("centralized", central,  "val_pheno_macro_auc",   "epoch"),
            ("PRISM",       vfl_mtl,  "val_pheno_macro_auroc", "round"),
            ("ST-Pheno",    st_pheno, "val_pheno_macro_auroc", "round"),
            ("local_C",     local_c,  "val_macro_auc",         "epoch"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Validation lollipop
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(22, 10))
fig.suptitle("Validation Performance: eICU Baseline Comparison",
             fontsize=35, fontweight="normal")

for ax, panel in zip(axes, PANELS):
    lp_models, lp_means, lp_stds, lp_colors = [], [], [], []

    for model_name, df, metric, ecol in reversed(panel["data"]):
        if df is None or metric not in df.columns:
            continue
        mu, sd = _final_stats(df, metric, epoch_col=ecol)
        lp_models.append(model_name)
        lp_means.append(mu)
        lp_stds.append(sd)
        lp_colors.append(PALETTE.get(model_name, _C[6]))

    if not lp_models:
        continue

    y_pos   = np.arange(len(lp_models))
    n_items = len(lp_models)
    buf     = 0.05

    xlim_lo = max(0.0, min(m - s for m, s in zip(lp_means, lp_stds)) - buf)
    xlim_hi = max(m + s for m, s in zip(lp_means, lp_stds)) + buf + 0.05

    for yi, (mu, sd, color) in enumerate(zip(lp_means, lp_stds, lp_colors)):
        if panel["invert"]:
            ax.hlines(yi, mu + sd, xlim_hi, color=color,
                      linewidth=2.0, alpha=0.35, zorder=2)
        else:
            ax.hlines(yi, xlim_lo, max(xlim_lo, mu - sd), color=color,
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
    ax.set_xlabel(panel["xlabel"], fontsize=28)
    ax.set_title(panel["title"], fontsize=32, pad=10)
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(-0.6, n_items - 0.2)
    ax.tick_params(axis="x", labelsize=26)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if panel["invert"]:
        ax.invert_xaxis()

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "eicu_baselines_val_metrics.png", dpi=150, bbox_inches="tight")
print("Saved: figures/eicu_baselines_val_metrics.png")
plt.close()

# ---------------------------------------------------------------------------
# Test metrics lollipop  (eicu_test_results.csv)
# ---------------------------------------------------------------------------

_DISP = {
    "eICU_VFL-MTL":      "PRISM",
    "eICU_ST-IHM":       "ST-IHM",
    "eICU_ST-RLOS":      "ST-RLOS",
    "eICU_ST-Pheno":     "ST-Pheno",
    "local_A":           "local_A",
    "local_B":           "local_B",
    "local_C":           "local_C",
    "centralized_oracle":"centralized",
}

test_df = _load(RES / "eicu_test_results.csv")

if test_df is not None:
    test_df["model_disp"] = test_df["model"].map(lambda m: _DISP.get(m, m))

    def _test_stats(disp, metric):
        vals = test_df[test_df["model_disp"] == disp][metric].dropna()
        if vals.empty:
            return None, None
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    TEST_PANELS = [
        {
            "title":       "IHM AUC-ROC",
            "xlabel":      "Test AUC",
            "metric":      "ihm_auc_roc",
            "invert":      False,
            "model_order": ["centralized", "PRISM", "ST-IHM", "local_A"],
        },
        {
            "title":       "RLOS MAE",
            "xlabel":      "Test MAE  (← lower is better)",
            "metric":      "rlos_mae",
            "invert":      True,
            "model_order": ["centralized", "PRISM", "ST-RLOS", "local_B"],
        },
        {
            "title":       "Pheno Macro-AUC",
            "xlabel":      "Test AUC",
            "metric":      "pheno_macro_auc",
            "invert":      False,
            "model_order": ["centralized", "PRISM", "ST-Pheno", "local_C"],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle("Held-Out Test Performance: eICU Baseline Comparison",
                 fontsize=35, fontweight="normal")

    for ax, panel in zip(axes, TEST_PANELS):
        lp_models, lp_means, lp_stds, lp_colors = [], [], [], []

        for disp in reversed(panel["model_order"]):
            mu, sd = _test_stats(disp, panel["metric"])
            if mu is None:
                continue
            lp_models.append(disp)
            lp_means.append(mu)
            lp_stds.append(sd)
            lp_colors.append(PALETTE.get(disp, _C[6]))

        if not lp_models:
            continue

        y_pos   = np.arange(len(lp_models))
        n_items = len(lp_models)
        buf     = 0.05

        xlim_lo = max(0.0, min(m - s for m, s in zip(lp_means, lp_stds)) - buf)
        xlim_hi = max(m + s for m, s in zip(lp_means, lp_stds)) + buf + 0.05

        for yi, (mu, sd, color) in enumerate(zip(lp_means, lp_stds, lp_colors)):
            if panel["invert"]:
                ax.hlines(yi, mu + sd, xlim_hi, color=color,
                          linewidth=2.0, alpha=0.35, zorder=2)
            else:
                ax.hlines(yi, xlim_lo, max(xlim_lo, mu - sd), color=color,
                          linewidth=2.0, alpha=0.35, zorder=2)
            ax.errorbar(mu, yi, xerr=sd, fmt="none",
                        color=color, elinewidth=3.0, capsize=8, capthick=2.8,
                        alpha=0.90, zorder=3)
            ax.scatter(mu, yi, color=color, s=130, zorder=4, linewidths=0)
            ax.text(mu, yi + 0.22, f"{mu:.3f}",
                    va="bottom", ha="center", fontsize=20,
                    fontweight="bold", color="#111111")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lp_models, fontsize=19)
        ax.set_xlabel(panel["xlabel"], fontsize=18)
        ax.set_title(panel["title"], fontsize=21, pad=10)
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(-0.6, n_items - 0.2)
        ax.tick_params(axis="x", labelsize=26)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if panel["invert"]:
            ax.invert_xaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "eicu_test_baselines_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/eicu_test_baselines_metrics.png")
    plt.close()
else:
    print("  [SKIP] eicu_test_results.csv not found")


# ---------------------------------------------------------------------------
# Privacy-utility curves  (eicu_privacy_utility_combined.csv)
# Matches privacy_utility_test.png format exactly.
# Panel 2 is RLOS MAE (regression, lower=better) — no floor/ε* for that task.
# ---------------------------------------------------------------------------

EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", "∞"]

MODE_STYLE = {
    "uniform":    {"color": _C[4], "ls": "-",  "label": "Uniform σ"},
    "stratified": {"color": _C[3], "ls": "--", "label": "Stratified σ"},
}

plt.rcParams.update({
    "axes.titlesize":  17,
    "axes.labelsize":  15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})


def _draw_priv_panels(axes, df, panels, legend_locs=None, hide_legend=None, floor_in_legend=True):
    """Draw privacy-utility panels into axes. df: one row per (mode, epsilon_level, seed)."""
    if legend_locs is None:
        legend_locs = {}
    if hide_legend is None:
        hide_legend = set()
    modes_present = df["mode"].unique().tolist()
    for ax, (task_name, task_col, ylabel, floor, invert_y) in zip(axes, panels):
        if task_col not in df.columns:
            ax.set_title(task_name)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        grp   = df.groupby(["mode", "epsilon_level"])[task_col]
        means = grp.mean()
        stds  = grp.std()

        for mode in ["uniform", "stratified"]:
            if mode not in modes_present:
                continue
            style = MODE_STYLE[mode]
            eps_vals, mu_vals, sd_vals = [], [], []
            for eps in EPS_ORDER:
                try:
                    mu = means.loc[(mode, eps)]
                    sd = stds.loc[(mode, eps)]
                    if np.isnan(mu):
                        continue
                    eps_vals.append(eps)
                    mu_vals.append(mu)
                    sd_vals.append(sd if not np.isnan(sd) else 0.0)
                except KeyError:
                    continue

            x_plot = [EPS_ORDER.index(e) for e in eps_vals]
            mu_arr = np.array(mu_vals)
            sd_arr = np.array(sd_vals)
            ax.plot(x_plot, mu_arr, color=style["color"], ls=style["ls"],
                    marker="o", ms=4, linewidth=1.4, label=style["label"])
            ax.fill_between(x_plot, mu_arr - sd_arr, mu_arr + sd_arr,
                            color=style["color"], alpha=0.15)

            if floor is not None:
                try:
                    sub = means.xs(mode, level="mode")
                    for eps in EPS_ORDER:
                        if eps in sub.index and sub[eps] >= floor:
                            x_star = EPS_ORDER.index(eps)
                            ax.axvline(x_star, color=style["color"],
                                       ls=":", alpha=0.6, linewidth=1.0)
                            ax.text(x_star, 1.02, f"ε*={EPS_LABELS[x_star]}",
                                    fontsize=10, color=style["color"],
                                    ha="center", va="top",
                                    bbox=dict(boxstyle="round,pad=0.15",
                                              fc="white", ec="none", alpha=0.85))
                            break
                except KeyError:
                    pass

        if floor is not None:
            if floor_in_legend:
                ax.axhline(floor, color="#888888", linestyle="--", linewidth=0.8,
                           label=f"Floor ({floor})")
            else:
                ax.axhline(floor, color="#888888", linestyle="--", linewidth=0.8)
                ax.text(0.02, floor + 0.015, f"floor={floor:.2f}",
                        ha="left", va="bottom", fontsize=10, color="#888888",
                        transform=ax.get_yaxis_transform())

        ax.set_xticks(range(len(EPS_ORDER)))
        ax.set_xticklabels(EPS_LABELS)
        ax.set_xlabel("Privacy budget ε")
        ax.set_ylabel(ylabel)
        ax.set_title(task_name)
        if not invert_y:
            ax.set_ylim(0.3, 1.05)
        if task_name not in hide_legend:
            default_loc = "lower right" if not invert_y else "upper right"
            loc = legend_locs.get(task_name, default_loc)
            ax.legend(fontsize=10, loc=loc)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Privacy-utility — validation
# ---------------------------------------------------------------------------

VAL_PANELS = [
    ("IHM",   "val_ihm_auroc",         "Mean AUC-ROC", 0.75, False),
    ("RLOS",  "val_rlos_mae",          "Mean RLOS MAE", None, True),
    ("Pheno", "val_pheno_macro_auroc", "Mean AUC-ROC", 0.65, False),
]

priv_df = _load(RES / "eicu_privacy_utility_combined.csv")

if priv_df is not None:
    priv_df["epsilon_level"] = pd.to_numeric(
        priv_df["epsilon_level"], errors="coerce").fillna(float("inf"))
    priv_last = priv_df.groupby(["mode", "epsilon_level", "seed"]).last().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        "Effect of Differential Privacy on Validation Performance (eICU)",
        fontweight="normal")
    _draw_priv_panels(axes, priv_last, VAL_PANELS,
                      hide_legend={"RLOS", "Pheno"}, floor_in_legend=False)
    fig.tight_layout()
    fig.savefig(OUT / "eicu_privacy_utility_val.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/eicu_privacy_utility_val.png")
    plt.close()
else:
    print("  [SKIP] eicu_privacy_utility_combined.csv not found")

# ---------------------------------------------------------------------------
# Privacy-utility — test
# ---------------------------------------------------------------------------

TEST_PANELS = [
    ("IHM",   "ihm_auc_roc",    "AUC-ROC", 0.75, False),
    ("RLOS",  "rlos_mae",       "RLOS MAE", None, True),
    ("Pheno", "pheno_macro_auc","AUC-ROC", 0.65, False),
]

test_dp_df = _load(RES / "eicu_test_dp.csv")

if test_dp_df is not None:
    test_dp_df["epsilon_level"] = pd.to_numeric(
        test_dp_df["epsilon_level"], errors="coerce").fillna(float("inf"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        "Effect of Differential Privacy on Held-Out Test Performance (eICU)",
        fontweight="normal")
    _draw_priv_panels(axes, test_dp_df, TEST_PANELS,
                      legend_locs={"Pheno": "upper right"})
    fig.tight_layout()
    fig.savefig(OUT / "eicu_privacy_utility_test.png", dpi=150, bbox_inches="tight")
    print("Saved: figures/eicu_privacy_utility_test.png")
    plt.close()
else:
    print("  [SKIP] eicu_test_dp.csv not found")
