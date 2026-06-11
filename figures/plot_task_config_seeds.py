"""
figures/plot_mimic_task_relatedness_seeds.py

Enhanced task-configuration figure (Paper 1, Exp 2):
  • Per-seed points shown so the single-run Decomp collapse is visible
  • Decompensation panel annotated to highlight 0.713 → 0.632 drop + wide std
  • Small scalability inset (2 vs 3 institutions, slopegraph) from test_exp3.csv

Output: figures/mimic_task_relatedness_seeds.png

Usage:
    python figures/plot_mimic_task_relatedness_seeds.py \
        --results_dir results/ --output figures/mimic_task_relatedness_seeds.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# ── Brand palette — identical to all other figures in this repo ────────────
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        16,
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize":   20,
    "axes.titleweight": "normal",
    "axes.labelsize":   17,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  13,
    "axes.linewidth":   0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

CONFIG_ORDER = ["all_tasks", "ihm_only", "ihm_decomp", "ihm_pheno"]
CONFIG_LABELS = {
    "all_tasks":  "All tasks",
    "ihm_only":   "IHM only",
    "ihm_decomp": "IHM+Decomp",
    "ihm_pheno":  "IHM+Pheno",
}
_TASK_COLOR = {
    "ihm_auc_roc":     _C[1],   # purple
    "decomp_auc_roc":  _C[2],   # dark charcoal
    "pheno_macro_auc": _C[3],   # burgundy
}
_INSET_METRICS = [
    ("ihm_auc_roc",     "IHM",    _C[1]),
    ("decomp_auc_roc",  "Decomp", _C[2]),
    ("pheno_macro_auc", "Pheno",  _C[3]),
]
FLOORS = {"ihm": 0.75, "decomp": 0.70, "pheno": 0.65}
JITTER = [-0.12, 0.0, 0.12]


# ── Inset: scalability slopegraph ─────────────────────────────────────────
def _draw_inset(ax_parent, test3_df: pd.DataFrame) -> None:
    inset = ax_parent.inset_axes([0.44, 0.50, 0.54, 0.48])

    xs = sorted(test3_df["n_sites"].unique())
    x_left, x_right = 0, 1

    for metric, label, color in _INSET_METRICS:
        g   = test3_df.groupby("n_sites")[metric]
        mu  = g.mean().reindex(xs)
        sd  = g.std().fillna(0).reindex(xs)
        pts = [(xi, float(mu[x]), float(sd[x]))
               for xi, x in zip([x_left, x_right], xs)
               if not np.isnan(mu[x])]

        if len(pts) == 2:
            inset.fill_between(
                [p[0] for p in pts],
                [p[1] - p[2] for p in pts],
                [p[1] + p[2] for p in pts],
                color=color, alpha=0.12,
            )
            inset.plot([p[0] for p in pts], [p[1] for p in pts],
                       color=color, linewidth=1.6)

        for xi, m, _ in pts:
            inset.scatter(xi, m, color=color, s=28, zorder=4)

        if pts:
            last = pts[-1]
            inset.text(x_right + 0.06, last[1], f"{last[1]:.3f}",
                       ha="left", va="center", fontsize=7.5, color=color)

    inset.set_xticks([x_left, x_right])
    inset.set_xticklabels(["2 sites", "3 sites"], fontsize=8)
    inset.set_xlim(-0.55, 1.70)
    inset.set_ylabel("AUC", fontsize=8, labelpad=2)
    inset.set_title("Scalability", fontsize=9, pad=3)
    inset.grid(True, axis="y", alpha=0.2)
    for spine in ("top", "right", "bottom"):
        inset.spines[spine].set_visible(False)
    inset.tick_params(bottom=False)
    inset.tick_params(axis="y", labelsize=7.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#aaaaaa")


# ── Per-task panel ─────────────────────────────────────────────────────────
def _draw_panel(ax, metric: str, title: str, configs: list[str],
                ref_cfg: str, floor: float, test2: pd.DataFrame) -> None:
    color   = _TASK_COLOR[metric]
    present = [c for c in configs if c in test2["task_config"].values]
    x_pos   = np.arange(len(present))

    # Per-seed dots
    for xi, cfg in enumerate(present):
        vals = test2[test2["task_config"] == cfg][metric].dropna().values
        for vi, v in enumerate(vals):
            jx = JITTER[vi] if vi < len(JITTER) else 0.0
            ax.scatter(xi + jx, v, color=color, s=75, zorder=3,
                       linewidths=0.6, edgecolors="white", alpha=0.72)

    # Mean ± std
    means, stds = [], []
    for cfg in present:
        vals = test2[test2["task_config"] == cfg][metric].dropna()
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    ax.errorbar(x_pos, means, yerr=stds, fmt="D", color=color,
                capsize=7, capthick=1.8, elinewidth=1.8,
                markersize=10, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)

    # Big bold mean labels
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.012, f"{m:.3f}",
                ha="center", va="bottom",
                fontsize=17, fontweight="bold", color="#1a1a1a")

    # Reference dashed line
    if ref_cfg in present:
        ref_mean = means[present.index(ref_cfg)]
        ax.axhline(ref_mean, color=color, lw=1.0, ls="--", alpha=0.45,
                   label=f"ref: {CONFIG_LABELS.get(ref_cfg, ref_cfg)}")

    # Clinical floor
    ax.axhline(floor, color="#888888", lw=0.9, ls=":",
               label=f"floor {floor:.2f}")

    # Axis limits
    all_vals = test2[test2["task_config"].isin(present)][metric].dropna()
    lo = max(0.0, float(all_vals.min()) - 0.07)
    hi = float(all_vals.max()) + 0.10
    ax.set_ylim(min(lo, floor - 0.05), max(hi, floor + 0.06))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in present],
                       rotation=12, ha="right", fontsize=15)
    ax.set_ylabel(title, fontsize=17)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=12, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# ── Decomp annotation: highlight collapse ─────────────────────────────────
def _annotate_decomp(ax, test2: pd.DataFrame) -> None:
    configs  = ["ihm_decomp", "all_tasks"]
    present  = [c for c in configs if c in test2["task_config"].values]
    if len(present) < 2:
        return

    xi_ref = 0  # ihm_decomp
    xi_all = 1  # all_tasks

    vals_ref = test2[test2["task_config"] == "ihm_decomp"]["decomp_auc_roc"].dropna()
    vals_all = test2[test2["task_config"] == "all_tasks"]["decomp_auc_roc"].dropna()
    m_ref = float(vals_ref.mean())
    m_all = float(vals_all.mean())
    s_all = float(vals_all.std(ddof=1)) if len(vals_all) > 1 else 0.0

    # Curved red arrow: ihm_decomp mean → all_tasks mean
    ax.annotate(
        "",
        xy=(xi_all - 0.08, m_all + 0.005),
        xytext=(xi_ref + 0.08, m_ref - 0.005),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#cc2222",
            lw=2.0,
            connectionstyle="arc3,rad=-0.30",
        ),
        zorder=7,
    )

    # Delta label midway between the two means
    ax.text(
        0.50, (m_ref + m_all) / 2 + 0.025,
        f"Δ = {m_all - m_ref:+.3f}",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", color="#cc2222",
        transform=ax.transData,
        zorder=8,
    )

    # Callout for the wide std
    ax.annotate(
        f"std = {s_all:.3f}",
        xy=(xi_all, m_all - s_all + 0.005),
        xytext=(xi_all + 0.30, m_all - s_all - 0.040),
        ha="left", va="top",
        fontsize=13, color="#cc2222",
        arrowprops=dict(arrowstyle="-", color="#cc2222", lw=1.2),
        zorder=8,
    )

    # Identify and label the collapsed seed
    min_val  = float(vals_all.min())
    min_seed = int(test2.loc[test2["task_config"] == "all_tasks",
                             ["seed", "decomp_auc_roc"]]
                          .dropna()
                          .sort_values("decomp_auc_roc")
                          .iloc[0]["seed"])
    ax.annotate(
        f"seed {min_seed}: {min_val:.3f}",
        xy=(xi_all - 0.12, min_val),
        xytext=(xi_all - 0.50, min_val - 0.030),
        ha="right", va="top",
        fontsize=12, color="#cc2222",
        arrowprops=dict(arrowstyle="-", color="#cc2222", lw=1.0),
        zorder=8,
    )


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output",      default="figures/mimic_task_relatedness_seeds.png")
    args = parser.parse_args()

    rd     = Path(args.results_dir)
    test2  = pd.read_csv(rd / "test_exp2.csv")
    test3  = pd.read_csv(rd / "test_exp3.csv")

    PANELS = [
        ("ihm_auc_roc",
         "IHM AUC-ROC",
         ["ihm_only", "ihm_decomp", "ihm_pheno", "all_tasks"],
         "ihm_only",   FLOORS["ihm"]),
        ("decomp_auc_roc",
         "Decomp AUC-ROC",
         ["ihm_decomp", "all_tasks"],
         "ihm_decomp", FLOORS["decomp"]),
        ("pheno_macro_auc",
         "Pheno Macro-AUC",
         ["ihm_pheno", "all_tasks"],
         "ihm_pheno",  FLOORS["pheno"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
    fig.suptitle(
        "Test-set AUC by task configuration  (per-seed · mean ± std · 3 seeds)",
        fontsize=18, fontweight="normal", y=1.02,
    )

    for ax, (metric, title, configs, ref_cfg, floor) in zip(axes, PANELS):
        _draw_panel(ax, metric, title, configs, ref_cfg, floor, test2)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── slide7: standalone scalability slopegraph ─────────────────────────
    _plot_slide7(test3, out.parent / "slide7.png")


def _plot_slide7(test3_df: pd.DataFrame, output: Path) -> None:
    xs = sorted(test3_df["n_sites"].unique())
    x_left, x_right = 0, 1

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig.suptitle("Scalability: 2 vs 3 institutions",
                 fontsize=18, fontweight="normal", y=1.01)

    all_pts = []
    for metric, label, color in _INSET_METRICS:
        g  = test3_df.groupby("n_sites")[metric]
        mu = g.mean().reindex(xs)
        sd = g.std().fillna(0).reindex(xs)

        pts = [(xi, float(mu[x]), float(sd[x]))
               for xi, x in zip([x_left, x_right], xs)
               if not np.isnan(mu[x])]
        all_pts.extend(m for _, m, _ in pts)

        if len(pts) == 2:
            ax.fill_between(
                [p[0] for p in pts],
                [p[1] - p[2] for p in pts],
                [p[1] + p[2] for p in pts],
                color=color, alpha=0.15,
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linewidth=2.2, zorder=2)

        # Per-seed dots
        for xi, x in zip([x_left, x_right], xs):
            seeds = test3_df[test3_df["n_sites"] == x][metric].dropna().values
            n = len(seeds)
            jit = np.linspace(-0.06, 0.06, n) if n > 1 else [0.0]
            for jx, v in zip(jit, seeds):
                ax.scatter(xi + jx, v, color=color, s=55, alpha=0.65,
                           linewidths=0.5, edgecolors="white", zorder=3)

        for xi, m, _ in pts:
            ax.scatter(xi, m, color=color, s=90, zorder=5,
                       linewidths=0.8, edgecolors="white")

        # Labels
        if pts and pts[0][0] == x_left:
            ax.text(x_left - 0.06, pts[0][1], f"{label}  {pts[0][1]:.3f}",
                    ha="right", va="center", fontsize=14,
                    fontweight="bold", color=color)
        if pts:
            ax.text(x_right + 0.06, pts[-1][1], f"{pts[-1][1]:.3f}  {label}",
                    ha="left", va="center", fontsize=14,
                    fontweight="bold", color=color)

        floor_val = {"ihm_auc_roc": FLOORS["ihm"],
                     "decomp_auc_roc": FLOORS["decomp"],
                     "pheno_macro_auc": FLOORS["pheno"]}[metric]
        ax.axhline(floor_val, color=color, lw=0.8, ls=":", alpha=0.45, zorder=1)

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["2 institutions", "3 institutions"], fontsize=15)
    ax.set_xlim(-1.0, 1.85)
    ax.set_ylabel("AUC-ROC", fontsize=16)
    if all_pts:
        margin = 0.06
        ax.set_ylim(min(all_pts) - margin, max(all_pts) + margin)
    ax.grid(True, axis="y", alpha=0.2)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
