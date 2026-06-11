"""
figures/scalability_curves.py

Three-panel figure: AUC slopegraph + convergence rounds + training time.

Sources:
  test_exp3.csv -- held-out test AUC metrics (n_sites = 2 and 3)
  exp3.csv      -- training-loop CSV (for convergence rounds and wall-clock)

Usage:
    python figures/scalability_curves.py \
        --exp3 results/exp3.csv \
        --test_exp3 results/test_exp3.csv \
        --output figures/scalability.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Brand palette — matches resilience_variance.py and plot_ablations_dp.py
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
})

_METRICS = [
    ("ihm_auc_roc",     "IHM",    _C[1], 0.75),   # purple
    ("decomp_auc_roc",  "Decomp", _C[2], 0.70),   # dark charcoal
    ("pheno_macro_auc", "Pheno",  _C[3], 0.65),   # burgundy
]

# Convergence round stats from exp3 (seeds 42, 123, 7): (mean, std)
_CONV_ROUNDS = {2: (60.3, 31.1), 3: (70.0, 10.01)}
# Per-round wall-clock seconds from exp4 Snellius timing
_SEC_PER_ROUND = {2: 4.18, 3: 5.54}


def _spread_labels(annots, min_gap=0.07):
    """Push label y-positions apart until no two are closer than min_gap."""
    if len(annots) < 2:
        return annots
    ys = [a[0] for a in annots]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    for _ in range(200):
        changed = False
        for k in range(len(order) - 1):
            i, j = order[k], order[k + 1]
            if ys[j] - ys[i] < min_gap:
                mid = (ys[i] + ys[j]) / 2
                ys[i] = mid - min_gap / 2
                ys[j] = mid + min_gap / 2
                changed = True
        if not changed:
            break
    return [(ys[k],) + annots[k][1:] for k in range(len(annots))]


def _plot_slopegraph(ax, test_df, fs_annot=12, fs_ticks=14, fs_ylabel=14,
                     fs_title=15, fs_yticks=14, annot_bold=False):
    xs = sorted(test_df["n_sites"].unique())
    x_left, x_right = 0, 1
    slope_xs = [x_left, x_right]
    all_auc = []
    all_auc_with_sd = []

    left_annots  = []  # (y, text, color)
    right_annots = []

    for metric, label, color, floor in _METRICS:
        g = test_df.groupby("n_sites")[metric]
        mu = g.mean().reindex(xs)
        sd = g.std().fillna(0).reindex(xs)

        pts = [(xi, float(mu[x]), float(sd[x]))
               for xi, x in zip(slope_xs, xs) if not np.isnan(mu[x])]
        all_auc.extend(m for _, m, _ in pts)
        all_auc_with_sd.extend(m - s for _, m, s in pts)
        all_auc_with_sd.extend(m + s for _, m, s in pts)

        if len(pts) == 2:
            ax.fill_between(
                [p[0] for p in pts],
                [p[1] - p[2] for p in pts],
                [p[1] + p[2] for p in pts],
                color=color, alpha=0.12, zorder=1,
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, linewidth=1.8, zorder=2)

        for xi, m, s in pts:
            ax.scatter(xi, m, color=color, s=60, zorder=4)

        if pts and pts[0][0] == x_left:
            left_annots.append((pts[0][1], f"{label}\n{pts[0][1]:.3f}", color))
        if pts:
            right_annots.append((pts[-1][1], f"{pts[-1][1]:.3f}\n{label}", color))

        ax.axhline(floor, color=color, linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)

    fw = "bold" if annot_bold else "normal"
    for y, text, color in _spread_labels(left_annots):
        ax.text(x_left - 0.04, y, text,
                ha="right", va="center", fontsize=fs_annot,
                fontweight=fw, color=color)
    for y, text, color in _spread_labels(right_annots):
        ax.text(x_right + 0.04, y, text,
                ha="left", va="center", fontsize=fs_annot,
                fontweight=fw, color=color)

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["2 institutions", "3 institutions"], fontsize=fs_ticks)
    ax.set_xlim(-1.15, 1.85)
    ax.set_ylabel("AUC-ROC", fontsize=fs_ylabel)
    ax.set_title("Per-task test AUC-ROC", fontsize=fs_title, pad=6)
    ax.tick_params(axis="y", labelsize=fs_yticks)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(bottom=False)
    if all_auc_with_sd:
        ax.set_ylim(min(all_auc_with_sd) - 0.06, max(all_auc_with_sd) + 0.06)


def _plot_cost(ax):
    ns = [2, 3]
    mu_r  = [_CONV_ROUNDS[n][0] for n in ns]
    sd_r  = [_CONV_ROUNDS[n][1] for n in ns]
    mu_wc = [_CONV_ROUNDS[n][0] * _SEC_PER_ROUND[n] / 60 for n in ns]
    sd_wc = [_CONV_ROUNDS[n][1] * _SEC_PER_ROUND[n] / 60 for n in ns]

    bar_color  = "#b0b8c8"
    line_color = "#3a3a3a"
    xs = [0, 1]

    ax.bar(xs, mu_r, yerr=sd_r, color=bar_color, width=0.45,
           capsize=5, zorder=2, label="Rounds", error_kw={"linewidth": 1.2})
    ax.set_ylabel("Convergence rounds", color="#5a6a80")
    ax.tick_params(axis="y", labelcolor="#5a6a80")
    ax.set_ylim(0, max(mu_r) + max(sd_r) + 22)

    ax2 = ax.twinx()
    ax2.plot(xs, mu_wc, color=line_color, linewidth=2, marker="o",
             markersize=7, zorder=3, label="Wall-clock (min)")
    ax2.errorbar(xs, mu_wc, yerr=sd_wc, fmt="none", color=line_color,
                 capsize=5, linewidth=1.2, zorder=3)
    ax2.set_ylabel("Total training time (min)", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, max(mu_wc) + max(sd_wc) + 2.5)

    # "rds" centred above the error bar cap; "min" to the right of the dot
    for xi, r, sr, wc in zip(xs, mu_r, sd_r, mu_wc):
        ax.text(xi, r + sr + 4, f"{r:.0f} rds",
                ha="center", va="bottom", fontsize=12, color="#1a2a3a")
        ax2.text(xi + 0.18, wc, f"{wc:.1f} min",
                 ha="left", va="center", fontsize=12, color=line_color)

    ax.set_xticks(xs)
    ax.set_xticklabels(["2 institutions", "3 institutions"], fontsize=14)
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("(b) Training cost", pad=6)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=12, framealpha=0.8)


def _plot_cost_strip(ax_rounds: plt.Axes, ax_time: plt.Axes,
                    exp3_df: pd.DataFrame) -> None:
    """Two faceted strip plots for convergence rounds and training time.

    Each seed is plotted as a semi-transparent dot; mean ± std is overlaid
    as a solid horizontal line with vertical whiskers and caps.
    """
    ns = [2, 3]
    x_labels = ["2 institutions", "3 institutions"]
    last_rounds = exp3_df.groupby(["n_sites", "seed"])["round"].max().reset_index()

    DOT_COLOR  = "#4a7aa8"
    MEAN_COLOR = "#1a2a3a"

    _panel_specs = [
        (ax_rounds, "Convergence Rounds",       lambda r, n: float(r)),
        (ax_time,   "Total Training Time (min)", lambda r, n: r * _SEC_PER_ROUND[n] / 60),
    ]

    for ax, ylabel, val_fn in _panel_specs:
        all_vals_flat: list[float] = []

        for xi, n in enumerate(ns):
            rows = last_rounds[last_rounds["n_sites"] == n]
            vals = [val_fn(r, n) for r in rows["round"].values]
            all_vals_flat.extend(vals)

            # Individual seed dots with light jitter
            n_pts = len(vals)
            jitter = np.linspace(-0.10, 0.10, n_pts) if n_pts > 1 else [0.0]
            for jx, v in zip(jitter, vals):
                ax.scatter(xi + jx, v, color=DOT_COLOR, s=65, alpha=0.50,
                           linewidths=0, zorder=3)

            # Mean ± std overlay
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n_pts > 1 else 0.0

            ax.hlines(mu, xi - 0.20, xi + 0.20,
                      color=MEAN_COLOR, linewidth=2.5, zorder=4)
            ax.vlines(xi, mu - sd, mu + sd,
                      color=MEAN_COLOR, linewidth=1.5, zorder=4)
            for cap_y in [mu - sd, mu + sd]:
                ax.hlines(cap_y, xi - 0.09, xi + 0.09,
                          color=MEAN_COLOR, linewidth=1.5, zorder=4)

            # Label to the right of the mean tick — avoids title overlap
            ax.text(xi + 0.30, mu, f"{mu:.1f}",
                    ha="left", va="center",
                    fontsize=19, fontweight="bold", color=MEAN_COLOR)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(x_labels, fontsize=14)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(True, axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _plot_cost_boxplot(ax_rounds: plt.Axes, ax_time: plt.Axes,
                       exp3_df: pd.DataFrame,
                       fs_val=20, fs_mean_sub=10,
                       fs_ticks=14, fs_ylabel=14, fs_yticks=15) -> None:
    """Box plots (with overlaid seed dots) for convergence rounds and training time."""
    ns = [2, 3]
    x_labels = ["2 institutions", "3 institutions"]
    last_rounds = exp3_df.groupby(["n_sites", "seed"])["round"].max().reset_index()

    DOT_COLOR  = "#1a2a3a"

    _panel_specs = [
        (ax_rounds, "Convergence Rounds",        lambda r, n: float(r),       "#c9a84c"),  # warm yellow
        (ax_time,   "Total Training Time (min)", lambda r, n: r * _SEC_PER_ROUND[n] / 60, "#9d7b78"),  # dusty pink
    ]

    for ax, ylabel, val_fn, box_color in _panel_specs:
        groups = []
        for n in ns:
            rows = last_rounds[last_rounds["n_sites"] == n]
            groups.append([val_fn(r, n) for r in rows["round"].values])

        bp = ax.boxplot(
            groups, positions=[0, 1], widths=0.45, patch_artist=True,
            medianprops=dict(color=DOT_COLOR, linewidth=2.5),
            boxprops=dict(facecolor=box_color, alpha=0.55, linewidth=1.5),
            whiskerprops=dict(color=DOT_COLOR, linewidth=1.5),
            capprops=dict(color=DOT_COLOR, linewidth=2.0),
            flierprops=dict(marker="o", markerfacecolor=DOT_COLOR,
                            markersize=6, alpha=0.7, markeredgewidth=0),
        )

        for xi, vals in enumerate(groups):
            n_pts = len(vals)
            jitter = np.linspace(-0.08, 0.08, n_pts) if n_pts > 1 else [0.0]
            for jx, v in zip(jitter, vals):
                ax.scatter(xi + jx, v, color=DOT_COLOR, s=70, alpha=0.75,
                           linewidths=0, zorder=5)
            mu = float(np.mean(vals))
            ax.text(xi + 0.30, mu, f"{mu:.1f}",
                    ha="left", va="bottom",
                    fontsize=fs_val, fontweight="bold", color=DOT_COLOR)
            ax.text(xi + 0.30, mu, "(mean)",
                    ha="left", va="top",
                    fontsize=fs_mean_sub, color=DOT_COLOR)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(x_labels, fontsize=fs_ticks)
        ax.set_xlim(-0.6, 1.75)
        ax.set_ylabel(ylabel, fontsize=fs_ylabel)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=fs_yticks)
        ax.grid(True, axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3",      default="results/exp3.csv",
                        help="Training-loop CSV (convergence rounds and wall-clock).")
    parser.add_argument("--test_exp3", default="results/test_exp3.csv")
    parser.add_argument("--output",    default="figures/scalability.png")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_exp3)
    exp3_df = pd.read_csv(args.exp3)

    fig, (ax_sl, ax_rnd, ax_tm) = plt.subplots(
        1, 3, figsize=(22, 9),
        gridspec_kw={"width_ratios": [4, 3, 3]},
    )
    fig.suptitle(
        "Scalability: 2 vs. 3 Institutions",
        fontsize=35, fontweight="normal",
    )
    _plot_slopegraph(ax_sl, test_df,
                     fs_annot=31, fs_ticks=26, fs_ylabel=28, fs_title=32,
                     fs_yticks=26, annot_bold=True)
    ax_rnd.set_title("Convergence Rounds", fontsize=32, pad=6)
    ax_tm.set_title("Training Time (min)", fontsize=32, pad=6)
    _plot_cost_boxplot(ax_rnd, ax_tm, exp3_df,
                       fs_val=31, fs_mean_sub=18,
                       fs_ticks=26, fs_ylabel=28, fs_yticks=26)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
