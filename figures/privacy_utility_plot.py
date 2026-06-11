"""
figures/privacy_utility_plot.py — Privacy-Utility Curves (MIMIC).

Three stacked panels (IHM / Decomp / Pheno).
  x-axis: ε ∈ {0.5, 1, 2, 5, 10, ∞}
  y-axis: AUC-ROC per seed trace + mean line
  Two conditions: uniform σ vs. task-stratified σ
  Horizontal dashed line: clinical utility floor

Usage:
    python figures/privacy_utility_plot.py \
        --input results/test_results_dp.csv \
        --output figures/privacy_utility_test.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Brand palette — matches plot_results_summary.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

CLINICAL_FLOORS = {"IHM": 0.75, "Decomp": 0.70, "Pheno": 0.65}
TASK_COLS_VAL = {
    "IHM":    "val_ihm_auroc",
    "Decomp": "val_decomp_auroc",
    "Pheno":  "val_pheno_macro_auroc",
}
TASK_COLS_TEST = {
    "IHM":    "ihm_auroc",
    "Decomp": "decomp_auroc",
    "Pheno":  "pheno_macro_auroc",
}
EPS_ORDER  = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = ["0.5", "1", "2", "5", "10", "∞"]

MODE_STYLE = {
    "uniform":    {"color": _C[4], "ls": "-",  "label": "Uniform σ"},
    "stratified": {"color": _C[3], "ls": "--", "label": "Stratified σ"},
}


def _last_round_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["mode", "epsilon_level", "seed"]).last().reset_index()


def _summarise(df: pd.DataFrame, task_col: str):
    grp = df.groupby(["mode", "epsilon_level"])[task_col]
    return grp.mean(), grp.std()


def _find_eps_star(means: pd.Series, floor: float, mode: str) -> float | None:
    sub = means.xs(mode, level="mode") if mode in means.index.get_level_values("mode") else None
    if sub is None:
        return None
    for eps in EPS_ORDER:
        if eps in sub.index and sub[eps] >= floor:
            return eps
    return None


def _plot_privacy(df: pd.DataFrame, output: Path,
                  task_cols: dict[str, str]) -> None:
    """Privacy-utility figure: seed traces + summary lines, no shading.

    Layout: three stacked panels (IHM / Decomp / Pheno), one per row.
    Uniform σ  — faint grey seed lines + dots connecting the same seed across ε;
                  solid black mean line.
    Stratified σ — faint red seed dots; solid red mean dots.
    Clinical floor: dashed grey horizontal line with value annotated above it.
    Single shared legend at top-right of top panel: Uniform σ / Stratified σ / Clinical floor.
    """
    UNIFORM_COLOR = "#1a1a1a"
    STRAT_COLOR   = "#c0392b"
    FLOOR_COLOR   = "#888888"

    SEED_STYLE: dict[str, dict] = {
        "uniform":    {"color": UNIFORM_COLOR, "alpha_line": 0.18, "alpha_dot": 0.30},
        "stratified": {"color": STRAT_COLOR,   "alpha_line": 0.15, "alpha_dot": 0.28},
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        "AUC-ROC vs. Privacy Budget ε",
        fontsize=35, fontweight="normal",
    )

    legend_handles = {}

    for ax, (task_name, task_col) in zip(axes, task_cols.items()):
        floor = CLINICAL_FLOORS[task_name]

        for mode in ["uniform", "stratified"]:
            mode_df = df[df["mode"] == mode]
            if mode_df.empty or task_col not in mode_df.columns:
                continue

            style      = SEED_STYLE[mode]
            is_uniform = mode == "uniform"

            seeds = list(mode_df["seed"].unique())
            n_seeds = len(seeds)
            jitters = np.linspace(-0.18, 0.18, n_seeds) if n_seeds > 1 else [0.0]
            for seed, jitter in zip(seeds, jitters):
                seed_df = mode_df[mode_df["seed"] == seed].sort_values("epsilon_level")
                x_s, y_s = [], []
                for _, row in seed_df.iterrows():
                    eps = row["epsilon_level"]
                    val = row[task_col]
                    if eps in EPS_ORDER and not np.isnan(val):
                        x_s.append(EPS_ORDER.index(eps))
                        y_s.append(float(val))
                if x_s:
                    x_jit = [x + jitter for x in x_s]
                    ax.plot(x_jit, y_s, color=style["color"],
                            alpha=style["alpha_line"], linewidth=1.1, zorder=2)
                    ax.scatter(x_jit, y_s, color=style["color"], s=30,
                               alpha=style["alpha_dot"], linewidths=0, zorder=3)

            eps_x_list, mu_list = [], []
            for eps in EPS_ORDER:
                sub = mode_df[mode_df["epsilon_level"] == eps][task_col].dropna()
                if not sub.empty:
                    eps_x_list.append(EPS_ORDER.index(eps))
                    mu_list.append(float(sub.mean()))

            if is_uniform:
                h, = ax.plot(eps_x_list, mu_list, color=UNIFORM_COLOR, ls="-",
                             marker="o", ms=6, linewidth=2.5, zorder=5)
                legend_handles.setdefault("uniform", h)
            else:
                h = ax.scatter(eps_x_list, mu_list, color=STRAT_COLOR, s=80,
                               marker="o", zorder=5)
                legend_handles.setdefault("stratified", h)

        floor_line = ax.axhline(floor, color=FLOOR_COLOR, linestyle="--",
                                linewidth=1.2, zorder=1)
        legend_handles.setdefault("floor", floor_line)

        # Floor value annotation slightly above the dashed line, left-aligned
        ax.text(0, floor + 0.012, f"{floor:.2f}",
                ha="left", va="bottom", fontsize=26, color=FLOOR_COLOR)

        ax.set_ylabel("AUC-ROC", fontsize=28)
        ax.set_title(task_name, fontsize=32, fontweight="bold", pad=6)
        ax.set_ylim(0.25, 1.05)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=26)

    axes[-1].set_xticks(range(len(EPS_ORDER)))
    axes[-1].set_xticklabels(EPS_LABELS, fontsize=26)
    axes[-1].set_xlabel("Privacy budget ε", fontsize=28)

    # Single shared legend in the top-right corner of the top panel
    seed_uniform_proxy = Line2D([0], [0], color=UNIFORM_COLOR, linewidth=1.1,
                                alpha=0.35, linestyle="-")
    leg_h = [
        legend_handles.get("uniform"),
        seed_uniform_proxy,
        legend_handles.get("stratified"),
        legend_handles.get("floor"),
    ]
    leg_l = [
        "Uniform σ (mean)",
        "Uniform σ (individual seeds)",
        "Stratified σ (mean)",
        "Clinical floor",
    ]
    valid = [(h, l) for h, l in zip(leg_h, leg_l) if h is not None]
    if valid:
        axes[0].legend(
            [h for h, _ in valid], [l for _, l in valid],
            fontsize=14, loc="lower right",
            framealpha=0.9, edgecolor="#cccccc",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/test_results_dp.csv")
    parser.add_argument("--output", default="figures/privacy_utility_test.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))
    _plot_privacy(df, Path(args.output), TASK_COLS_TEST)


if __name__ == "__main__":
    main()
