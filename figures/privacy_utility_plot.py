"""
figures/privacy_utility_plot.py — Figure 1: Privacy-Utility Curves.

Three-panel line plot (one per task: IHM / Decomp / Pheno).
  x-axis: ε ∈ {0.5, 1, 2, 5, 10, ∞} (log scale)
  y-axis: mean AUC-ROC ± std across seeds
  Two lines: uniform σ vs. task-stratified σ
  Horizontal dashed line: clinical utility floor
  Vertical marker: ε* — crossing point below the floor

Usage:
    python figures/privacy_utility_plot.py \
        --input results/privacy_utility.csv \
        --output figures/privacy_utility.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

# Brand palette — matches plot_results_summary.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

CLINICAL_FLOORS = {"IHM": 0.75, "Decomp": 0.70, "Pheno": 0.65}
TASK_COLS = {
    "IHM":    "val_ihm_auroc",
    "Decomp": "val_decomp_auroc",
    "Pheno":  "val_pheno_macro_auroc",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/privacy_utility.csv")
    parser.add_argument("--output", default="figures/privacy_utility.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))
    df = _last_round_per_seed(df)

    modes_present = df["mode"].unique().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Privacy-Utility Curves (VFL-MTL)", fontsize=12, fontweight="bold")

    for ax, (task_name, task_col) in zip(axes, TASK_COLS.items()):
        floor = CLINICAL_FLOORS[task_name]
        means, stds = _summarise(df, task_col)

        for mode in ["uniform", "stratified"]:
            if mode not in modes_present:
                continue
            style = MODE_STYLE[mode]
            eps_vals, mu_vals, sd_vals = [], [], []
            for eps in EPS_ORDER:
                try:
                    mu = means.loc[(mode, eps)]
                    sd = stds.loc[(mode, eps)]
                    eps_vals.append(eps)
                    mu_vals.append(mu)
                    sd_vals.append(sd if not np.isnan(sd) else 0.0)
                except KeyError:
                    continue

            # sentinel value for ∞ on linear x-axis
            x_plot = [e if e != float("inf") else 20.0 for e in eps_vals]
            mu_arr = np.array(mu_vals)
            sd_arr = np.array(sd_vals)

            ax.plot(x_plot, mu_arr, color=style["color"], ls=style["ls"],
                    marker="o", ms=4, linewidth=1.4, label=style["label"])
            ax.fill_between(x_plot, mu_arr - sd_arr, mu_arr + sd_arr,
                            color=style["color"], alpha=0.15)

            # ε* marker
            eps_star = _find_eps_star(means, floor, mode)
            if eps_star is not None:
                x_star = eps_star if eps_star != float("inf") else 20.0
                y_star = means.loc[(mode, eps_star)]
                ax.axvline(x_star, color=style["color"], ls=":", alpha=0.5, linewidth=1.0)
                ax.annotate(f"ε*={EPS_LABELS[EPS_ORDER.index(eps_star)]}",
                            xy=(x_star, y_star), xytext=(x_star + 0.3, y_star + 0.01),
                            fontsize=7, color=style["color"])

        # clinical floor reference line — matches existing style (#888888, lw=0.8)
        ax.axhline(floor, color="#888888", linestyle="--", linewidth=0.8,
                   label=f"Floor ({floor})")

        x_ticks = [e if e != float("inf") else 20.0 for e in EPS_ORDER]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(EPS_LABELS, fontsize=8)
        ax.set_xlabel("Privacy budget ε", fontsize=8)
        ax.set_ylabel("Mean AUC-ROC", fontsize=8)
        ax.set_title(task_name, fontsize=9, fontweight="bold")
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
