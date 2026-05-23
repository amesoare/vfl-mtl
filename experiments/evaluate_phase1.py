"""experiments/evaluate_phase1.py — PCMU Phase 1 isolation analyses (PCMUmetric.md §Phase 1).

Usage:
    python experiments/evaluate_phase1.py [--results_dir results] [--out_dir results]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compute_pcmu import (
    PCMUConfig,
    comm_efficiency,
    convergence_round,
    convergence_round_mean,
    evaluate_sweep,
    load_centralized_aurocs,
    load_st_aurocs,
    multitask_gain,
    peak_aurocs,
    task_relatedness_from_prevalences,
)

CFG = PCMUConfig()

_PREVALENCES: dict[str, float | np.ndarray] = {
    "ihm":    0.116,   # Harutyunyan et al. 2019
    "decomp": 0.028,
    "pheno":  np.full(25, 0.14),
}

# Which tasks are active (trained on) per exp3 task_config label.
_ACTIVE_TASKS: dict[str, set[str]] = {
    "all_tasks":  {"ihm", "decomp", "pheno"},
    "ihm_decomp": {"ihm", "decomp"},
    "ihm_pheno":  {"ihm", "pheno"},
    "ihm_only":   {"ihm"},
}


# ── Phase 1a: privacy isolation ────────────────────────────────────────────

def phase1a(privacy_df: pd.DataFrame, exp1_df: pd.DataFrame) -> pd.DataFrame:
    df = evaluate_sweep(privacy_df, exp1_df, mode="uniform", config=CFG)
    summary = (
        df.groupby("epsilon_level")[["delta_m", "eta_priv", "eta_comm", "pcmu", "r"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n── Phase 1a: privacy isolation (η_priv vs ε, embed_dim=64, all_tasks) ──")
    print(summary.to_string())
    return df


# ── Phase 1b: communication isolation ─────────────────────────────────────

def phase1b(abl_df: pd.DataFrame, exp1_df: pd.DataFrame) -> pd.DataFrame:
    abl3 = abl_df[abl_df["ablation"] == "abl3"].copy()

    # R_ref: mean convergence round of VFL-MTL at ε=∞ from exp1 (embed_dim=64 baseline)
    mtl_nodp = exp1_df[exp1_df["model"] == "VFL-MTL"]
    r_ref = convergence_round_mean(
        mtl_nodp, "val_ihm_auroc", threshold=CFG.conv_threshold
    )

    rows = []
    for _, row in abl3.iterrows():
        # Per-round data unavailable for abl3; R=100 is a conservative upper bound.
        r_approx = 100.0
        ec = comm_efficiency(r_approx, r_ref)
        rows.append({
            "embed_dim":     row["embed_dim"],
            "epsilon_level": row["epsilon_level"],
            "seed":          row["seed"],
            "val_ihm_auroc":          row["val_ihm_auroc"],
            "val_decomp_auroc":       row["val_decomp_auroc"],
            "val_pheno_macro_auroc":  row["val_pheno_macro_auroc"],
            "eta_comm_approx":        round(ec, 4),
            "r":                      r_approx,
            "r_ref":                  round(r_ref, 1),
        })

    df = pd.DataFrame(rows)
    nodp = df[df["epsilon_level"] == float("inf")]
    summary = (
        nodp.groupby("embed_dim")[
            ["val_ihm_auroc", "val_decomp_auroc", "val_pheno_macro_auroc", "eta_comm_approx"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n── Phase 1b: comm isolation (η_comm vs embed_dim, ε=∞, all_tasks) ──")
    print(f"   R_ref (embed_dim=64, ε=∞, VFL-MTL) = {r_ref:.1f} rounds")
    print(f"   Note: R=100 used for abl3 (no per-round data) — η_comm is a lower bound")
    print(summary.to_string())
    return df


# ── Phase 1c: multi-task gain isolation ───────────────────────────────────

def phase1c(exp3_df: pd.DataFrame, exp1_df: pd.DataFrame) -> pd.DataFrame:
    st_aurocs = load_st_aurocs(exp1_df)

    # Task-relatedness via Hellinger distance (lower = more similar distributions)
    relatedness = task_relatedness_from_prevalences(_PREVALENCES)
    print("\n── Phase 1c: MTL gain isolation (Δ_m vs task config + relatedness) ──")
    print("   Hellinger distances (lower = more related, more similar label distributions):")
    for pair, dist in relatedness.items():
        print(f"     H({pair[0]}, {pair[1]}) = {dist:.4f}")

    rows = []
    for cfg_name, cfg_df in exp3_df.groupby("task_config"):
        active = _ACTIVE_TASKS.get(str(cfg_name), set(CFG.task_weights.keys()))
        for seed_val, seed_df in cfg_df.groupby("seed"):
            seed_df = pd.DataFrame(seed_df)
            mtl = peak_aurocs(seed_df)

            # Only include tasks that were actually trained in this config
            active_weights = {t: w for t, w in CFG.task_weights.items() if t in active}
            if not active_weights:
                continue

            # Renormalise weights to sum to 1 for sub-task configs
            total_w = sum(active_weights.values())
            norm_weights = {t: w / total_w for t, w in active_weights.items()}

            try:
                dm = multitask_gain(mtl, st_aurocs, norm_weights)
            except ValueError:
                dm = float("nan")

            rows.append({
                "task_config":           cfg_name,
                "n_active_tasks":        len(active),
                "seed":                  seed_val,
                "delta_m":               round(dm, 4),
                "auroc_ihm":             mtl.get("ihm", float("nan")),
                "auroc_decomp":          mtl.get("decomp", float("nan")),
                "auroc_pheno":           mtl.get("pheno", float("nan")),
            })

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["task_config", "n_active_tasks"])["delta_m"]
        .agg(["mean", "std"])
        .round(4)
        .rename(columns={"mean": "delta_m_mean", "std": "delta_m_std"})
        .sort_values("n_active_tasks")
    )
    print("\n   Δ_m per task config (mean ± std across seeds):")
    print(summary.to_string())
    print("\n   Hypothesis: configs with more related tasks (lower H) → larger Δ_m")
    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir",     default="results")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    od = Path(args.out_dir)

    exp1_df    = pd.read_csv(rd / "exp1.csv")
    exp3_df    = pd.read_csv(rd / "exp2.csv")
    privacy_df = pd.read_csv(rd / "privacy_utility_combined.csv")
    abl_df     = pd.read_csv(rd / "dp_ablations.csv")

    df1a = phase1a(privacy_df, exp1_df)
    df1b = phase1b(abl_df, exp1_df)
    df1c = phase1c(exp3_df, exp1_df)

    df1a.to_csv(od / "pcmu_phase1a.csv", index=False)
    df1b.to_csv(od / "pcmu_phase1b.csv", index=False)
    df1c.to_csv(od / "pcmu_phase1c.csv", index=False)
    print(f"\nSaved → {od}/pcmu_phase1{{a,b,c}}.csv")


if __name__ == "__main__":
    main()
