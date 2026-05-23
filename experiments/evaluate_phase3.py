"""
experiments/evaluate_phase3.py — PCMU Phase 3: Metric validation.

Three checks (PCMUmetric.md §Phase 3):

  3a — Convergence-locked R consistency
    Recompute R_locked from per-round data using the same 90%-of-plateau threshold
    applied identically to all configurations.
    Gate: Spearman ρ(R_locked, R_factorial) ≥ 0.90 (round orderings preserved).
          Recompute η_comm from R_locked; verify Spearman ρ(η_comm_locked, η_comm_factorial) ≥ 0.95.

  3b — Pareto monotonicity
    Enumerate all pairs (i, j) where configuration i strictly dominates j on all three
    raw components (Δm_i > Δm_j AND η_priv_i > η_priv_j AND η_comm_i > η_comm_j).
    Gate: PCMU_i > PCMU_j for every such pair (zero violations).

  3c — Per-site stability
    Each VFL site maps to one task: Site A → IHM, Site B → Decomp, Site C → Pheno.
    For each factorial cell compute per-site utility ratio: η_s = M_s / M_s^cen.
    Report std(η_A, η_B, η_C) per cell as the within-configuration cross-site spread.
    Gate: median cross-site std ≤ 0.15 at ε=∞ (no-DP baseline; DP spread is expected to grow).

Outputs results/pcmu_phase3_gate.csv.

Usage:
    python experiments/evaluate_phase3.py
    python experiments/evaluate_phase3.py --factorial  results/pcmu_phase2_factorial.csv
                                          --rounds     results/pcmu_phase2_factorial_rounds.csv
                                          --centralized results/centralized.csv
                                          --exp1       results/exp1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.evaluate_phase2 import (
    load_centralized_aurocs,
    load_st_aurocs,
    compute_components,
    compute_centralized_components,
    add_additive_pcmu,
)

CONV_THRESHOLD      = 0.90   # fraction of plateau used to define convergence
SPEARMAN_R_THRESH   = 0.90   # gate for round-ordering agreement (3a)
SPEARMAN_EC_THRESH  = 0.95   # gate for η_comm ordering agreement (3a)
SITE_STD_CEIL       = 0.15   # gate for per-site spread at ε=∞ (3c)

_TASK_COLS = {
    "ihm":    "val_ihm_auroc",
    "decomp": "val_decomp_auroc",
    "pheno":  "val_pheno_macro_auroc",
}
W = {"ihm": 0.5, "decomp": 0.3, "pheno": 0.2}
PCMU_WEIGHTS = {"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10}


# ---------------------------------------------------------------------------
# 3a — Convergence-locked R consistency
# ---------------------------------------------------------------------------

def _weighted_auroc(row: pd.Series, active_tasks: list[str]) -> float:
    total_w = sum(W[t] for t in active_tasks)
    if total_w == 0:
        return float("nan")
    return sum(W[t] * float(row[_TASK_COLS[t]]) for t in active_tasks) / total_w


def compute_r_locked(
    rounds_df: pd.DataFrame,
    factorial_df: pd.DataFrame,
) -> pd.Series:
    """
    For each factorial cell, find the first round where val_ihm_auroc ≥ 90% of
    its plateau. IHM is the primary convergence signal (same criterion used when
    building the factorial). The same threshold is applied identically to all
    configurations — the convergence-locking guarantee.

    Returns a Series indexed the same as factorial_df with R_locked values.
    """
    r_locked_vals = []

    for idx, frow in factorial_df.iterrows():
        ed   = int(frow["embed_dim"])
        eps  = float(frow["epsilon_level"])
        cfg  = str(frow["task_config"])
        seed = int(frow["seed"])

        cell_rounds = rounds_df[
            (rounds_df["embed_dim"]     == ed) &
            (rounds_df["epsilon_level"] == eps) &
            (rounds_df["task_config"]   == cfg) &
            (rounds_df["seed"]          == seed)
        ].sort_values("round")

        if cell_rounds.empty:
            r_locked_vals.append(float("nan"))
            continue

        plateau = float(cell_rounds["val_ihm_auroc"].max())
        target  = CONV_THRESHOLD * plateau
        hit     = cell_rounds[cell_rounds["val_ihm_auroc"] >= target]
        r_locked_vals.append(float(hit["round"].min()) if not hit.empty else float("nan"))

    return pd.Series(r_locked_vals, index=factorial_df.index, name="R_locked")


def test_convergence_locked(
    df: pd.DataFrame,
    rounds_df: pd.DataFrame,
) -> tuple[bool, list[dict]]:
    """
    Recompute R_locked from per-round data; compare to factorial convergence_round.
    Then recompute η_comm from R_locked and compare to existing η_comm.
    """
    rows: list[dict] = []

    r_locked = compute_r_locked(rounds_df, df)
    df = df.copy()
    df["R_locked"] = r_locked

    valid = df[["convergence_round", "R_locked"]].dropna()
    rho_r, pval_r = stats.spearmanr(valid["convergence_round"], valid["R_locked"])
    rho_r = float(rho_r)

    # η_comm recomputed from R_locked using same per-seed R_ST
    inf_rows = df[df["epsilon_level"] == float("inf")]
    r_st_base = inf_rows[inf_rows["task_config"] == "ihm_only"].set_index(
        ["embed_dim", "seed"]
    )["R_locked"]
    R_ref_global = float(r_st_base.median()) if not r_st_base.empty else float("nan")

    eta_comm_locked = []
    for idx, r in df.iterrows():
        ed   = int(r["embed_dim"])
        seed = int(r["seed"])
        R    = float(r["R_locked"])
        try:
            R_ST = float(r_st_base.loc[(ed, seed)])
        except (KeyError, TypeError):
            R_ST = R_ref_global
        ec = np.log(1.0 + R_ST / R) if np.isfinite(R) and R > 0 else float("nan")
        eta_comm_locked.append(ec)

    df["eta_comm_locked"] = eta_comm_locked

    valid_ec = df[["eta_comm", "eta_comm_locked"]].dropna()
    rho_ec, pval_ec = stats.spearmanr(valid_ec["eta_comm"], valid_ec["eta_comm_locked"])
    rho_ec = float(rho_ec)

    gate_r  = rho_r  >= SPEARMAN_R_THRESH
    gate_ec = rho_ec >= SPEARMAN_EC_THRESH
    gate    = gate_r and gate_ec

    flag_r  = "ok" if gate_r  else "FAIL"
    flag_ec = "ok" if gate_ec else "FAIL"

    nan_frac = float(r_locked.isna().mean())

    rows.append({
        "test":         "3a_convergence_locked",
        "metric":       "R_ordering (Spearman ρ: R_locked vs R_factorial)",
        "spearman_rho": round(rho_r,  4),
        "p":            round(float(pval_r),  4),
        "threshold":    SPEARMAN_R_THRESH,
        "nan_frac":     round(nan_frac, 3),
        "gate":         flag_r,
    })
    rows.append({
        "test":         "3a_convergence_locked",
        "metric":       "eta_comm_ordering (Spearman ρ: locked vs factorial)",
        "spearman_rho": round(rho_ec, 4),
        "p":            round(float(pval_ec), 4),
        "threshold":    SPEARMAN_EC_THRESH,
        "nan_frac":     float("nan"),
        "gate":         flag_ec,
    })

    print(f"  R ordering:      Spearman ρ={rho_r:.4f}  p={pval_r:.4f}  [{flag_r}]")
    print(f"  η_comm ordering: Spearman ρ={rho_ec:.4f}  p={pval_ec:.4f}  [{flag_ec}]")
    print(f"  NaN fraction in R_locked: {nan_frac:.1%}")

    return gate, rows


# ---------------------------------------------------------------------------
# 3b — Pareto monotonicity
# ---------------------------------------------------------------------------

def test_pareto_monotonicity(df: pd.DataFrame) -> tuple[bool, list[dict]]:
    """
    For all pairs (i,j) where i strictly dominates j on Δm, η_priv, η_comm,
    verify PCMU(i) > PCMU(j).
    Gate: zero violations.
    """
    cols = ["delta_m", "eta_priv", "eta_comm", "pcmu"]
    d = df[cols].dropna().reset_index(drop=True)

    n_pairs      = 0
    n_violations = 0
    violation_rows: list[dict] = []

    for i in range(len(d)):
        for j in range(len(d)):
            if i == j:
                continue
            ri, rj = d.iloc[i], d.iloc[j]
            dominates = (
                ri["delta_m"]  > rj["delta_m"]  and
                ri["eta_priv"] > rj["eta_priv"] and
                ri["eta_comm"] > rj["eta_comm"]
            )
            if not dominates:
                continue
            n_pairs += 1
            if ri["pcmu"] <= rj["pcmu"]:
                n_violations += 1
                violation_rows.append({
                    "i_delta_m":   round(ri["delta_m"],  4),
                    "j_delta_m":   round(rj["delta_m"],  4),
                    "i_eta_priv":  round(ri["eta_priv"], 4),
                    "j_eta_priv":  round(rj["eta_priv"], 4),
                    "i_eta_comm":  round(ri["eta_comm"], 4),
                    "j_eta_comm":  round(rj["eta_comm"], 4),
                    "i_pcmu":      round(ri["pcmu"],     4),
                    "j_pcmu":      round(rj["pcmu"],     4),
                })

    gate = (n_violations == 0)
    flag = "ok" if gate else "FAIL"
    row = {
        "test":          "3b_pareto_monotonicity",
        "n_pairs":       n_pairs,
        "n_violations":  n_violations,
        "violation_frac": round(n_violations / n_pairs, 4) if n_pairs > 0 else 0.0,
        "gate":          flag,
    }

    print(f"  Dominance pairs evaluated: {n_pairs}")
    print(f"  Violations: {n_violations}  [{flag}]")
    if violation_rows:
        print(f"  First violation: i_pcmu={violation_rows[0]['i_pcmu']}  j_pcmu={violation_rows[0]['j_pcmu']}")

    return gate, [row]


# ---------------------------------------------------------------------------
# 3c — Per-site stability
# ---------------------------------------------------------------------------

def test_per_site_stability(
    df: pd.DataFrame,
    centralized_aurocs: dict[str, float],
) -> tuple[bool, list[dict]]:
    """
    Each site maps to one task: A→IHM, B→Decomp, C→Pheno.
    For each factorial cell compute η_s = M_s / M_s^cen per site.
    Report std(η_A, η_B, η_C) as cross-site spread per cell.
    Gate: median cross-site std ≤ SITE_STD_CEIL at ε=∞.
    """
    site_map = {
        "A": ("ihm",    "val_ihm_auroc"),
        "B": ("decomp", "val_decomp_auroc"),
        "C": ("pheno",  "val_pheno_macro_auroc"),
    }

    stds = []
    stds_nodp = []
    result_rows: list[dict] = []

    for _, r in df.iterrows():
        eta_sites = []
        for site, (task, col) in site_map.items():
            m_cen = centralized_aurocs.get(task, float("nan"))
            m_val = float(r[col]) if not np.isnan(float(r[col])) else float("nan")
            if m_cen > 0 and np.isfinite(m_val):
                eta_sites.append(m_val / m_cen)

        std_val = float(np.std(eta_sites)) if len(eta_sites) >= 2 else float("nan")
        stds.append(std_val)

        is_nodp = not np.isfinite(float(r["epsilon_level"]))
        if is_nodp:
            stds_nodp.append(std_val)

        result_rows.append({
            "embed_dim":     int(r["embed_dim"]),
            "epsilon_level": float(r["epsilon_level"]),
            "task_config":   str(r["task_config"]),
            "seed":          int(r["seed"]),
            "site_std":      round(std_val, 4) if np.isfinite(std_val) else float("nan"),
        })

    median_nodp = float(np.nanmedian(stds_nodp)) if stds_nodp else float("nan")
    median_all  = float(np.nanmedian(stds))

    gate = np.isfinite(median_nodp) and median_nodp <= SITE_STD_CEIL
    flag = "ok" if gate else "FAIL"

    summary_row = {
        "test":           "3c_per_site_stability",
        "median_std_nodp": round(median_nodp, 4) if np.isfinite(median_nodp) else float("nan"),
        "median_std_all":  round(median_all,  4) if np.isfinite(median_all)  else float("nan"),
        "threshold":      SITE_STD_CEIL,
        "gate":           flag,
    }

    print(f"  Median cross-site std (ε=∞): {median_nodp:.4f}  threshold={SITE_STD_CEIL}  [{flag}]")
    print(f"  Median cross-site std (all ε): {median_all:.4f}")

    return gate, [summary_row]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial",    default="results/pcmu_phase2_factorial.csv")
    parser.add_argument("--rounds",       default="results/pcmu_phase2_factorial_rounds.csv")
    parser.add_argument("--centralized",  default="results/centralized.csv")
    parser.add_argument("--exp1",         default="results/exp1.csv")
    parser.add_argument("--out",          default="results/pcmu_phase3_gate.csv")
    args = parser.parse_args()

    # Load factorial and rebuild PCMU components (same pipeline as Phase 2/2b)
    df = pd.read_csv(args.factorial)
    df = df[pd.to_numeric(df["embed_dim"], errors="coerce").notna()].copy()
    df["embed_dim"]     = df["embed_dim"].astype(int)
    df["epsilon_level"] = df["epsilon_level"].astype(float)
    df["seed"]          = df["seed"].astype(int)
    print(f"Loaded {len(df)} rows from {args.factorial}")

    rounds_df = pd.read_csv(args.rounds)
    rounds_df["embed_dim"]     = rounds_df["embed_dim"].astype(int)
    rounds_df["epsilon_level"] = rounds_df["epsilon_level"].astype(float)
    rounds_df["seed"]          = rounds_df["seed"].astype(int)
    print(f"Loaded {len(rounds_df)} per-round rows from {args.rounds}")

    cen_df  = pd.read_csv(args.centralized)
    exp1_df = pd.read_csv(args.exp1)
    centralized_aurocs = load_centralized_aurocs(cen_df)
    st_aurocs          = load_st_aurocs(exp1_df)
    print(f"Centralized AUROCs: { {k: round(v, 4) for k, v in centralized_aurocs.items()} }")
    print(f"VFL-ST AUROCs:      { {k: round(v, 4) for k, v in st_aurocs.items()} }")

    # Rebuild components + PCMU (same as Phase 2b pipeline)
    df, _R_ref    = compute_components(df, centralized_aurocs, st_aurocs)
    eta_comm_max  = float(df["eta_comm"].dropna().max())
    cen_comps     = compute_centralized_components(centralized_aurocs, st_aurocs, eta_comm_max)
    df            = add_additive_pcmu(df, cen_comps)

    # Rename pcmu_additive → pcmu for downstream tests
    if "pcmu_additive" in df.columns and "pcmu" not in df.columns:
        df = df.rename(columns={"pcmu_additive": "pcmu"})
    elif "pcmu_additive" in df.columns:
        df["pcmu"] = df["pcmu_additive"]

    print("\n" + "=" * 60)
    print("PHASE 3 GATE — METRIC VALIDATION")
    print("=" * 60)

    overall_pass = True
    all_rows: list[dict] = []

    # Test 3a
    print("\n[Test 3a — Convergence-locked R consistency]")
    t3a_pass, t3a_rows = test_convergence_locked(df, rounds_df)
    all_rows.extend(t3a_rows)
    if not t3a_pass:
        overall_pass = False

    # Test 3b
    print("\n[Test 3b — Pareto monotonicity]")
    t3b_pass, t3b_rows = test_pareto_monotonicity(df)
    all_rows.extend(t3b_rows)
    if not t3b_pass:
        overall_pass = False

    # Test 3c
    print("\n[Test 3c — Per-site stability]")
    t3c_pass, t3c_rows = test_per_site_stability(df, centralized_aurocs)
    all_rows.extend(t3c_rows)
    if not t3c_pass:
        overall_pass = False

    print("\n" + "=" * 60)
    if overall_pass:
        print("✅ PHASE 3 GATE: PASS — PCMU metric is validated.")
        print("   PCMU is ready for use in Paper 1 and Paper 2 analyses.")
    else:
        print("❌ PHASE 3 GATE: FAIL — inspect failed tests before reporting PCMU.")
    print("=" * 60)

    gate_df = pd.DataFrame(all_rows)
    gate_df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
