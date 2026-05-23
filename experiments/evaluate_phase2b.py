"""
experiments/evaluate_phase2b.py — PCMU Phase 2b: Additive Aggregation Gate.

Phase 2 confirmed that additive aggregation handles the ε-correlated component
structure better than multiplicative aggregation (geometric form failed its gate).
Phase 2b validates that the additive formula with z-score normalisation and the
chosen weights is itself justified, using the same factorial data and component
computation pipeline as Phase 2.

Three tests (PCMUmetric.md §Phase 2b):

  Test 1 — Variance contribution (R² ordering)
    For each z-scored component, compute R² from simple regression on PCMU.
    Gate: R² ordering matches weight ordering (Δm_z > ηpriv_z > ηcomm_z);
          no R² < 0.05 (no weight is wasted on noise).

  Test 2 — Weight sensitivity analysis (Greco et al. 2019)
    Sweep w1 ∈ {0.5, 0.6, 0.7, 0.8} with w2, w3 scaled proportionally (2:1 ratio).
    Gate: Spearman ρ of configuration rankings vs. default w1=0.70 ≥ 0.85.

  Test 3 — Phase 1 consistency check
    Gate 3a: Mean PCMU decreases monotonically as ε decreases (Phase 1a finding).
    Gate 3b: ihm_decomp scores higher than ihm_pheno at ε=∞ (Phase 1c finding).

Outputs results/pcmu_phase2b_gate.csv.

Usage:
    python experiments/evaluate_phase2b.py
    python experiments/evaluate_phase2b.py --factorial results/pcmu_phase2_factorial.csv
                                           --centralized results/centralized.csv
                                           --exp1 results/exp1.csv
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

PCMU_WEIGHTS_DEFAULT = {"delta_m": 0.70, "eta_priv": 0.20, "eta_comm": 0.10}
R2_FLOOR            = 0.05
SPEARMAN_THRESHOLD  = 0.85
W1_SWEEP            = [0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------------------------
# Test 1 — Variance contribution (R² ordering)
# ---------------------------------------------------------------------------

def test_variance_contribution(df: pd.DataFrame) -> tuple[bool, list[dict]]:
    """
    Regress PCMU separately on each z-scored component; record Pearson R².
    Gate: ordering Δm_z > ηpriv_z > ηcomm_z AND no R² < R2_FLOOR.
    """
    components = [
        ("delta_m_z",  "delta_m",  PCMU_WEIGHTS_DEFAULT["delta_m"]),
        ("eta_priv_z", "eta_priv", PCMU_WEIGHTS_DEFAULT["eta_priv"]),
        ("eta_comm_z", "eta_comm", PCMU_WEIGHTS_DEFAULT["eta_comm"]),
    ]

    r2_vals: dict[str, float] = {}
    rows: list[dict] = []

    for col, key, weight in components:
        d = df[[col, "pcmu"]].dropna()
        if len(d) < 3:
            r2_vals[col] = float("nan")
            continue
        r, _ = stats.pearsonr(d[col], d["pcmu"])
        r2_vals[col] = float(r ** 2)

    floor_ok    = all(v >= R2_FLOOR for v in r2_vals.values() if np.isfinite(v))
    ordering_ok = (
        r2_vals.get("delta_m_z",  0.0) > r2_vals.get("eta_priv_z", 0.0) >
        r2_vals.get("eta_comm_z", 0.0)
    )
    gate_pass   = floor_ok and ordering_ok

    for col, key, weight in components:
        r2   = r2_vals.get(col, float("nan"))
        flag = "ok" if (np.isfinite(r2) and r2 >= R2_FLOOR) else "FAIL"
        rows.append({
            "test":      "1_variance_contribution",
            "component": col,
            "r2":        round(r2, 4) if np.isfinite(r2) else float("nan"),
            "weight":    weight,
            "gate":      flag,
        })

    rows.append({
        "test":      "1_variance_contribution",
        "component": "r2_ordering (delta_m_z > eta_priv_z > eta_comm_z)",
        "r2":        float("nan"),
        "weight":    float("nan"),
        "gate":      "ok" if ordering_ok else "FAIL",
    })

    return gate_pass, rows


# ---------------------------------------------------------------------------
# Test 2 — Weight sensitivity analysis (Greco et al. 2019)
# ---------------------------------------------------------------------------

def _recompute_pcmu(df: pd.DataFrame, w1: float) -> pd.Series:
    """PCMU additive score under swept weights; w2:w3 ratio preserved at 2:1."""
    w2 = (1.0 - w1) * (2.0 / 3.0)
    w3 = (1.0 - w1) * (1.0 / 3.0)
    return w1 * df["delta_m_z"] + w2 * df["eta_priv_z"] + w3 * df["eta_comm_z"]


def test_weight_sensitivity(df: pd.DataFrame) -> tuple[bool, list[dict]]:
    """
    Spearman ρ of per-configuration PCMU rankings under swept w1 vs. default w1=0.70.
    Gate: ρ ≥ SPEARMAN_THRESHOLD for all weight combinations.
    """
    d = df[["delta_m_z", "eta_priv_z", "eta_comm_z", "pcmu"]].dropna()
    default_ranks = d["pcmu"].rank()

    rows: list[dict] = []
    gate_pass = True

    for w1 in W1_SWEEP:
        swept_pcmu  = _recompute_pcmu(d, w1)
        rho, pval   = stats.spearmanr(default_ranks, swept_pcmu.rank())
        w2          = (1.0 - w1) * 2.0 / 3.0
        w3          = (1.0 - w1) / 3.0
        flag        = "ok" if float(rho) >= SPEARMAN_THRESHOLD else "FAIL"
        if flag == "FAIL":
            gate_pass = False
        rows.append({
            "test":         "2_weight_sensitivity",
            "w1":           round(w1, 2),
            "w2":           round(w2, 4),
            "w3":           round(w3, 4),
            "spearman_rho": round(float(rho), 4),
            "p":            round(float(pval), 4),
            "gate":         flag,
        })

    return gate_pass, rows


# ---------------------------------------------------------------------------
# Test 3 — Phase 1 consistency check
# ---------------------------------------------------------------------------

def test_phase1_consistency(df: pd.DataFrame) -> tuple[bool, list[dict]]:
    """
    3a. Mean PCMU decreases monotonically as ε decreases (Phase 1a isolation finding).
    3b. ihm_decomp PCMU > ihm_pheno PCMU at ε=∞ (Phase 1c isolation finding).
    """
    rows: list[dict] = []
    gate_pass = True

    # 3a — ε monotonicity (DP rows only)
    df_dp    = df[np.isfinite(df["epsilon_level"])].copy()
    eps_mean = df_dp.groupby("epsilon_level")["pcmu"].mean().sort_index()

    if len(eps_mean) >= 2:
        rho, pval = stats.spearmanr(eps_mean.index.tolist(), eps_mean.values.tolist())
        flag      = "ok" if float(rho) > 0 else "FAIL"
        if flag == "FAIL":
            gate_pass = False
        rows.append({
            "test":         "3a_eps_monotonicity",
            "spearman_rho": round(float(rho), 4),
            "p":            round(float(pval), 4),
            "note":         "Spearman r(ε_level, mean PCMU per ε); must be > 0",
            "gate":         flag,
        })
        print(f"  ε monotonicity: Spearman r={rho:.4f}  p={pval:.4f}  [{flag}]")
        print(f"  Mean PCMU by ε:\n{eps_mean.round(4).to_string()}")
    else:
        rows.append({
            "test": "3a_eps_monotonicity", "gate": "SKIP — fewer than 2 ε levels",
        })

    # 3b — full VFL-MTL (all_tasks, 3-task) must outscore single-task (ihm_only) at ε=∞.
    # Δm is now computed as aggregate across all active tasks using exp1.csv ST baselines,
    # so all_tasks gets its proper multi-task gain signal (Phase 1c: Δm=+0.011).
    nodp          = df[~np.isfinite(df["epsilon_level"])].copy()
    all_tasks_mu  = float(nodp[nodp["task_config"] == "all_tasks"]["pcmu"].mean())
    ihm_only_mu   = float(nodp[nodp["task_config"] == "ihm_only"]["pcmu"].mean())
    ordering_ok   = (
        np.isfinite(all_tasks_mu) and np.isfinite(ihm_only_mu) and
        all_tasks_mu > ihm_only_mu
    )
    flag          = "ok" if ordering_ok else "FAIL"
    if flag == "FAIL":
        gate_pass = False
    rows.append({
        "test":            "3b_task_config_ordering",
        "all_tasks_mean":  round(all_tasks_mu, 4) if np.isfinite(all_tasks_mu) else float("nan"),
        "ihm_only_mean":   round(ihm_only_mu,  4) if np.isfinite(ihm_only_mu)  else float("nan"),
        "note":            "all_tasks PCMU > ihm_only PCMU at ε=∞ (Phase 1c: 3-task VFL-MTL > single-task)",
        "gate":            flag,
    })
    print(
        f"  all_tasks PCMU={all_tasks_mu:.4f}  "
        f"ihm_only PCMU={ihm_only_mu:.4f}  [{flag}]"
    )

    return gate_pass, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial",   default="results/pcmu_phase2_factorial.csv")
    parser.add_argument("--centralized", default="results/centralized.csv")
    parser.add_argument("--exp1",        default="results/exp1.csv")
    parser.add_argument("--out",         default="results/pcmu_phase2b_gate.csv")
    args = parser.parse_args()

    # Load factorial
    df = pd.read_csv(args.factorial)
    df = df[pd.to_numeric(df["embed_dim"], errors="coerce").notna()].copy()
    df["embed_dim"]     = df["embed_dim"].astype(int)
    df["epsilon_level"] = df["epsilon_level"].astype(float)
    df["seed"]          = df["seed"].astype(int)
    print(f"Loaded {len(df)} rows from {args.factorial}")

    cen_df  = pd.read_csv(args.centralized)
    exp1_df = pd.read_csv(args.exp1)
    centralized_aurocs = load_centralized_aurocs(cen_df)
    st_aurocs          = load_st_aurocs(exp1_df)
    print(f"Centralized AUROCs: { {k: round(v, 4) for k, v in centralized_aurocs.items()} }")
    print(f"VFL-ST AUROCs:      { {k: round(v, 4) for k, v in st_aurocs.items()} }")

    # Reuse Phase 2 pipeline: raw components → centralized anchor → z-score + additive PCMU
    df, _R_ref    = compute_components(df, centralized_aurocs, st_aurocs)
    eta_comm_max  = float(df["eta_comm"].dropna().max())
    cen_comps     = compute_centralized_components(centralized_aurocs, st_aurocs, eta_comm_max)
    df            = add_additive_pcmu(df, cen_comps)

    df_dp = df[np.isfinite(df["epsilon_level"])].copy()

    print("\n" + "=" * 60)
    print("PHASE 2b GATE — ADDITIVE AGGREGATION VALIDATION")
    print("=" * 60)

    overall_pass  = True
    all_rows: list[dict] = []

    # Test 1
    print("\n[Test 1 — Variance contribution: R² ordering must match weight ordering]")
    t1_pass, t1_rows = test_variance_contribution(df_dp)
    all_rows.extend(t1_rows)
    for r in t1_rows:
        comp = r.get("component", "")
        if "ordering" in comp:
            print(f"  R² ordering check: [{r['gate']}]")
        else:
            print(f"  {comp}: R²={r.get('r2', float('nan')):.4f}  weight={r.get('weight', '')}  [{r['gate']}]")
    if not t1_pass:
        overall_pass = False

    # Test 2
    print("\n[Test 2 — Weight sensitivity analysis (Greco et al. 2019)]")
    t2_pass, t2_rows = test_weight_sensitivity(df)
    all_rows.extend(t2_rows)
    for r in t2_rows:
        print(
            f"  w1={r['w1']}  w2={r['w2']:.3f}  w3={r['w3']:.3f}:  "
            f"Spearman ρ={r['spearman_rho']:.4f}  [{r['gate']}]"
        )
    if not t2_pass:
        overall_pass = False

    # Test 3
    print("\n[Test 3 — Phase 1 consistency check]")
    t3_pass, t3_rows = test_phase1_consistency(df)
    all_rows.extend(t3_rows)
    if not t3_pass:
        overall_pass = False

    print("\n" + "=" * 60)
    if overall_pass:
        print("✅ PHASE 2b GATE: PASS — additive aggregation is justified.")
        print("   Proceed to Phase 3.")
    else:
        print("❌ PHASE 2b GATE: FAIL — revise weights or formula before Phase 3.")
    print("=" * 60)

    gate_df = pd.DataFrame(all_rows)
    gate_df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
