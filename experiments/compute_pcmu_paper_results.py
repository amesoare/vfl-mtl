"""
experiments/compute_pcmu_paper_results.py

Apply the validated PCMU formula to the paper's main experimental results.
Z-score parameters are fixed from the 108-cell factorial (pcmu_phase2_factorial.csv)
so the scale is consistent across all comparisons.

Sources processed:
  exp1.csv                    — VFL-MTL vs. single-task baselines at ε=∞ (Paper 1 main table)
  exp3.csv                    — task configs: all_tasks / ihm_only / ihm_decomp / ihm_pheno (Paper 1 MTL gain)
  privacy_utility_combined.csv — ε sweep (Paper 2 main result)
  dp_ablations.csv            — uniform vs. stratified σ, task-pair coupling, embed_dim × DP (Paper 2 ablations)

Output: results/pcmu_paper_results.csv

η_comm note: requires rounds-to-convergence.  R_MTL = last training round per run.
  R_ST = last round of ihm_only in exp3 per seed (actual single-task reference run).
  dp_ablations.csv has no round column; η_comm is set to NaN for those rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.evaluate_phase2 import (
    load_centralized_aurocs,
    load_st_aurocs,
    compute_components,
    compute_centralized_components,
    add_additive_pcmu,
    W,
    PCMU_WEIGHTS,
    _TASK_COLS,
    _CEN_COLS,
)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

FACTORIAL_PATH  = "results/pcmu_phase2_factorial.csv"
CENTRALIZED_PATH = "results/centralized.csv"
EXP1_PATH       = "results/exp1.csv"
EXP3_PATH       = "results/exp2.csv"
PUC_PATH        = "results/privacy_utility_combined.csv"
DPA_PATH        = "results/dp_ablations.csv"
OUT_PATH        = "results/pcmu_paper_results.csv"


# ---------------------------------------------------------------------------
# Load fixed z-score parameters from the 108-cell factorial
# ---------------------------------------------------------------------------

def load_zparams(
    factorial_path: str,
    centralized_path: str,
    exp1_path: str,
) -> tuple[dict[str, tuple[float, float]], float, float]:
    """
    Returns z_params, cen_pcmu_raw, and the median R_ST from the factorial.
    These are frozen from the validation run and must not be re-estimated.
    """
    df     = pd.read_csv(factorial_path)
    cen_df = pd.read_csv(centralized_path)
    exp1   = pd.read_csv(exp1_path)

    ca  = load_centralized_aurocs(cen_df)
    st  = load_st_aurocs(exp1)
    df, R_ref_global = compute_components(df, ca, st)
    eta_max = float(df["eta_comm"].dropna().max())
    cen     = compute_centralized_components(ca, st, eta_max)
    df      = add_additive_pcmu(df, cen)

    z_params: dict[str, tuple[float, float]] = {}
    for col in ("delta_m", "eta_priv", "eta_comm"):
        vals = df[col].dropna()
        z_params[col] = (float(vals.mean()), float(vals.std(ddof=1)))

    mu_dm, s_dm = z_params["delta_m"]
    mu_ep, s_ep = z_params["eta_priv"]
    mu_ec, s_ec = z_params["eta_comm"]
    cen_raw = (
        PCMU_WEIGHTS["delta_m"]  * (cen["delta_m"]  - mu_dm) / s_dm +
        PCMU_WEIGHTS["eta_priv"] * (cen["eta_priv"] - mu_ep) / s_ep +
        PCMU_WEIGHTS["eta_comm"] * (cen["eta_comm"] - mu_ec) / s_ec
    )

    return z_params, float(cen_raw), R_ref_global


# ---------------------------------------------------------------------------
# Core PCMU computation from raw components
# ---------------------------------------------------------------------------

def pcmu_from_components(
    delta_m: float,
    eta_priv: float,
    eta_comm: float | float,
    z_params: dict[str, tuple[float, float]],
    cen_raw: float,
) -> float:
    mu_dm, s_dm = z_params["delta_m"]
    mu_ep, s_ep = z_params["eta_priv"]
    mu_ec, s_ec = z_params["eta_comm"]
    if np.isnan(eta_comm):
        return float("nan")
    return (
        PCMU_WEIGHTS["delta_m"]  * (delta_m  - mu_dm) / s_dm +
        PCMU_WEIGHTS["eta_priv"] * (eta_priv - mu_ep) / s_ep +
        PCMU_WEIGHTS["eta_comm"] * (eta_comm - mu_ec) / s_ec
        - cen_raw + 1.0
    )


def _eta_priv(aurocs: dict[str, float], ca: dict[str, float], active: dict[str, bool]) -> float:
    total_w = sum(W[t] for t in W if active.get(t, False))
    if total_w == 0:
        return float("nan")
    return sum(
        (W[t] / total_w) * (aurocs.get(t, 0.0) / ca[t])
        for t in W
        if active.get(t, False) and ca.get(t, 0.0) > 0
    )


def _delta_m(aurocs: dict[str, float], st: dict[str, float], active: dict[str, bool]) -> float:
    active_tasks = [t for t in W if active.get(t, False) and st.get(t, 0.0) > 0]
    if len(active_tasks) <= 1:
        return 0.0
    T = len(active_tasks)
    return sum(
        W[t] * (aurocs.get(t, 0.0) - st[t]) / st[t]
        for t in active_tasks
    ) / T


def _eta_comm(R_mtl: float, R_st: float) -> float:
    if R_mtl <= 0 or np.isnan(R_mtl):
        return float("nan")
    return float(np.log(1.0 + R_st / R_mtl))


# ---------------------------------------------------------------------------
# Build R_ST lookup: last round of ihm_only per seed in exp3
# ---------------------------------------------------------------------------

def build_r_st(exp3_path: str) -> dict[int, float]:
    exp3 = pd.read_csv(exp3_path)
    ihm_only = exp3[exp3["task_config"] == "ihm_only"]
    return ihm_only.groupby("seed")["round"].max().to_dict()


# ---------------------------------------------------------------------------
# Process exp1: VFL-MTL vs. single-task models at ε=∞
# ---------------------------------------------------------------------------

def process_exp1(
    exp1_path: str,
    ca: dict[str, float],
    st: dict[str, float],
    r_st: dict[int, float],
    z_params: dict,
    cen_raw: float,
) -> pd.DataFrame:
    df    = pd.read_csv(exp1_path)
    rows  = []

    # Active tasks per model: single-task models only have their own task
    task_active_map = {
        "VFL-MTL":   {"ihm": True, "decomp": True, "pheno": True},
        "ST-IHM":    {"ihm": True, "decomp": False, "pheno": False},
        "ST-Decomp": {"ihm": False, "decomp": True, "pheno": False},
        "ST-Pheno":  {"ihm": False, "decomp": False, "pheno": True},
    }

    for (model, seed), grp in df.groupby(["model", "seed"]):
        peak = grp[list(_TASK_COLS.values())].max()
        aurocs = {t: float(peak[col]) for t, col in _TASK_COLS.items()}
        active = task_active_map.get(model, {"ihm": True, "decomp": True, "pheno": True})

        R_mtl  = float(grp["round"].max())
        R_st   = float(r_st.get(int(seed), np.median(list(r_st.values()))))

        dm  = _delta_m(aurocs, st, active)
        ep  = _eta_priv(aurocs, ca, active)
        ec  = _eta_comm(R_mtl, R_st)
        pv  = pcmu_from_components(dm, ep, ec, z_params, cen_raw)

        rows.append({
            "source":                  "exp1",
            "config":                  model,
            "seed":                    int(seed),
            "epsilon":                 float("inf"),
            "delta_m":                 dm,
            "eta_priv":                ep,
            "eta_comm":                ec,
            "pcmu":                    pv,
            "val_ihm_auroc":           aurocs["ihm"],
            "val_decomp_auroc":        aurocs["decomp"],
            "val_pheno_macro_auroc":   aurocs["pheno"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Process exp3: task configs (MTL gain analysis)
# ---------------------------------------------------------------------------

def process_exp3(
    exp3_path: str,
    ca: dict[str, float],
    st: dict[str, float],
    r_st: dict[int, float],
    z_params: dict,
    cen_raw: float,
) -> pd.DataFrame:
    df   = pd.read_csv(exp3_path)
    rows = []

    task_active_map = {
        "all_tasks":  {"ihm": True, "decomp": True, "pheno": True},
        "ihm_only":   {"ihm": True, "decomp": False, "pheno": False},
        "ihm_decomp": {"ihm": True, "decomp": True, "pheno": False},
        "ihm_pheno":  {"ihm": True, "decomp": False, "pheno": True},
    }

    for (config, seed), grp in df.groupby(["task_config", "seed"]):
        peak   = grp[list(_TASK_COLS.values())].max()
        aurocs = {t: float(peak[col]) for t, col in _TASK_COLS.items()}
        active = task_active_map.get(config, {"ihm": True, "decomp": True, "pheno": True})

        R_mtl = float(grp["round"].max())
        R_st  = float(r_st.get(int(seed), np.median(list(r_st.values()))))

        dm  = _delta_m(aurocs, st, active)
        ep  = _eta_priv(aurocs, ca, active)
        ec  = _eta_comm(R_mtl, R_st)
        pv  = pcmu_from_components(dm, ep, ec, z_params, cen_raw)

        rows.append({
            "source":                  "exp3",
            "config":                  config,
            "seed":                    int(seed),
            "epsilon":                 float("inf"),
            "delta_m":                 dm,
            "eta_priv":                ep,
            "eta_comm":                ec,
            "pcmu":                    pv,
            "val_ihm_auroc":           aurocs["ihm"],
            "val_decomp_auroc":        aurocs["decomp"],
            "val_pheno_macro_auroc":   aurocs["pheno"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Process privacy_utility_combined: ε sweep
# ---------------------------------------------------------------------------

def process_puc(
    puc_path: str,
    ca: dict[str, float],
    st: dict[str, float],
    r_st: dict[int, float],
    z_params: dict,
    cen_raw: float,
) -> pd.DataFrame:
    df   = pd.read_csv(puc_path)
    rows = []
    active = {"ihm": True, "decomp": True, "pheno": True}

    for (eps, seed), grp in df.groupby(["epsilon_level", "seed"]):
        peak   = grp[list(_TASK_COLS.values())].max()
        aurocs = {t: float(peak[col]) for t, col in _TASK_COLS.items()}

        R_mtl = float(grp["round"].max())
        R_st  = float(r_st.get(int(seed), np.median(list(r_st.values()))))

        dm  = _delta_m(aurocs, st, active)
        ep  = _eta_priv(aurocs, ca, active)
        ec  = _eta_comm(R_mtl, R_st)
        pv  = pcmu_from_components(dm, ep, ec, z_params, cen_raw)

        rows.append({
            "source":                  "privacy",
            "config":                  f"eps_{eps}",
            "seed":                    int(seed),
            "epsilon":                 float(eps),
            "delta_m":                 dm,
            "eta_priv":                ep,
            "eta_comm":                ec,
            "pcmu":                    pv,
            "val_ihm_auroc":           aurocs["ihm"],
            "val_decomp_auroc":        aurocs["decomp"],
            "val_pheno_macro_auroc":   aurocs["pheno"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Process dp_ablations: uniform vs. stratified, task pairs, embed × DP
# ---------------------------------------------------------------------------

def process_dp_ablations(
    dpa_path: str,
    ca: dict[str, float],
    st: dict[str, float],
    z_params: dict,
    cen_raw: float,
) -> pd.DataFrame:
    df   = pd.read_csv(dpa_path)
    rows = []

    abl2_active = {
        "ihm_decomp": {"ihm": True, "decomp": True, "pheno": False},
        "ihm_pheno":  {"ihm": True, "decomp": False, "pheno": True},
    }

    eps_map = {
        "uniform_eps5":   5.0,  "stratified_eps5": 5.0,
        "ihm_decomp":     float("inf"), "ihm_pheno": float("inf"),
        "embed32_eps1":   1.0, "embed32_eps5":  5.0, "embed32_epsinf":  float("inf"),
        "embed64_eps1":   1.0, "embed64_eps5":  5.0, "embed64_epsinf":  float("inf"),
        "embed128_eps1":  1.0, "embed128_eps5": 5.0, "embed128_epsinf": float("inf"),
    }

    for _, r in df.iterrows():
        abl    = str(r["ablation"])
        config = str(r["config"])
        seed   = int(r["seed"])
        aurocs = {
            "ihm":    float(r["val_ihm_auroc"]),
            "decomp": float(r["val_decomp_auroc"]),
            "pheno":  float(r["val_pheno_macro_auroc"]),
        }
        active = abl2_active.get(config, {"ihm": True, "decomp": True, "pheno": True})

        dm  = _delta_m(aurocs, st, active)
        ep  = _eta_priv(aurocs, ca, active)
        pv  = pcmu_from_components(dm, ep, float("nan"), z_params, cen_raw)

        rows.append({
            "source":                  f"dp_ablations_{abl}",
            "config":                  config,
            "seed":                    seed,
            "epsilon":                 eps_map.get(config, float("nan")),
            "delta_m":                 dm,
            "eta_priv":                ep,
            "eta_comm":                float("nan"),
            "pcmu":                    pv,
            "val_ihm_auroc":           aurocs["ihm"],
            "val_decomp_auroc":        aurocs["decomp"],
            "val_pheno_macro_auroc":   aurocs["pheno"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table: mean ± std across seeds per (source, config)
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["delta_m", "eta_priv", "eta_comm", "pcmu",
                "val_ihm_auroc", "val_decomp_auroc", "val_pheno_macro_auroc"]
    agg = df.groupby(["source", "config", "epsilon"])[num_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading z-score parameters from factorial...")
    z_params, cen_raw, R_ref_global = load_zparams(FACTORIAL_PATH, CENTRALIZED_PATH, EXP1_PATH)
    print(f"  z_params loaded. Centralized anchor (pre-shift): {cen_raw:.4f}")
    for k, (mu, s) in z_params.items():
        print(f"  {k}: mean={mu:.4f}  std={s:.4f}")

    print("\nLoading centralized AUROCs and ST baselines...")
    cen_df = pd.read_csv(CENTRALIZED_PATH)
    exp1   = pd.read_csv(EXP1_PATH)
    ca     = load_centralized_aurocs(cen_df)
    st     = load_st_aurocs(exp1)
    print(f"  Centralized AUROCs: {ca}")
    print(f"  ST baselines: {st}")

    print("\nBuilding R_ST lookup from exp3 ihm_only...")
    r_st = build_r_st(EXP3_PATH)
    print(f"  R_ST per seed: {r_st}")

    print("\nProcessing exp1...")
    df_exp1 = process_exp1(EXP1_PATH, ca, st, r_st, z_params, cen_raw)

    print("Processing exp3...")
    df_exp3 = process_exp3(EXP3_PATH, ca, st, r_st, z_params, cen_raw)

    print("Processing privacy_utility_combined...")
    df_puc = process_puc(PUC_PATH, ca, st, r_st, z_params, cen_raw)

    print("Processing dp_ablations...")
    df_dpa = process_dp_ablations(DPA_PATH, ca, st, z_params, cen_raw)

    all_results = pd.concat([df_exp1, df_exp3, df_puc, df_dpa], ignore_index=True)
    all_results.to_csv(OUT_PATH.replace(".csv", "_raw.csv"), index=False)
    print(f"\nRaw per-seed results saved -> {OUT_PATH.replace('.csv', '_raw.csv')}")

    summary = make_summary(all_results)
    summary.to_csv(OUT_PATH, index=False)
    print(f"Summary (mean+std across seeds) saved -> {OUT_PATH}")

    print("\n--- Paper 1: exp1 PCMU summary ---")
    s1 = summary[summary["source"] == "exp1"][["config", "pcmu_mean", "pcmu_std",
                                                "val_ihm_auroc_mean", "val_decomp_auroc_mean",
                                                "val_pheno_macro_auroc_mean"]].sort_values("pcmu_mean", ascending=False)
    print(s1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Paper 1: exp3 task-config PCMU summary ---")
    s3 = summary[summary["source"] == "exp3"][["config", "pcmu_mean", "pcmu_std",
                                                "delta_m_mean"]].sort_values("pcmu_mean", ascending=False)
    print(s3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Paper 2: PCMU across ε sweep ---")
    sp = summary[summary["source"] == "privacy"][["epsilon", "pcmu_mean", "pcmu_std",
                                                   "val_ihm_auroc_mean", "val_decomp_auroc_mean",
                                                   "val_pheno_macro_auroc_mean"]].sort_values("epsilon", ascending=False)
    print(sp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
