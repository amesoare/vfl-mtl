"""
experiments/run_eicu_baselines.py — eICU baselines: local-only + centralized oracle.

Mirrors run_baselines.py for eICU. Handles RLOS regression at Site B.

Usage:
    # Skip rerun, just aggregate from pre-computed CSVs:
    python experiments/run_eicu_baselines.py \
        --skip_rerun \
        --local_a  results/eicu_local_only_A.csv \
        --local_b  results/eicu_local_only_B.csv \
        --local_c  results/eicu_local_only_C.csv \
        --central  results/eicu_centralized.csv \
        --vfl_results results/eicu_exp1.csv

    # Full run:
    python experiments/run_eicu_baselines.py \
        --root /home/asoare/vfl_mlt \
        --n_epochs 50 \
        --vfl_results results/eicu_exp1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.local_only import train_local
from baselines.centralized import train_centralized

SEEDS = [42, 123, 7]


def _extract_local_eicu(path: str, site: str) -> list[dict]:
    df = pd.read_csv(path)
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("epoch").iloc[-1]
        row  = {"model": f"local_{site}", "seed": int(seed),
                "ihm_auroc": float("nan"), "ihm_auprc": float("nan"),
                "rlos_mae":  float("nan"), "rlos_rmse": float("nan"),
                "pheno_macro_auroc": float("nan")}
        if site == "A":
            row["ihm_auroc"] = float(last.get("val_auc_roc", float("nan")))
            row["ihm_auprc"] = float(last.get("val_auc_pr",  float("nan")))
        elif site == "B":
            row["rlos_mae"]  = float(last.get("val_mae",  float("nan")))
            row["rlos_rmse"] = float(last.get("val_rmse", float("nan")))
        elif site == "C":
            row["pheno_macro_auroc"] = float(last.get("val_macro_auc", float("nan")))
        rows.append(row)
    return rows


def _extract_centralized_eicu(path: str) -> list[dict]:
    df = pd.read_csv(path)
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("epoch").iloc[-1]
        rows.append({
            "model":             "centralized_oracle",
            "seed":              int(seed),
            "ihm_auroc":         float(last.get("val_ihm_auc_roc",    float("nan"))),
            "ihm_auprc":         float(last.get("val_ihm_auc_pr",     float("nan"))),
            "rlos_mae":          float(last.get("val_rlos_mae",       float("nan"))),
            "rlos_rmse":         float(last.get("val_rlos_rmse",      float("nan"))),
            "pheno_macro_auroc": float(last.get("val_pheno_macro_auc",float("nan"))),
        })
    return rows


def _extract_vfl_mtl_eicu(path: str) -> list[dict]:
    df   = pd.read_csv(path)
    df   = df[df["model"].isin(["eICU_VFL-MTL", "VFL-MTL"])].copy()
    if df.empty:
        raise ValueError(f"No VFL-MTL rows in {path}. Run run_eicu_exp1.py first.")
    rows = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("round").iloc[-1]
        rows.append({
            "model":             "eICU_VFL-MTL",
            "seed":              int(seed),
            "ihm_auroc":         float(last.get("val_ihm_auroc",         float("nan"))),
            "ihm_auprc":         float(last.get("val_ihm_auprc",         float("nan"))),
            "rlos_mae":          float(last.get("val_rlos_mae",           float("nan"))),
            "rlos_rmse":         float(last.get("val_rlos_rmse",          float("nan"))),
            "pheno_macro_auroc": float(last.get("val_pheno_macro_auroc", float("nan"))),
        })
    return rows


def _rows_to_summary(rows: list[dict], model_name: str, site: str) -> list[dict]:
    df  = pd.DataFrame(rows)
    out = []
    for seed, grp in df.groupby("seed"):
        last = grp.sort_values("epoch").iloc[-1]
        row  = {"model": model_name, "seed": int(seed),
                "ihm_auroc": float("nan"), "ihm_auprc": float("nan"),
                "rlos_mae":  float("nan"), "rlos_rmse": float("nan"),
                "pheno_macro_auroc": float("nan")}
        if site == "A":
            row["ihm_auroc"] = float(last.get("val_auc_roc", float("nan")))
            row["ihm_auprc"] = float(last.get("val_auc_pr",  float("nan")))
        elif site == "B":
            row["rlos_mae"]  = float(last.get("val_mae",  float("nan")))
            row["rlos_rmse"] = float(last.get("val_rmse", float("nan")))
        elif site == "C":
            row["pheno_macro_auroc"] = float(last.get("val_macro_auc", float("nan")))
        elif site == "central":
            row["ihm_auroc"]         = float(last.get("val_ihm_auc_roc",     float("nan")))
            row["ihm_auprc"]         = float(last.get("val_ihm_auc_pr",      float("nan")))
            row["rlos_mae"]          = float(last.get("val_rlos_mae",         float("nan")))
            row["rlos_rmse"]         = float(last.get("val_rlos_rmse",        float("nan")))
            row["pheno_macro_auroc"] = float(last.get("val_pheno_macro_auc",  float("nan")))
        out.append(row)
    return out


def _print_summary(all_rows: list[dict]) -> None:
    df      = pd.DataFrame(all_rows)
    metrics = ["ihm_auroc", "rlos_mae", "pheno_macro_auroc"]
    header  = f"{'Model':<22}" + "".join(f"{m:>20}" for m in metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model, grp in df.groupby("model", sort=False):
        vals = []
        for m in metrics:
            col = grp[m].dropna()
            vals.append("     —     " if col.empty
                        else f"{col.mean():.4f}±{col.std():.4f}")
        print(f"{model:<22}" + "".join(f"{v:>20}" for v in vals))
    print("─" * len(header))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",         default=".")
    p.add_argument("--n_epochs",     type=int,  default=50)
    p.add_argument("--batch_size",   type=int,  default=64)
    p.add_argument("--use_synthetic",action="store_true")
    p.add_argument("--skip_rerun",   action="store_true")
    p.add_argument("--local_a",      default="results/eicu_local_only_A.csv")
    p.add_argument("--local_b",      default="results/eicu_local_only_B.csv")
    p.add_argument("--local_c",      default="results/eicu_local_only_C.csv")
    p.add_argument("--central",      default="results/eicu_centralized.csv")
    p.add_argument("--vfl_results",  default="results/eicu_exp1.csv")
    p.add_argument("--output",       default="results/eicu_baselines_comparison.csv")
    args = p.parse_args()
    if args.use_synthetic:
        _p = Path(args.output); args.output = str(_p.parent / f"smoketest_{_p.name}")

    all_rows: list[dict] = []

    if args.skip_rerun:
        all_rows.extend(_extract_local_eicu(args.local_a, "A"))
        all_rows.extend(_extract_local_eicu(args.local_b, "B"))
        all_rows.extend(_extract_local_eicu(args.local_c, "C"))
        all_rows.extend(_extract_centralized_eicu(args.central))
    else:
        la, lb, lc, cent = [], [], [], []
        for site, store in [("A", la), ("B", lb), ("C", lc)]:
            for seed in SEEDS:
                store.extend(train_local(
                    site=site, root=args.root, n_epochs=args.n_epochs,
                    lr=1e-3, batch_size=args.batch_size, seed=seed,
                    use_synthetic=args.use_synthetic, dataset="eicu"))
        for seed in SEEDS:
            cent.extend(train_centralized(
                root=args.root, n_epochs=args.n_epochs, lr=1e-3,
                batch_size=args.batch_size, seed=seed,
                use_synthetic=args.use_synthetic, dataset="eicu"))
        all_rows.extend(_rows_to_summary(la,   "local_A",           "A"))
        all_rows.extend(_rows_to_summary(lb,   "local_B",           "B"))
        all_rows.extend(_rows_to_summary(lc,   "local_C",           "C"))
        all_rows.extend(_rows_to_summary(cent, "centralized_oracle","central"))

    vfl_path = Path(args.vfl_results)
    if vfl_path.exists():
        all_rows.extend(_extract_vfl_mtl_eicu(str(vfl_path)))
    else:
        print(f"[WARNING] {vfl_path} not found — run run_eicu_exp1.py first.")

    _print_summary(all_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "ihm_auroc", "ihm_auprc",
              "rlos_mae", "rlos_rmse", "pheno_macro_auroc"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\neICU baselines comparison → {args.output}")


if __name__ == "__main__":
    main()
