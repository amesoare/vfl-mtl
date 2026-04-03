"""
experiments/run_exp1.py — Exp 1: Task heterogeneity vs. homogeneity.

Compares:
  VFL-MTL      : 3 sites, 3 tasks (ihm + los + pheno)
  VFL-SingleTask: 3 sites, IHM only (los and pheno weights set to 0)

Seeds: [42, 123, 7]
Output: results/exp1.csv
  columns: model, seed, round, train_loss, ihm_loss, los_loss, pheno_loss,
           val_ihm_auroc, val_ihm_auprc, val_los_kappa, val_pheno_macro_auroc,
           elapsed_s

Usage:
    # Real data (on Snellius):
    python experiments/run_exp1.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
        --n_rounds 50 --device cpu

    # Smoke test (local, no data):
    python experiments/run_exp1.py --n_rounds 3 --use_synthetic
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig

SEEDS = [42, 123, 7]

CONFIGS = {
    "VFL-MTL": {
        "task_weights": {"ihm": 1.0, "los": 1.0, "pheno": 1.0},
    },
    # Per-site single-task baselines: each trains on its own site's task only.
    # ST-IHM (Site A), ST-LOS (Site B), ST-Pheno (Site C) together form the
    # local single-task reference — MTL contribution is measured against these.
    "ST-IHM": {
        "task_weights": {"ihm": 1.0, "los": 0.0, "pheno": 0.0},
    },
    "ST-LOS": {
        "task_weights": {"ihm": 0.0, "los": 1.0, "pheno": 0.0},
    },
    "ST-Pheno": {
        "task_weights": {"ihm": 0.0, "los": 0.0, "pheno": 1.0},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--output",        default="results/exp1.csv")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use random synthetic data (smoke test, no real data needed)")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    args = parser.parse_args()

    all_rows = []

    for model_name, model_cfg in CONFIGS.items():
        for seed in SEEDS:
            print(f"\n=== {model_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                use_fedavg=True,
                fedavg_every=5,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
                **model_cfg,
            )
            results = run_training(cfg)
            for r in results:
                all_rows.append({"model": model_name, "seed": seed, **r})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 1 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
