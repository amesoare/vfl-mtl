"""
experiments/run_eicu_exp1.py — Exp 1 on eICU: VFL-MTL + single-task variants.

Mirrors run_exp1.py with dataset="eicu" and decomp as RLOS regression.
Output columns: model, seed, round, train_loss, ihm_loss, decomp_loss, pheno_loss,
                val_ihm_auroc, val_ihm_auprc, val_rlos_mae, val_rlos_rmse,
                val_pheno_macro_auroc, elapsed_s

Usage:
    python experiments/run_eicu_exp1.py \
        --root /home/asoare/vfl_mlt \
        --n_rounds 100 --patience 15 --device cuda \
        --output results/eicu_exp1.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import TrainConfig, run_training
from data_prep.dataset import build_site_loaders

SEEDS = [42, 123, 7]

CONFIGS = {
    "eICU_VFL-MTL": {
        "task_weights":        {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0},
        "uncertainty_weighting": True,
    },
    "eICU_ST-IHM": {
        "task_weights": {"ihm": 1.0, "decomp": 0.0, "pheno": 0.0},
    },
    "eICU_ST-RLOS": {
        "task_weights": {"ihm": 0.0, "decomp": 1.0, "pheno": 0.0},
    },
    "eICU_ST-Pheno": {
        "task_weights": {"ihm": 0.0, "decomp": 0.0, "pheno": 1.0},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",        default=".")
    parser.add_argument("--n_rounds",    type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--device",      default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--output",      default="results/eicu_exp1.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--patience",    type=int,   default=15)
    args = parser.parse_args()
    if args.use_synthetic:
        _p = Path(args.output); args.output = str(_p.parent / f"smoketest_{_p.name}")

    if args.use_synthetic:
        prebuilt = None
    else:
        print("[eicu_exp1] Pre-loading data loaders...")
        project_root = Path(args.root)
        prebuilt = {
            "train": build_site_loaders(project_root, "train", args.batch_size,
                                        dataset="eicu"),
            "val":   build_site_loaders(project_root, "val",   args.batch_size,
                                        dataset="eicu"),
            "decomp_pos_weight": 1.0,  # RLOS regression; pos_weight unused
        }
        print("[eicu_exp1] Data loaded.")

    all_rows = []
    for model_name, model_cfg in CONFIGS.items():
        for seed in SEEDS:
            print(f"\n=== {model_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir       = str(Path(args.root) / "data" / "eicu_vertical_splits"),
                dataset          = "eicu",
                task_types       = {"decomp": "regression"},
                n_rounds         = args.n_rounds,
                batch_size       = args.batch_size,
                device           = args.device,
                seed             = seed,
                use_fedavg       = True,
                fedavg_every     = 5,
                use_synthetic    = args.use_synthetic,
                model_name       = model_name,
                ckpt_dir         = str(Path(args.root) / "checkpoints" / "eicu"),
                patience         = args.patience,
                decomp_pos_weight = 1.0,
                **model_cfg,
            )
            results = run_training(cfg, prebuilt_loaders=prebuilt)
            for r in results:
                all_rows.append({"model": model_name, "seed": seed, **r})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\neICU Exp 1 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
