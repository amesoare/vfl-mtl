"""
experiments/run_exp4.py — Exp 4: Scalability (2 vs. 3 institutions).

Measures:
  - Communication rounds to convergence (loss delta < 0.001 over 5 rounds)
  - Wall-clock time per round
  - Per-task AUC at final round

Configurations:
  n_sites=2 : Sites A + B only (IHM + LOS tasks; pheno weight = 0)
  n_sites=3 : All three sites (IHM + LOS + Pheno)

Seeds: [42, 123, 7]
Output: results/exp4.csv

Usage:
    # Real data (on Snellius):
    python experiments/run_exp4.py \
        --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
        --n_rounds 50 --device cpu

    # Smoke test (local, no data):
    python experiments/run_exp4.py --n_rounds 3 --use_synthetic
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig

SEEDS = [42, 123, 7]


def rounds_to_convergence(losses: list[float], threshold: float = 0.001, window: int = 5) -> int:
    """Return round index where total loss delta over `window` rounds drops below threshold."""
    for i in range(window, len(losses)):
        delta = abs(losses[i - window] - losses[i])
        if delta < threshold:
            return i
    return len(losses)  # did not converge within budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--output",        default="results/exp4.csv")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use random synthetic data (smoke test, no real data needed)")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    args = parser.parse_args()

    all_rows = []

    for n_sites in [2, 3]:
        task_weights = (
            {"ihm": 1.0, "los": 1.0, "pheno": 0.0}
            if n_sites == 2
            else {"ihm": 1.0, "los": 1.0, "pheno": 1.0}
        )
        for seed in SEEDS:
            print(f"\n=== n_sites={n_sites} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                n_sites=n_sites,
                task_weights=task_weights,
                use_fedavg=True,
                fedavg_every=5,
                use_synthetic=args.use_synthetic,
                n_synthetic=args.n_synthetic,
            )
            results = run_training(cfg)
            losses = [r["train_loss"] for r in results]
            conv_round = rounds_to_convergence(losses)
            print(f"  Convergence round: {conv_round}/{args.n_rounds}")
            for r in results:
                all_rows.append({
                    "n_sites":          n_sites,
                    "seed":             seed,
                    "convergence_round": conv_round,
                    **r,
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 4 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
