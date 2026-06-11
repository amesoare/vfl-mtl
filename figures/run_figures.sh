#!/usr/bin/env bash
# figures/run_figures.sh — Regenerate all manuscript figures in order.
#
# Run from the repository root:
#   bash figures/run_figures.sh
#
# All PNGs are written to figures/
#
# Output file → content
# ─────────────────────────────────────────────────────────────────────────────
# baselines_test_metrics.png        MIMIC: baseline model comparison (test set)
# baselines_learning_curves.png     MIMIC: validation learning curves
# baselines_val_metrics.png         MIMIC: final-epoch validation lollipop
# mimic_task_relatedness.png        MIMIC: task relatedness (Exp 2, test set)
# negative_transfer.png             MIMIC: negative transfer loss heatmap
# scalability.png                   MIMIC: 2 vs. 3 institutions (Exp 3)
# resilience_variance.png           MIMIC: DP seed variance across ε
# privacy_utility_test.png          MIMIC: privacy-utility curves (test set)
# ablations_test.png                MIMIC: architectural ablation study
# ablations_dp_abl1.png             MIMIC: DP ablation 1 — uniform vs. stratified σ
# ablations_dp_abl2.png             MIMIC: DP ablation 2 — gradient coupling
# ablations_dp_abl3.png             MIMIC: DP ablation 3 — embed dim × ε
# eicu_test_baselines_metrics.png   eICU: baseline model comparison (test set)
# eicu_baselines_val_metrics.png    eICU: baseline model comparison (val)
# eicu_privacy_utility_test.png     eICU: privacy-utility curves (test set)
# eicu_privacy_utility_val.png      eICU: privacy-utility curves (val)
# bound_validation.png              MIMIC: theoretical DP bound vs. empirical attacks
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."   # ensure we are in repo root

mkdir -p figures

echo "=== MIMIC baselines ==="
python3 figures/plot_baselines.py

echo ""
echo "=== MIMIC task relatedness and negative transfer ==="
python3 figures/negative_transfer_heatmap.py
python3 figures/plot_results_summary.py

echo ""
echo "=== MIMIC scalability ==="
python3 figures/scalability_curves.py

echo ""
echo "=== MIMIC DP resilience variance ==="
python3 figures/resilience_variance.py

echo ""
echo "=== MIMIC privacy-utility curves ==="
python3 figures/privacy_utility_plot.py

echo ""
echo "=== MIMIC ablations ==="
python3 figures/plot_ablations.py --source test \
    --input results/test_ablations.csv

echo ""
echo "=== MIMIC DP ablations ==="
python3 figures/plot_ablations_dp.py --abl 1
python3 figures/plot_ablations_dp.py --abl 2
python3 figures/plot_ablations_dp.py --abl 3

echo ""
echo "=== eICU baselines and privacy-utility ==="
python3 figures/plot_eicu.py

echo ""
echo "=== Bound validation ==="
python3 figures/bound_validation.py

echo ""
echo "=== Done — all figures written to figures/ ==="
ls -lh figures/*.png 2>/dev/null || echo "(no PNG files found)"
