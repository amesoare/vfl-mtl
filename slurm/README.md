# SLURM Scripts — Snellius (SURF)

These scripts run VFL-MTL experiments on the Snellius HPC cluster using the
`gpu_a100` partition. Adjust `--account`, `--partition`, and module versions to
match your allocation and cluster configuration.

All scripts assume they are submitted from the `vfl_mlt/` root directory:

```bash
cd /path/to/vfl_mlt
sbatch slurm/<script>.sh
```

---

## Prerequisites

1. Data must be set up (see `data/DATA_SETUP.md`).
2. Create a `logs/` directory before submitting:
   ```bash
   mkdir -p logs
   ```
3. Install Python dependencies once (interactive node or prologue):
   ```bash
   pip install --user opacus>=1.4.0 scikit-multilearn>=0.2.0
   ```

---

## Script overview

**Setup**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `preprocess_mimic.sh` | YerevaNN MIMIC-III preprocessing pipeline (run once) | ~2 h |

**Paper 1 — training**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_local_baselines.sh` | Local-only baselines — array job over sites A/B/C | 180 min |
| `run_centralized_gpu.sh` | Centralized oracle baseline | 60 min |
| `run_exp1.sh` | Exp 1: VFL-MTL vs. single-task baselines (3 seeds) | 45 min |
| `run_exp2.sh` | Exp 2: task relatedness and negative transfer (3 seeds × 4 configs) | 90 min |
| `run_exp3.sh` | Exp 3: scalability — n_sites ∈ {2, 3} (3 seeds each) | 60 min |
| `run_ablations.sh` | Architecture ablations (7 configs × 3 seeds) | 90 min |

**Paper 1 — test-set evaluation**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_evaluate_test.sh` | Test-set inference for Exp 1/2/3 | 20 min |
| `run_evaluate_exp2.sh` | Test-set inference for Exp 2 rerun | 10 min |
| `run_evaluate_exp3.sh` | Test-set inference for Exp 3 rerun | 10 min |
| `run_evaluate_ablations.sh` | Test-set inference for ablation checkpoints | 20 min |

**Paper 2 — DP training**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_privacy_curves_eps10.sh` | ε sweep ε=10 (3 seeds) | 60 min |
| `run_privacy_curves_eps5.sh` | ε sweep ε=5, incl. stratified (3 seeds) | 60 min |
| `run_privacy_curves_eps2.sh` | ε sweep ε=2 (3 seeds) | 60 min |
| `run_privacy_curves_eps1.sh` | ε sweep ε=1 (3 seeds) | 60 min |
| `run_privacy_curves_eps05.sh` | ε sweep ε=0.5 (3 seeds) | 60 min |
| `run_ablations_dp.sh` | DP ablations: uniform vs. stratified σ, embed_dim × ε | 60 min |

**Paper 2 — test-set evaluation and attacks**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_evaluate_test_dp.sh` | Test-set inference for DP ε sweep checkpoints | 20 min |
| `run_evaluate_test_ablations_dp.sh` | Test-set inference for DP ablation checkpoints | 20 min |
| `run_attacks.sh` | Label inference + MIA attacks across ε levels | 30 min |
| `run_bound_validation.sh` | Multi-task label inference bound validation | 20 min |

**External validity (eICU)**

| Script | Purpose | Approx. wall time |
|--------|---------|-------------------|
| `run_eicu_download.sh` | eICU data download from PhysioNet | depends on connection |
| `run_eicu_extraction.sh` | eICU benchmark preprocessing | ~2 h |
| `run_eicu_vertical_split.sh` | eICU vertical split and PSI alignment | 30 min |
| `run_eicu_exp1.sh` | eICU Exp 1 replication (3 seeds) | 45 min |
| `run_eicu_baselines.sh` | eICU local-only and centralized baselines | 60 min |
| `run_eicu_privacy_curves.sh` | eICU ε sweep (6 levels × 3 seeds) | 60 min each |
| `run_eicu_evaluate_test.sh` | eICU test-set inference (no DP) | 20 min |
| `run_eicu_evaluate_test_dp.sh` | eICU test-set inference (DP) | 20 min |

---

## Order of execution

Run in this order to reproduce all results:

1. `preprocess_mimic.sh` (once — data setup only)
2. `run_local_baselines.sh` + `run_centralized_gpu.sh`
3. `run_exp1.sh`, `run_exp2.sh`, `run_exp3.sh`
4. `run_ablations.sh`
5. `run_evaluate_test.sh`, `run_evaluate_exp2.sh`, `run_evaluate_exp3.sh`, `run_evaluate_ablations.sh`
6. `run_privacy_curves_eps{10,5,2,1,05}.sh` (submit all five in parallel)
7. `run_ablations_dp.sh`
8. `run_evaluate_test_dp.sh`, `run_evaluate_test_ablations_dp.sh`
9. `run_attacks.sh` (requires checkpoints from step 6)
10. `run_bound_validation.sh`
11. eICU: `run_eicu_download.sh` → `run_eicu_extraction.sh` → `run_eicu_vertical_split.sh` → `run_eicu_exp1.sh` + `run_eicu_baselines.sh` → `run_eicu_privacy_curves.sh` → `run_eicu_evaluate_test.sh` + `run_eicu_evaluate_test_dp.sh`
