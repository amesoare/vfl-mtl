# PRISM-VFL

**Privacy-Resilient Integrated System for Multi-task Vertical Federated Learning**

A framework for vertical federated multi-task learning (VFL-MTL) on clinical time-series data, with differential privacy and embedding-space privacy attack evaluation. Primary experiments use the MIMIC-III benchmark with a three-site vertical feature partition; eICU-CRD is used for external validity.

---

## Contents

```
vfl_mlt/
├── train.py                     # Main training entry point (MIMIC-III and eICU)
├── data/
│   ├── DATA.md                  # Instructions for MIMIC-III and eICU access and preprocessing
│   └── vertical_splits/         # Generated vertical split CSVs (see data/DATA.md)
├── data_prep/
│   ├── vertical_split.py        # MIMIC-III vertical partition into three sites
│   ├── psi_alignment.py         # MIMIC-III patient set intersection and cohort alignment
│   ├── eicu_vertical_split.py   # eICU vertical partition into three sites
│   ├── eicu_psi_alignment.py    # eICU patient set intersection and cohort alignment
│   ├── dataset.py               # PyTorch DataLoader builder
│   ├── verify_workspace.py      # Workspace integrity checks
│   └── eicu_benchmark/          # eICU data extraction pipeline
│       └── data_extraction/     # Per-patient timeseries extraction scripts
├── model/
│   ├── encoder.py               # Per-site LSTM encoder (cut layer)
│   └── mmoe.py                  # Server-side MMoE with per-task heads
├── fl/
│   ├── client.py                # VFL client: embedding send and gradient receive
│   ├── server.py                # VFL server: aggregation, loss, gradient similarity
│   ├── fedavg.py                # FedAvg encoder aggregation
│   └── fedprox.py               # FedProx proximal regularisation
├── privacy/
│   ├── adaptive_dpsgd.py        # Per-task DP-SGD gradient clipping and noise
│   └── renyi_accountant.py      # Renyi DP accounting and coupling matrix
├── attacks/
│   ├── label_inference.py       # Logistic probe on cut-layer embeddings
│   └── embedding_mia.py         # Membership inference on embeddings
├── baselines/
│   ├── centralized.py           # Centralised oracle
│   └── local_only.py            # Per-site local-only baselines
├── experiments/
│   ├── run_exp1.py              # Exp 1: VFL-MTL vs single-task baselines (MIMIC-III)
│   ├── run_exp2.py              # Exp 2: task relatedness and negative transfer
│   ├── run_exp3.py              # Exp 3: scalability (n_sites in {2, 3})
│   ├── run_ablations.py         # Architecture ablations (MMoE, gating, embed_dim)
│   ├── run_baselines.py         # MIMIC-III centralized and local-only baselines
│   ├── run_eicu_exp1.py         # Exp 1 replicated on eICU
│   ├── run_eicu_baselines.py    # eICU centralized and local-only baselines
│   ├── privacy_utility_curves.py# Epsilon sweep with Renyi accounting
│   ├── ablations_dp.py          # Uniform vs. stratified noise ablations
│   ├── validate_bound.py        # Multi-task label inference bound validation
│   ├── metrics.py               # Per-task metric helpers
│   ├── evaluate_test.py         # Test-set evaluation for main VFL-MTL results
│   ├── evaluate_test_dp.py      # Test-set evaluation for DP runs
│   ├── evaluate_test_ablations_dp.py # Test-set evaluation for DP ablations
│   ├── evaluate_ablations.py    # Test-set evaluation for architecture ablations
│   ├── evaluate_exp2.py         # Test-set evaluation for Exp 2
│   ├── evaluate_exp3.py         # Test-set evaluation for Exp 3
│   └── evaluate_eicu_test.py    # Test-set evaluation for eICU runs
├── figures/
│   ├── plot_baselines.py        # Baselines comparison figures
│   ├── negative_transfer_heatmap.py # Task relatedness heatmap
│   ├── scalability_curves.py    # Scalability curves
│   ├── resilience_variance.py   # std(AUC) vs. epsilon
│   ├── privacy_utility_plot.py  # AUC vs. epsilon per task
│   ├── plot_ablations.py        # Architecture ablations
│   ├── plot_ablations_dp.py     # DP ablations
│   ├── bound_validation.py      # Multi-task label inference bound
│   ├── plot_eicu_baselines.py   # eICU baselines figures
│   ├── plot_eicu.py             # eICU privacy-utility figures
│   ├── plot_results_summary.py  # Aggregated results summary
│   ├── plot_task_config_seeds.py# Per-seed task configuration plots
│   └── run_figures.sh           # Convenience script to run all figure scripts
├── plots/                       # Generated figures (PNG output)
├── results/                     # Experiment result CSVs
├── checkpoints/                 # Saved model checkpoints (generated during training)
├── logs/                        # SLURM job logs (generated during training)
├── slurm/
│   ├── README.md                # Submission order and cluster notes
│   ├── preprocess_mimic.sh
│   ├── run_local_baselines.sh
│   ├── run_centralized_gpu.sh
│   ├── run_exp1.sh
│   ├── run_exp2.sh
│   ├── run_exp3.sh
│   ├── run_ablations.sh
│   ├── run_evaluate_test.sh
│   ├── run_evaluate_exp2.sh
│   ├── run_evaluate_exp3.sh
│   ├── run_evaluate_ablations.sh
│   ├── run_privacy_curves_eps{05,1,2,5,10}.sh
│   ├── run_ablations_dp.sh
│   ├── run_evaluate_test_dp.sh
│   ├── run_evaluate_test_ablations_dp.sh
│   ├── run_attacks.sh
│   ├── run_bound_validation.sh
│   ├── run_eicu_download.sh
│   ├── run_eicu_extraction.sh
│   ├── run_eicu_vertical_split.sh
│   ├── run_eicu_exp1.sh
│   ├── run_eicu_baselines.sh
│   ├── run_eicu_privacy_curves.sh
│   ├── run_eicu_evaluate_test.sh
│   └── run_eicu_evaluate_test_dp.sh
├── tests/
│   ├── test_fedavg.py
│   ├── test_privacy.py
│   ├── test_centralized.py
│   ├── test_local_only.py
│   ├── test_mimic_integration.py
│   ├── test_eicu_integration.py
│   ├── test_eicu_model_integration.py
│   └── check_model_reliability.py
├── copy_to_snellius.sh
├── sync_with_snellius.sh
└── requirements.txt
```

---

## Requirements

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.2.0`, `opacus>=1.4.0`, `scikit-learn>=1.4`, `scikit-multilearn>=0.2.0`.

For GPU training, install a CUDA-compatible build of PyTorch before running `pip install -r requirements.txt`.

---

## Data Setup

Experiments require MIMIC-III (primary) and eICU-CRD (external validity). Both are hosted on PhysioNet under a Data Use Agreement and are not included in this repository. Full setup instructions are in **`data/DATA.md`**.

Access requires completing CITI training and signing the relevant Data Use Agreement at physionet.org.

### MIMIC-III Clinical Database v1.4

#### Step 1: Download

```bash
mkdir -p data/mimic-iii-clinical-database-1.4
wget -r -N -c -np \
    --user USERNAME --ask-password \
    -P data/ \
    https://physionet.org/files/mimiciii/1.4/
```

On Snellius, use `sbatch slurm/preprocess_mimic.sh` instead.

#### Step 2: Run the YerevaNN benchmark pipeline

```bash
git clone https://github.com/YerevaNN/mimic3-benchmarks.git

MIMIC_RAW=data/mimic-iii-clinical-database-1.4
BENCH_OUT=data/mimic3-benchmarks/data

python -m mimic3benchmark.scripts.extract_subjects $MIMIC_RAW $BENCH_OUT/root/
python -m mimic3benchmark.scripts.validate_events $BENCH_OUT/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects $BENCH_OUT/root/
python -m mimic3benchmark.scripts.split_train_and_test $BENCH_OUT/root/

python -m mimic3benchmark.scripts.create_in_hospital_mortality \
    $BENCH_OUT/root/ $BENCH_OUT/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation \
    $BENCH_OUT/root/ $BENCH_OUT/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay \
    $BENCH_OUT/root/ $BENCH_OUT/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping \
    $BENCH_OUT/root/ $BENCH_OUT/phenotyping/
```

#### Step 3: Create the vertical split

```bash
python data_prep/vertical_split.py \
    --root data/mimic3-benchmarks/data/ \
    --output data/vertical_splits/
```

| File | Features | Task label |
|------|----------|------------|
| `site_A_vitals.csv` | HR, SBP, DBP, Temp, SpO2, RespRate, GCS Total (7 vars) | In-hospital mortality (binary, 48 h) |
| `site_B_labs.csv` | Glucose, pH, FiO2, CapRefill (4 vars) | Decompensation (binary, 24 h) |
| `site_C_composite.csv` | Height, Weight, MeanBP (3 vars) | Phenotyping (25 ICD-9 codes, multi-label) |

#### Step 4: Align patient sets

```bash
python data_prep/psi_alignment.py \
    --site_a data/vertical_splits/site_A_vitals.csv \
    --site_b data/vertical_splits/site_B_labs.csv \
    --site_c data/vertical_splits/site_C_composite.csv \
    --output data/vertical_splits/aligned_patient_ids.csv
```

#### Step 5: Verify

```bash
python data_prep/verify_workspace.py
```

---

### eICU Collaborative Research Database v2.0

#### Step 1: Download

```bash
mkdir -p data/eicu-crd-2.0
wget -r -N -c -np \
    --user USERNAME --ask-password \
    -P data/ \
    https://physionet.org/files/eicu-crd/2.0/
```

On Snellius, use `sbatch slurm/run_eicu_download.sh` instead.

#### Step 2: Extract per-patient timeseries

```bash
mkdir -p data/eicu_root
python data_prep/eicu_benchmark/data_extraction/data_extraction_root.py \
    --eicu_dir  data/eicu-crd-2.0/ \
    --output_dir data/eicu_root/
```

On Snellius, use `sbatch slurm/run_eicu_extraction.sh`.

#### Step 3: Create the vertical split

```bash
python data_prep/eicu_vertical_split.py \
    --root_dir data/eicu_root/ \
    --eicu_dir data/eicu-crd-2.0/ \
    --output   data/eicu_vertical_splits/ \
    --seed 42
```

| File | Features | Task label |
|------|----------|------------|
| `site_A_eicu.csv` | HR, SBP, DBP, MAP, SpO2, RespRate, Temp (7 vars) | In-hospital mortality (binary) |
| `site_B_eicu.csv` | Glucose, pH, FiO2 (3 vars) | Remaining length of stay (regression, days) |
| `site_C_eicu.csv` | GCS Total, Height, Weight (3 vars) | Phenotyping (25 ICD-9 codes, multi-label) |

#### Step 4: Align patient sets

```bash
python data_prep/eicu_psi_alignment.py \
    --site_a data/eicu_vertical_splits/site_A_eicu.csv \
    --site_b data/eicu_vertical_splits/site_B_eicu.csv \
    --site_c data/eicu_vertical_splits/site_C_eicu.csv \
    --output data/eicu_vertical_splits/aligned_patient_ids_eicu.csv
```

On Snellius, Steps 3 and 4 are combined in `sbatch slurm/run_eicu_vertical_split.sh`.

---

## Training

### Basic run

```bash
python train.py --root . --rounds 50 --seed 42
```

### eICU

```bash
python train.py --root . --dataset eicu --rounds 100 --seed 42
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mimic` | Dataset: `mimic` or `eicu` |
| `--rounds` | 50 | Training rounds |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Adam learning rate for all components |
| `--hidden-dim` | 128 | LSTM hidden size |
| `--embed-dim` | 64 | Cut-layer embedding dimension |
| `--num-experts` | 4 | MMoE shared expert count |
| `--fedprox-mu` | 0.0 | FedProx proximal coefficient (0 = disabled) |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | 42 | Random seed |
| `--use-synthetic` | off | Smoke-test with random data (no data required) |

All results reported in the paper use seeds 42, 123, and 7.

Differential privacy is configured programmatically via `privacy_config` in the experiment scripts (`privacy_utility_curves.py`, `ablations_dp.py`). See the Privacy Module section below.

---

## Reproducing Experiments

Run all scripts from the repository root (`vfl_mlt/`) after completing data setup.

### MIMIC-III VFL-MTL experiments

```bash
python experiments/run_baselines.py --splits_dir data/vertical_splits

python experiments/run_exp1.py --splits_dir data/vertical_splits --output results/exp1.csv
python experiments/run_exp2.py --splits_dir data/vertical_splits --output results/exp2.csv
python experiments/run_exp3.py --splits_dir data/vertical_splits --output results/exp3.csv

python experiments/run_ablations.py --splits_dir data/vertical_splits \
    --output results/ablations.csv

python experiments/evaluate_test.py --splits_dir data/vertical_splits
python experiments/evaluate_exp2.py --splits_dir data/vertical_splits
python experiments/evaluate_exp3.py --splits_dir data/vertical_splits
python experiments/evaluate_ablations.py --splits_dir data/vertical_splits
```

### Differential privacy experiments

```bash
for EPS in 0.5 1.0 2.0 5.0 10.0; do
    python experiments/privacy_utility_curves.py \
        --splits_dir data/vertical_splits \
        --epsilon $EPS \
        --output results/privacy_utility_combined.csv
done

python experiments/ablations_dp.py \
    --splits_dir data/vertical_splits --output results/dp_ablations.csv

python experiments/evaluate_test_dp.py --splits_dir data/vertical_splits
python experiments/evaluate_test_ablations_dp.py \
    --splits_dir data/vertical_splits --output results/test_ablations_dp.csv

python attacks/label_inference.py \
    --splits_dir data/vertical_splits --ckpt_dir checkpoints \
    --output results/label_inference.csv
python attacks/embedding_mia.py \
    --splits_dir data/vertical_splits --ckpt_dir checkpoints \
    --output results/embedding_mia.csv

python experiments/validate_bound.py \
    --results_dir results --output results/bound_validation.csv
```

### eICU external validity experiments

```bash
python experiments/run_eicu_baselines.py --output results/eicu_baselines.csv
python experiments/run_eicu_exp1.py --output results/eicu_exp1.csv
python experiments/evaluate_eicu_test.py --root .
```

### Running on Snellius (SURF HPC)

SLURM scripts for all experiments are in `slurm/`. See `slurm/README.md` for the recommended submission order.

```bash
mkdir -p logs

sbatch slurm/run_local_baselines.sh
sbatch slurm/run_centralized_gpu.sh
sbatch slurm/run_exp1.sh
sbatch slurm/run_exp2.sh
sbatch slurm/run_exp3.sh
sbatch slurm/run_ablations.sh
sbatch slurm/run_evaluate_test.sh
sbatch slurm/run_evaluate_exp2.sh
sbatch slurm/run_evaluate_exp3.sh
sbatch slurm/run_evaluate_ablations.sh

sbatch slurm/run_privacy_curves_eps10.sh
sbatch slurm/run_privacy_curves_eps5.sh
sbatch slurm/run_privacy_curves_eps2.sh
sbatch slurm/run_privacy_curves_eps1.sh
sbatch slurm/run_privacy_curves_eps05.sh
sbatch slurm/run_ablations_dp.sh
sbatch slurm/run_evaluate_test_dp.sh
sbatch slurm/run_evaluate_test_ablations_dp.sh
sbatch slurm/run_attacks.sh
sbatch slurm/run_bound_validation.sh

sbatch slurm/run_eicu_download.sh
sbatch slurm/run_eicu_extraction.sh
sbatch slurm/run_eicu_vertical_split.sh
sbatch slurm/run_eicu_exp1.sh
sbatch slurm/run_eicu_baselines.sh
sbatch slurm/run_eicu_privacy_curves.sh
sbatch slurm/run_eicu_evaluate_test.sh
sbatch slurm/run_eicu_evaluate_test_dp.sh
```

---

## Figures

All figure scripts read from `results/` and write PNGs to `figures/`. Run after all experiments complete.

```bash
bash figures/run_figures.sh
```

| Script | Output | Source CSV |
|--------|--------|-----------|
| `plot_baselines.py` | Baselines comparison | `results/exp1.csv`, baseline CSVs |
| `negative_transfer_heatmap.py` | Task relatedness heatmap | `results/exp2.csv`, `results/test_exp2.csv` |
| `scalability_curves.py` | Scalability curves | `results/exp3.csv`, `results/test_exp3.csv` |
| `resilience_variance.py` | std(AUC) vs. ε | `results/privacy_utility_combined.csv` |
| `privacy_utility_plot.py` | AUC vs. ε per task | `results/privacy_utility_combined.csv` |
| `plot_ablations.py` | Architecture ablations | `results/ablations.csv`, `results/test_ablations.csv` |
| `plot_ablations_dp.py` | DP ablations | `results/dp_ablations.csv`, `results/test_ablations_dp.csv` |
| `bound_validation.py` | Label inference bound | `results/bound_validation.csv` |
| `plot_eicu_baselines.py` | eICU baselines | `results/eicu_baselines.csv` |
| `plot_eicu.py` | eICU privacy-utility | `results/eicu_privacy_utility_combined.csv` |

---

## Tests

```bash
pytest tests/ -v
```

Tests cover FedAvg aggregation, DP-SGD clipping behaviour, centralized baseline, local-only baseline, and integration smoke tests for both MIMIC-III and eICU.

---

## Vertical Split Protocol

The 17 variables marked STATUS="ready" in the YerevaNN benchmark are partitioned across three simulated hospital sites. Three GCS sub-scores are excluded to prevent feature reconstruction via the GCS total identity. The remaining 14 variables are assigned as follows:

| Site | Variables | Task |
|------|-----------|------|
| A | Heart Rate, Systolic BP, Diastolic BP, Temperature, SpO2, Respiratory Rate, GCS Total | In-hospital mortality |
| B | Glucose, pH, FiO2, Capillary Refill Rate | Decompensation |
| C | Height, Weight, Mean Blood Pressure | Phenotyping (25 ICD-9 codes) |

Each site holds its own task label. The server receives only embedding vectors from the cut layer; no raw features or labels are transmitted.

---

## Architecture

**Per-site client:**
- `SiteEncoder`: LSTM (hidden 128, 2 layers) followed by a linear projection to 64-dimensional embeddings and LayerNorm. Input shape: `(B, T, n_features_at_site)`. Output: `(B, 64)`.

**Server:**
- `MMoEServer`: four shared ExpertMLPs (64-128-64 with ReLU), three per-task softmax gating networks, three task heads (binary sigmoid for IHM and Decompensation; 25-way sigmoid for Phenotyping).
- Input: concatenated site embeddings `(B, 192)`.

**Training protocol:**
1. Each client runs its LSTM encoder and sends a detached 64-dimensional embedding to the server.
2. The server concatenates embeddings, runs MMoE, and computes weighted BCE losses.
3. The server backpropagates and slices the embedding gradient back to each client.
4. Each client applies the gradient to its encoder.
5. Optional FedAvg aggregation every five rounds.

---

## Privacy Module

Differential privacy is implemented via Opacus. The `AdaptiveDPSGD` class supports two modes:

- **Uniform**: one noise multiplier sigma applied to all task gradients.
- **Stratified**: per-task sigma values (sigma_ihm, sigma_decomp, sigma_pheno).

Privacy is configured via `privacy_config` passed to the training function in experiment scripts:

```python
# Uniform noise
privacy_config = {'mode': 'uniform', 'sigma': 1.0, 'max_grad_norm': 1.0, 'delta': 1e-5}

# Task-stratified noise
privacy_config = {'mode': 'stratified', 'sigma_ihm': 0.5, 'sigma_decomp': 1.0,
                  'sigma_pheno': 1.5, 'max_grad_norm': 1.0, 'delta': 1e-5}
```

Privacy accounting uses Renyi DP (Mironov 2017) via `opacus.accountants.RDPAccountant`, with one accountant per task. The `RenyiAccountant` wrapper adds cross-task gradient correlation tracking via `cross_task_coupling_matrix()`.

---