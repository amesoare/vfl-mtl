# VFL-MTL Paper 1 Completion — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete all remaining Paper 1 components — dataset loader, federated aggregation, training loop, four experiments, metric utilities, and result figures — building on the already-implemented encoder, MMoE, client, server, and data-prep pipeline.

**Architecture:** Per-site raw timeseries (T=48, 1-hour binned) are loaded by `VFLSiteDataset` via `build_site_loaders()`; a round-based training loop orchestrates three `VFLClient` encoders and one `VFLServer` with optional FedAvg/FedProx; experiment scripts sweep configurations and write CSVs; figure scripts plot the results.

**Tech Stack:** PyTorch 2.2, scikit-learn (metrics), pandas, matplotlib/seaborn, pytest

---

## Status Audit — What Is Already Done

### ✅ Implemented and complete

- [x] `data_prep/vertical_split.py` — `load_clip_bounds()` reads VALID_LOW/VALID_HIGH from `variable_ranges.csv` at runtime; `clip_features()` applied per-row before ffill/bfill; **Snellius re-run pending**
- [x] `data_prep/psi_alignment.py` — `check_label_balance()` + `stratify_aligned_cohort()` added; iterative stratification across IHM + LOS bins + 25 phenotypes; **Snellius re-run pending**
- [x] `data_prep/verify_workspace.py` — 5-check workspace readiness verifier
- [x] `model/encoder.py` — `SiteEncoder`: LSTM(hidden=128, layers=2) + Linear + LayerNorm → (B, 64)
- [x] `model/mmoe.py` — `MMoEServer`: 4 shared ExpertMLPs + per-task gating + IHM/LOS/Pheno heads
- [x] `fl/client.py` — `VFLClient`: forward → detached embedding; receive_gradient → local backprop; get/set params for FedAvg
- [x] `fl/server.py` — `VFLServer`: aggregate embeddings → MMoE → weighted loss → backward → gradient slices
- [x] `fl/__init__.py` — exports VFLClient, VFLServer
- [x] `fl/fedavg.py` — `fedavg_aggregate()`: weighted average of compatible encoder state dicts; 5 tests passing
- [x] `fl/fedprox.py` — `fedprox_penalty()`: `(mu/2)||w_local - w_global||²`; device-aware; 5 tests passing
- [x] `data_prep/dataset.py` — `VFLSiteDataset` + `build_site_loaders()`; loads raw T=48 timeseries, YerevaNN CustomBins LOS; synced from Snellius
- [x] `tests/test_integration.py` — 4 tests: loss decreases × 3 rounds, embedding shapes, prediction output shapes, encoder weights update
- [x] `tests/test_fedavg.py` — 5 tests: weighted average correctness, param round-trip, penalty zero/positive/mu-scaling
- [x] `requirements.txt` — torch, opacus, numpy, pandas, sklearn, scipy, matplotlib, seaborn, pytest, scikit-multilearn
- [x] `train.py` — CLI + `TrainConfig` dataclass + `run_training()` + FedAvg + `site_input_dims` + `n_sites`
- [x] `experiments/metrics.py` — `ihm_metrics()`, `los_metrics()`, `pheno_metrics()`, `compute_all_metrics()`
- [x] `experiments/run_exp1.py` — task heterogeneity vs. homogeneity; smoke-tested
- [x] `experiments/run_exp2.py` — feature asymmetry; smoke-tested
- [x] `experiments/run_exp3.py` — task relatedness / negative transfer; smoke-tested
- [x] `experiments/run_exp4.py` — scalability (2 vs 3 sites); smoke-tested
- [x] `results/plot_results.py` — `load_results()`, `summary_table()`, `loss_curves()`, `val_metric_curves()`, `comparison_table()`
- [x] `figures/negative_transfer_heatmap.py` — loss delta heatmap vs. IHM-only baseline
- [x] `figures/scalability_curves.py` — convergence bar chart + loss curves by n_sites
- [x] `figures/feature_split_sensitivity.py` — grouped bar chart per split config

### ✅ Data location

Vertical splits confirmed on Snellius at `/home/asoare/vfl_mlt/data/vertical_splits/` (regenerated 2026-03-19, after clipping + stratification fixes):
- `site_A_vitals.csv`  (3.44 MB)
- `site_B_labs.csv`    (5.08 MB)
- `site_C_composite.csv` (5.88 MB)
- `aligned_patient_ids.csv` (204 kB)

For Snellius experiment runs, pass `--splits_dir /home/asoare/vfl_mlt/data/vertical_splits/`.
For local smoke tests, use `--use-synthetic` (flag uses hyphen, not underscore).

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `data_prep/vertical_split.py` | ✅ done | `load_clip_bounds()` + `clip_features()` before ffill/bfill; bounds from `variable_ranges.csv` |
| `data_prep/psi_alignment.py` | ✅ done | `check_label_balance()` + `stratify_aligned_cohort()` after PSI intersection |
| `data_prep/dataset.py` | ✅ done | `VFLSiteDataset` + `build_site_loaders()`; raw T=48 sequences, YerevaNN CustomBins |
| `fl/fedavg.py` | ✅ done | `fedavg_aggregate()`: weighted average of compatible encoder state dicts |
| `fl/fedprox.py` | ✅ done | `fedprox_penalty()`: `(mu/2)||w_local-w_global||²`, device-aware |
| `tests/test_fedavg.py` | ✅ done | 5 tests: weighted avg, round-trip, penalty zero/positive/mu-scaling |
| `train.py` | ✅ done | `run_training()` + `TrainConfig` + FedAvg + `site_input_dims` + `n_sites` |
| `experiments/metrics.py` | ✅ done | `ihm_metrics()`, `los_metrics()`, `pheno_metrics()`, `compute_all_metrics()` |
| `experiments/run_exp1.py` | ✅ done | VFL-MTL vs. VFL-SingleTask; per-task AUC; smoke-tested |
| `experiments/run_exp2.py` | ✅ done | Feature asymmetry: 3 split configs × 3 seeds; smoke-tested |
| `experiments/run_exp3.py` | ✅ done | Task relatedness / negative transfer rate; smoke-tested |
| `experiments/run_exp4.py` | ✅ done | Scalability: 2/3 sites × rounds → AUC + wall-clock; smoke-tested |
| `results/plot_results.py` | ✅ done | `load_results()`, `summary_table()`, `loss_curves()`, `val_metric_curves()`, `comparison_table()` |
| `figures/negative_transfer_heatmap.py` | ✅ done | Loss delta heatmap vs. IHM-only baseline |
| `figures/scalability_curves.py` | ✅ done | Convergence bar chart + loss curves by n_sites |
| `figures/feature_split_sensitivity.py` | ✅ done | Grouped bar chart per split configuration |
| `tests/test_train.py` | ✅ done | Smoke test for run_training() on synthetic data |

---

## Chunk 0: Data Pipeline Fixes (EDA-driven, must complete before Chunk 1)

**Why this chunk exists:** EDA on the current Snellius CSVs revealed two problems that will corrupt model training if left unaddressed: (1) data entry artefacts survive imputation (e.g. SpO2 = 29,818%, Weight = 3,761,721 kg) because clipping was never applied before ffill/bfill/mean-impute; (2) the PSI-aligned cohort split assignment is inherited from YerevaNN random splits without checking whether label balance is preserved through the intersection step, which can silently skew val/test class distributions.

**Fixes:**
- `vertical_split.py`: clip each feature to a clinical hard-limit range **before** ffill/bfill; recompute train means on clipped data
- `psi_alignment.py`: after computing the PSI intersection, check label balance per split; re-assign if imbalance exceeds threshold using stratified splitting

**Dependency:** Chunks 1–4 depend on clean CSVs. Do not run experiments until new CSVs are on Snellius.

---

### Task 0a: Outlier clipping — update `data_prep/vertical_split.py` ✅ DONE

**Approach:** Load `VALID_LOW`/`VALID_HIGH` at runtime from `variable_ranges.csv` via
`load_clip_bounds()`. Applied per-row before ffill/bfill in `aggregate_stays()`.
`extract_episodes_from_subjects.py` now has a TODO comment at the dead
`--reference_range_file` argument explaining the correct upstream fix.

**Why not hardcode bounds:** `variable_ranges.csv` is already the canonical source in
the benchmark. Reading it at runtime avoids duplication and keeps provenance clear.

**Why not fix upstream:** Timeseries CSVs already exist on Snellius. Re-running
`extract_episodes_from_subjects.py` costs hours of compute. `vertical_split.py` is fast
to re-run and is the last step before experiments.

- [x] Add `load_clip_bounds()` — reads `VALID_LOW`/`VALID_HIGH` from `variable_ranges.csv`
- [x] Add `clip_features()` — applies bounds per-row before ffill/bfill
- [x] Wire into `aggregate_stays()` and all three `build_site_*()` functions
- [x] Load bounds once in `main()`, pass through to all site builders
- [x] Add TODO comment in `extract_episodes_from_subjects.py` at dead `--reference_range_file`
- [x] **Re-run `vertical_split.py` on Snellius to overwrite existing CSVs** (done 2026-03-19)

```bash
# on Snellius
python data_prep/vertical_split.py \
    --root /home/asoare/vfl_mlt/mimic3-benchmarks/data/ \
    --output /home/asoare/vfl_mlt/data/
```

- [x] **Commit** (already in git history)

---

### Task 0b: Stratified split assignment — update `data_prep/psi_alignment.py` ✅ DONE

**Approach:** keep YerevaNN train/val/test listfile splits (preserving benchmark comparability). After PSI intersection, verify that IHM mortality rate, LOS bin distribution, and phenotype prevalences are consistent across splits within the aligned cohort. If any label deviates by more than 3 percentage points from the train distribution, re-assign split membership using stratified splitting. Phenotyping uses iterative stratification (`scikit-multilearn`).

- [x] **Step 1: Add `scikit-multilearn` to `requirements.txt`**

```
skmultilearn>=0.2.0
```

- [x] **Step 2: Add `check_label_balance()` and `stratify_aligned_cohort()` to `psi_alignment.py`**

```python
def check_label_balance(aligned_ids: pd.DataFrame,
                        site_a: pd.DataFrame,
                        site_b: pd.DataFrame,
                        site_c: pd.DataFrame,
                        pheno_cols: list[str],
                        tol: float = 0.03) -> bool:
    """
    Returns True if all splits are balanced within tol of the train distribution.
    Checks: IHM rate (binary), LOS bin distribution (10-class), per-phenotype rate (25 labels).
    """
    merged = (aligned_ids
              .merge(site_a[["subject_id", "y_ihm"]], on="subject_id", how="left")
              .merge(site_b[["subject_id", "y_los"]], on="subject_id", how="left")
              .merge(site_c[["subject_id"] + pheno_cols], on="subject_id", how="left"))

    train_ihm = merged.loc[merged["split"] == "train", "y_ihm"].mean()
    for split in ["val", "test"]:
        split_ihm = merged.loc[merged["split"] == split, "y_ihm"].mean()
        if abs(split_ihm - train_ihm) > tol:
            print(f"  IHM imbalance in {split}: train={train_ihm:.3f}, {split}={split_ihm:.3f}")
            return False

    # Check each phenotype label
    for col in pheno_cols:
        train_prev = merged.loc[merged["split"] == "train", col].mean()
        for split in ["val", "test"]:
            split_prev = merged.loc[merged["split"] == split, col].mean()
            if abs(split_prev - train_prev) > tol:
                print(f"  Phenotype '{col}' imbalance in {split}: "
                      f"train={train_prev:.3f}, {split}={split_prev:.3f}")
                return False
    return True


def stratify_aligned_cohort(aligned_ids: pd.DataFrame,
                             site_a: pd.DataFrame,
                             site_b: pd.DataFrame,
                             site_c: pd.DataFrame,
                             pheno_cols: list[str],
                             val_frac: float = 0.15,
                             test_frac: float = 0.15,
                             seed: int = 42) -> pd.DataFrame:
    """
    Re-assigns train/val/test within the aligned cohort using stratified splitting.
    - IHM: stratified on binary y_ihm label
    - LOS: stratified on 10-bin LOS class
    - Phenotyping: iterative stratification across all 25 labels (Sechidis et al. 2011)
    Combines all three stratification signals by concatenating the label matrix.
    Returns updated aligned_ids DataFrame with new 'split' column.
    """
    from sklearn.model_selection import train_test_split
    from skmultilearn.model_selection import IterativeStratification

    merged = (aligned_ids[["subject_id"]]
              .merge(site_a[["subject_id", "y_ihm"]], on="subject_id", how="left")
              .merge(site_b[["subject_id", "y_los"]], on="subject_id", how="left")
              .merge(site_c[["subject_id"] + pheno_cols], on="subject_id", how="left"))

    # Build combined label matrix: [y_ihm | y_los_onehot | y_pheno]
    # One-hot encode LOS bins so iterative stratification sees them as binary signals
    los_bins = pd.get_dummies(merged["y_los"].round().astype(int), prefix="los")
    label_matrix = pd.concat(
        [merged[["y_ihm"]], los_bins, merged[pheno_cols]], axis=1
    ).fillna(0).to_numpy()

    ids = merged["subject_id"].to_numpy()

    # Split into (train+val) vs test
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_frac, 1 - test_frac],
    )
    test_idx, trainval_idx = next(stratifier.split(ids.reshape(-1, 1), label_matrix))

    # Split (train+val) into train vs val
    val_frac_of_trainval = val_frac / (1 - test_frac)
    stratifier2 = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[val_frac_of_trainval, 1 - val_frac_of_trainval],
    )
    val_idx, train_idx = next(stratifier2.split(
        ids[trainval_idx].reshape(-1, 1), label_matrix[trainval_idx]
    ))
    val_idx_global  = trainval_idx[val_idx]
    train_idx_global = trainval_idx[train_idx]

    split_col = np.empty(len(ids), dtype=object)
    split_col[train_idx_global] = "train"
    split_col[val_idx_global]   = "val"
    split_col[test_idx]         = "test"

    result = pd.DataFrame({"subject_id": ids, "split": split_col})
    print(f"  Stratified split: train={( split_col=='train').sum()}, "
          f"val={(split_col=='val').sum()}, test={(split_col=='test').sum()}")
    return result
```

- [x] **Step 3: Wire into `psi_alignment.py` main flow**

After computing `aligned_ids` from the PSI intersection, add:

```python
# Check label balance; re-stratify if any split deviates > 3pp from train
balanced = check_label_balance(aligned_ids, site_a, site_b, site_c, pheno_cols)
if not balanced:
    print("  Label imbalance detected — re-assigning splits with iterative stratification ...")
    aligned_ids = stratify_aligned_cohort(aligned_ids, site_a, site_b, site_c, pheno_cols)
else:
    print("  Label balance OK — keeping inherited YerevaNN splits.")
```

- [x] **Step 4: Re-run `psi_alignment.py` on Snellius** (done 2026-03-19)

```bash
# on Snellius — re-run after vertical_split.py has produced updated CSVs
python data_prep/psi_alignment.py \
    --site_a /home/asoare/vfl_mlt/data/site_A_vitals.csv \
    --site_b /home/asoare/vfl_mlt/data/site_B_labs.csv \
    --site_c /home/asoare/vfl_mlt/data/site_C_composite.csv \
    --output /home/asoare/vfl_mlt/data/aligned_patient_ids.csv
```

Confirm output log shows either "Label balance OK" or reports which splits were re-stratified.

- [x] **Step 5: Commit** (already in git history)

---

## Chunk 1: Data Layer and Federated Aggregation

### ✅ Task 1: VFLDataset — `data_prep/dataset.py`

**Context:** `vertical_split.py` produces three CSVs where each row is one ICU stay, with mean-aggregated feature values and task labels. `psi_alignment.py` produces `aligned_patient_ids.csv` with `subject_id` and `split` columns. The Dataset must: filter to aligned IDs for the requested split, return per-site feature tensors with T=1 (single aggregated timestep), a ones mask, and all task labels. LOS is bucketed into 10 bins using YerevaNN-compatible thresholds.

**LOS bins (hours):** `[0, 8, 16, 24, 36, 48, 72, 120, 168, 336, ∞)` — 10 intervals, label = index 0–9.

**Files:**
- Create: `data_prep/dataset.py`
- Create: `tests/test_dataset.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_prep.dataset import VFLDataset, bin_los

def make_site_csvs(tmp_path):
    """Create minimal synthetic site CSVs for testing."""
    subject_ids = list(range(10))
    # Site A
    df_a = pd.DataFrame({
        "stay": [f"{i}_ep1.csv" for i in subject_ids],
        "subject_id": subject_ids,
        "split": ["train"] * 8 + ["val"] * 2,
        "Heart Rate": np.random.rand(10),
        "Systolic blood pressure": np.random.rand(10),
        "Diastolic blood pressure": np.random.rand(10),
        "Temperature": np.random.rand(10),
        "Oxygen saturation": np.random.rand(10),
        "Respiratory rate": np.random.rand(10),
        "Glascow coma scale total": np.random.rand(10),
        "y_ihm": np.random.randint(0, 2, 10).astype(float),
    })
    # Site B
    df_b = pd.DataFrame({
        "stay": [f"{i}_ep1.csv" for i in subject_ids],
        "subject_id": subject_ids,
        "split": ["train"] * 8 + ["val"] * 2,
        "Glucose": np.random.rand(10),
        "pH": np.random.rand(10),
        "Fraction inspired oxygen": np.random.rand(10),
        "Capillary refill rate": np.random.rand(10),
        "y_los": np.random.uniform(0, 400, 10),
    })
    # Site C (just 3 features + 2 phenotype labels for simplicity)
    pheno_cols = {f"Pheno_{k}": np.random.randint(0, 2, 10).astype(float) for k in range(25)}
    df_c = pd.DataFrame({
        "stay": [f"{i}_ep1.csv" for i in subject_ids],
        "subject_id": subject_ids,
        "split": ["train"] * 8 + ["val"] * 2,
        "Height": np.random.rand(10),
        "Weight": np.random.rand(10),
        "Mean blood pressure": np.random.rand(10),
        **pheno_cols,
    })
    # Aligned IDs
    df_aligned = pd.DataFrame({
        "subject_id": subject_ids,
        "split": ["train"] * 8 + ["val"] * 2,
    })

    df_a.to_csv(tmp_path / "site_A_vitals.csv", index=False)
    df_b.to_csv(tmp_path / "site_B_labs.csv", index=False)
    df_c.to_csv(tmp_path / "site_C_composite.csv", index=False)
    df_aligned.to_csv(tmp_path / "aligned_patient_ids.csv", index=False)
    return tmp_path


def test_bin_los():
    assert bin_los(0.0)   == 0
    assert bin_los(8.0)   == 1
    assert bin_los(7.9)   == 0
    assert bin_los(336.0) == 9
    assert bin_los(999.0) == 9


def test_dataset_length(tmp_path):
    make_site_csvs(tmp_path)
    ds = VFLDataset(splits_dir=tmp_path, split="train")
    assert len(ds) == 8


def test_dataset_item_shapes(tmp_path):
    make_site_csvs(tmp_path)
    ds = VFLDataset(splits_dir=tmp_path, split="train")
    item = ds[0]
    assert item["x_A"].shape == (1, 7)
    assert item["mask_A"].shape == (1,)
    assert item["mask_A"][0] == 1.0
    assert item["x_B"].shape == (1, 4)
    assert item["x_C"].shape == (1, 3)
    assert item["y_ihm"].shape == ()         # scalar
    assert item["y_los"].shape == ()          # scalar int64
    assert item["y_pheno"].shape == (25,)


def test_dataset_los_binned(tmp_path):
    make_site_csvs(tmp_path)
    ds = VFLDataset(splits_dir=tmp_path, split="train")
    for i in range(len(ds)):
        y_los = ds[i]["y_los"].item()
        assert 0 <= y_los <= 9, f"LOS bin out of range: {y_los}"


def test_dataset_val_split(tmp_path):
    make_site_csvs(tmp_path)
    ds = VFLDataset(splits_dir=tmp_path, split="val")
    assert len(ds) == 2
```

- [x] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ameliasoare/Documents/codes
python -m pytest tests/test_dataset.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'data_prep.dataset'`

- [x] **Step 3: Implement `data_prep/dataset.py`**

```python
"""
data_prep/dataset.py — PyTorch Dataset wrapping the three vertical split CSVs.

Each __getitem__ returns a dict with tensors for all three sites and all task labels.
LOS (continuous hours) is bucketed into 10 bins using YerevaNN-compatible thresholds.

Feature columns are T=1 sequences (mean-aggregated by vertical_split.py); mask is
all-ones with shape (1,).

Usage:
    from data_prep.dataset import VFLDataset
    from torch.utils.data import DataLoader

    ds_train = VFLDataset(splits_dir="data/vertical_splits", split="train")
    loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    for batch in loader:
        x_A, mask_A = batch["x_A"], batch["mask_A"]  # (B,1,7), (B,1)
        y_ihm = batch["y_ihm"]                         # (B,)  float32
        y_los = batch["y_los"]                         # (B,)  int64
        y_pheno = batch["y_pheno"]                     # (B,25) float32
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# LOS bin edges in hours — 10 bins (0-indexed)
_LOS_BINS = [0, 8, 16, 24, 36, 48, 72, 120, 168, 336]  # right edges exclusive


def bin_los(hours: float) -> int:
    """Map LOS in hours to a bin index 0-9."""
    for i, edge in enumerate(_LOS_BINS[1:], start=1):
        if hours < edge:
            return i - 1
    return len(_LOS_BINS) - 1  # bin 9: >= 336 h


# Column names for each site (must match vertical_split.py output)
_SITE_A_COLS = [
    "Heart Rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Temperature", "Oxygen saturation", "Respiratory rate",
    "Glascow coma scale total",
]
_SITE_B_COLS = [
    "Glucose", "pH", "Fraction inspired oxygen", "Capillary refill rate",
]
_SITE_C_COLS = [
    "Height", "Weight", "Mean blood pressure",
]

_META_COLS = {"stay", "subject_id", "split", "y_ihm", "y_los"}


class VFLDataset(Dataset):
    """
    Parameters
    ----------
    splits_dir : directory containing site_A_vitals.csv, site_B_labs.csv,
                 site_C_composite.csv, aligned_patient_ids.csv
    split      : 'train', 'val', or 'test'
    """

    def __init__(self, splits_dir: str | Path, split: str):
        self.split = split
        splits_dir = Path(splits_dir)

        # Load aligned patient IDs for this split
        aligned = pd.read_csv(splits_dir / "aligned_patient_ids.csv")
        aligned_ids = set(
            aligned.loc[aligned["split"] == split, "subject_id"].tolist()
        )

        def load_and_filter(fname: str) -> pd.DataFrame:
            df = pd.read_csv(splits_dir / fname)
            df = df[(df["split"] == split) & (df["subject_id"].isin(aligned_ids))]
            return df.set_index("subject_id")

        df_a = load_and_filter("site_A_vitals.csv")
        df_b = load_and_filter("site_B_labs.csv")
        df_c = load_and_filter("site_C_composite.csv")

        # Phenotype label columns: everything not in meta/feature columns
        pheno_label_cols = sorted(
            c for c in df_c.columns if c not in _META_COLS and c not in _SITE_C_COLS
        )

        # Common index (PSI-intersected subject IDs present in all three CSVs)
        common = sorted(
            set(df_a.index) & set(df_b.index) & set(df_c.index)
        )

        df_a = df_a.loc[common]
        df_b = df_b.loc[common]
        df_c = df_c.loc[common]

        # Build numpy arrays
        self.x_A = df_a[_SITE_A_COLS].to_numpy(dtype=np.float32)   # (N, 7)
        self.x_B = df_b[_SITE_B_COLS].to_numpy(dtype=np.float32)   # (N, 4)
        self.x_C = df_c[_SITE_C_COLS].to_numpy(dtype=np.float32)   # (N, 3)

        self.y_ihm   = df_a["y_ihm"].to_numpy(dtype=np.float32)     # (N,)
        self.y_los   = np.array(
            [bin_los(h) for h in df_b["y_los"].to_numpy()],
            dtype=np.int64,
        )                                                             # (N,)
        self.y_pheno = df_c[pheno_label_cols].to_numpy(
            dtype=np.float32
        )                                                             # (N, 25)

    def __len__(self) -> int:
        return len(self.y_ihm)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # Add T=1 dimension so shapes are (1, n_features) — LSTM-compatible
        x_A = torch.from_numpy(self.x_A[idx]).unsqueeze(0)   # (1, 7)
        x_B = torch.from_numpy(self.x_B[idx]).unsqueeze(0)   # (1, 4)
        x_C = torch.from_numpy(self.x_C[idx]).unsqueeze(0)   # (1, 3)

        mask_A = torch.ones(1, dtype=torch.float32)
        mask_B = torch.ones(1, dtype=torch.float32)
        mask_C = torch.ones(1, dtype=torch.float32)

        return {
            "x_A": x_A, "mask_A": mask_A,
            "x_B": x_B, "mask_B": mask_B,
            "x_C": x_C, "mask_C": mask_C,
            "y_ihm":   torch.tensor(self.y_ihm[idx],   dtype=torch.float32),
            "y_los":   torch.tensor(self.y_los[idx],   dtype=torch.long),
            "y_pheno": torch.from_numpy(self.y_pheno[idx]),
        }
```

- [x] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dataset.py -v
```
Expected: 6 tests PASSED.

- [x] **Step 5: Commit**

```bash
git add data_prep/dataset.py tests/test_dataset.py
git commit -m "feat: add VFLDataset with LOS binning and PSI-aligned filtering"
```

---

### ✅ Task 2: FedAvg + FedProx — `fl/fedavg.py`, `fl/fedprox.py`

**Context:** `VFLClient.get_encoder_params()` and `set_encoder_params()` already exist. FedAvg averages the state dicts weighted by local dataset size. FedProx adds a proximal term `(mu/2) * ||w - w_global||^2` to the client's objective; it's applied during `receive_gradient()` in a subclass or by calling `fedprox_loss()` before backward.

**Files:**
- Create: `fl/fedavg.py`
- Create: `fl/fedprox.py`
- Create: `tests/test_fedavg.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/test_fedavg.py
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fl.client import VFLClient
from fl.fedavg import fedavg_aggregate
from fl.fedprox import fedprox_penalty


def make_clients(n: int = 3, input_dim: int = 7):
    return [VFLClient(input_dim=input_dim) for _ in range(n)]


def test_fedavg_output_is_weighted_average():
    """fedavg_aggregate must return a state dict that is the weighted average."""
    torch.manual_seed(0)
    clients = make_clients()

    # Artificially set each client's first param to a known value
    key = list(clients[0].encoder.state_dict().keys())[0]
    for i, c in enumerate(clients):
        state = c.encoder.state_dict()
        state[key] = torch.zeros_like(state[key]) + float(i)
        c.set_encoder_params(state)

    weights = [1, 2, 3]  # client 0 has 1 sample, client 1 has 2, client 2 has 3
    params_list = [c.get_encoder_params() for c in clients]
    avg = fedavg_aggregate(params_list, weights)

    expected_val = (0*1 + 1*2 + 2*3) / (1+2+3)  # = 8/6 ≈ 1.333
    actual_val = avg[key].mean().item()
    assert abs(actual_val - expected_val) < 1e-5, f"Expected {expected_val}, got {actual_val}"


def test_fedavg_sets_params_on_clients():
    """After aggregation, setting params on clients should update their weights."""
    torch.manual_seed(1)
    clients = make_clients()
    weights = [10, 10, 10]
    params_list = [c.get_encoder_params() for c in clients]
    avg = fedavg_aggregate(params_list, weights)

    # Should not raise
    for c in clients:
        c.set_encoder_params(avg)


def test_fedprox_penalty_zero_when_same():
    """Penalty must be 0 when local params equal global params."""
    torch.manual_seed(2)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    penalty = fedprox_penalty(client.encoder, global_params, mu=1.0)
    assert penalty.item() < 1e-8, f"Expected ~0, got {penalty.item()}"


def test_fedprox_penalty_positive_when_different():
    """Penalty must be positive when local params differ from global."""
    torch.manual_seed(3)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    # Perturb local params
    for param in client.encoder.parameters():
        param.data += 1.0
    penalty = fedprox_penalty(client.encoder, global_params, mu=1.0)
    assert penalty.item() > 0.0


def test_fedprox_penalty_scales_with_mu():
    """Doubling mu must double the penalty."""
    torch.manual_seed(4)
    client = VFLClient(input_dim=7)
    global_params = client.get_encoder_params()
    for param in client.encoder.parameters():
        param.data += 0.5

    p1 = fedprox_penalty(client.encoder, global_params, mu=1.0).item()
    p2 = fedprox_penalty(client.encoder, global_params, mu=2.0).item()
    assert abs(p2 - 2 * p1) < 1e-5, f"Expected p2=2*p1, got p1={p1}, p2={p2}"
```

- [x] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_fedavg.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError`

- [x] **Step 3: Implement `fl/fedavg.py`**

```python
"""
fl/fedavg.py — FedAvg aggregation for VFL-MTL encoder parameters.

fedavg_aggregate() computes a weighted average of encoder state dicts
(weighted by local dataset size) and returns a new global state dict.

Usage in train.py:
    from fl.fedavg import fedavg_aggregate

    global_params = fedavg_aggregate(
        [c.get_encoder_params() for c in clients],
        weights=[len(train_split_A), len(train_split_B), len(train_split_C)],
    )
    for client in clients:
        client.set_encoder_params(global_params)
"""

from __future__ import annotations
import torch


def fedavg_aggregate(
    params_list: list[dict],
    weights: list[int],
) -> dict:
    """
    Compute weighted average of encoder state dicts.

    Parameters
    ----------
    params_list : list of state dicts from VFLClient.get_encoder_params()
    weights     : non-negative integer weights (e.g. local dataset sizes)

    Returns
    -------
    Averaged state dict (same keys and dtypes as inputs)
    """
    assert len(params_list) == len(weights), "params_list and weights must have same length"
    total = sum(weights)
    assert total > 0, "Total weight must be positive"

    avg = {}
    for key in params_list[0]:
        weighted_sum = sum(
            params[key].float() * w
            for params, w in zip(params_list, weights)
        )
        avg[key] = (weighted_sum / total).to(params_list[0][key].dtype)

    return avg
```

- [x] **Step 4: Implement `fl/fedprox.py`**

```python
"""
fl/fedprox.py — FedProx proximal penalty for VFL-MTL client training.

The proximal term (mu/2) * ||w_local - w_global||^2 is added to the
client's loss before backpropagation, penalising deviation from the
global model. This improves convergence in heterogeneous federated settings.

Reference: Li et al. (2020), "Federated Optimization in Heterogeneous Networks"

Usage in train.py:
    from fl.fedprox import fedprox_penalty

    # After server returns gradient but before client backward:
    # (Alternative: add penalty to total_loss before backward_and_step)
    penalty = fedprox_penalty(client.encoder, global_params, mu=0.01)
    # Include in loss passed to backward
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


def fedprox_penalty(
    local_model: nn.Module,
    global_params: dict,
    mu: float = 0.01,
) -> Tensor:
    """
    Compute the FedProx proximal penalty.

    penalty = (mu / 2) * sum_over_layers( ||w_local - w_global||^2 )

    Parameters
    ----------
    local_model   : the client's local nn.Module (SiteEncoder)
    global_params : global state dict from fedavg_aggregate()
    mu            : proximal penalty coefficient (default 0.01)

    Returns
    -------
    Scalar tensor (differentiable w.r.t. local_model.parameters())
    """
    penalty = torch.tensor(0.0)
    for name, param in local_model.named_parameters():
        if name in global_params:
            global_val = global_params[name].to(param.device)
            penalty = penalty + ((param - global_val) ** 2).sum()
    return (mu / 2) * penalty
```

- [x] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_fedavg.py -v
```
Expected: 5 tests PASSED.

- [x] **Step 6: Update `fl/__init__.py`**

Add exports:
```python
from .client import VFLClient
from .server import VFLServer
from .fedavg import fedavg_aggregate
from .fedprox import fedprox_penalty
```

- [x] **Step 7: Commit**

```bash
git add fl/fedavg.py fl/fedprox.py fl/__init__.py tests/test_fedavg.py
git commit -m "feat: add FedAvg aggregation and FedProx proximal penalty"
```

---

## Chunk 2: Training Loop

### ✅ Task 3: Training orchestration — `train.py`

**Context:** `train.py` is the central round-based training loop used by all experiment scripts. It accepts configuration as a dataclass/dict and returns per-round metrics. Each round: (1) all clients forward-pass their batches, (2) server aggregates and computes loss, (3) server backprops and returns gradients, (4) clients update. Optional FedAvg runs every `fedavg_every` rounds. Optional FedProx adds the proximal penalty to each client before backward.

**Data interface:** `dataset.py` uses `build_site_loaders(root, split)` returning `{'A': DataLoader, 'B': DataLoader, 'C': DataLoader}`. Each loader yields `(x, mask, y)` tuples — NOT dicts. Batches across sites must be iterated in lockstep using `zip(loader_A, loader_B, loader_C)`. The `mimic3-benchmarks/` directory must be on the path (already handled inside `dataset.py` itself).

**Evaluation:** after each round, run `eval_round()` that calls `client.eval_forward()` on the val DataLoader and `server.predict()`, then computes all task metrics.

**Files:**
- Create: `experiments/metrics.py`
- Create: `train.py`
- Create: `tests/test_train.py`

- [x] **Step 1: Write `experiments/metrics.py`** (no test needed — thin wrapper around sklearn)

```python
"""
experiments/metrics.py — Per-task metric computation for VFL-MTL.

All functions take numpy arrays. Call after collecting predictions over
the full val/test set.

Metrics per task:
  IHM (binary)        : AUC-ROC, AUC-PR
  LOS (10-class)      : Cohen's kappa (quadratic), mean absolute deviation
  Phenotyping (multi) : Macro-AUC, Micro-AUC
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)


def ihm_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N,) int or float binary labels
    y_prob : (N,) predicted probabilities
    """
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr":  average_precision_score(y_true, y_prob),
    }


def los_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true     : (N,) int64 bin indices 0-9
    y_pred_bin : (N,) int64 predicted bin indices (argmax of logits)
    """
    kappa = cohen_kappa_score(y_true, y_pred_bin, weights="quadratic")
    mad   = float(np.mean(np.abs(y_true.astype(float) - y_pred_bin.astype(float))))
    return {"kappa": kappa, "mad": mad}


def pheno_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    y_true : (N, 25) binary multi-label targets
    y_prob : (N, 25) predicted probabilities
    """
    macro_auc = roc_auc_score(y_true, y_prob, average="macro")
    micro_auc = roc_auc_score(y_true, y_prob, average="micro")
    return {"macro_auc": macro_auc, "micro_auc": micro_auc}


def compute_all_metrics(
    ihm_true, ihm_prob,
    los_true, los_pred_bin,
    pheno_true, pheno_prob,
) -> dict[str, float]:
    """Convenience: compute all task metrics and return flat dict."""
    out = {}
    out.update({f"ihm_{k}": v  for k, v in ihm_metrics(ihm_true, ihm_prob).items()})
    out.update({f"los_{k}": v  for k, v in los_metrics(los_true, los_pred_bin).items()})
    out.update({f"pheno_{k}": v for k, v in pheno_metrics(pheno_true, pheno_prob).items()})
    return out
```

Run sanity check:
```bash
python -c "
import numpy as np; import sys; sys.path.insert(0,'.')
from experiments.metrics import compute_all_metrics
N=100
print(compute_all_metrics(
    np.random.randint(0,2,N), np.random.rand(N),
    np.random.randint(0,10,N), np.random.randint(0,10,N),
    np.random.randint(0,2,(N,25)), np.random.rand(N,25)
))
"
```
Expected: dict with 6 float values, no errors.

- [x] **Step 2: Write the failing train test**

```python
# tests/test_train.py
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig


def test_run_training_loss_decreases_synthetic():
    """run_training on synthetic data must return decreasing loss over 5 rounds."""
    torch.manual_seed(42)
    cfg = TrainConfig(
        n_rounds=5,
        batch_size=16,
        use_fedavg=False,
        use_fedprox=False,
        device="cpu",
        seed=42,
        use_synthetic=True,  # skip real data; generate tensors internally
        n_synthetic=64,
    )
    results = run_training(cfg)
    losses = [r["train_loss"] for r in results]
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_run_training_returns_metrics():
    """run_training must return a list of dicts with expected keys."""
    torch.manual_seed(0)
    cfg = TrainConfig(
        n_rounds=2, batch_size=16, use_synthetic=True, n_synthetic=64,
        use_fedavg=False, use_fedprox=False, device="cpu", seed=0,
    )
    results = run_training(cfg)
    assert len(results) == 2
    for r in results:
        assert "round" in r
        assert "train_loss" in r


def test_run_training_with_fedavg():
    """FedAvg run must complete without errors."""
    torch.manual_seed(1)
    cfg = TrainConfig(
        n_rounds=3, batch_size=16, use_synthetic=True, n_synthetic=64,
        use_fedavg=True, fedavg_every=1, use_fedprox=False, device="cpu", seed=1,
    )
    results = run_training(cfg)
    assert len(results) == 3


def test_run_training_with_fedprox():
    """FedProx run must complete without errors."""
    torch.manual_seed(2)
    cfg = TrainConfig(
        n_rounds=3, batch_size=16, use_synthetic=True, n_synthetic=64,
        use_fedavg=False, use_fedprox=True, mu=0.01, device="cpu", seed=2,
    )
    results = run_training(cfg)
    assert len(results) == 3
```

- [x] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_train.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'train'`

- [x] **Step 4: Implement `train.py`**

```python
"""
train.py — Round-based VFL-MTL training loop.

Orchestrates three VFLClients and one VFLServer across communication rounds.
Supports optional FedAvg encoder aggregation and FedProx proximal penalty.

Usage (from experiment scripts):
    from train import run_training, TrainConfig

    cfg = TrainConfig(
        splits_dir="data/vertical_splits",
        n_rounds=50,
        batch_size=64,
        use_fedavg=True,
        fedavg_every=5,
        use_fedprox=False,
        device="cpu",
        seed=42,
    )
    results = run_training(cfg)  # list of per-round dicts

Usage (CLI):
    python train.py --splits_dir data/vertical_splits --n_rounds 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl.client import VFLClient
from fl.fedavg import fedavg_aggregate
from fl.fedprox import fedprox_penalty
from fl.server import VFLServer


@dataclass
class TrainConfig:
    # Data — project root (contains data/vertical_splits/ and data/mimic3-benchmarks/)
    project_root: str = "."
    num_workers: int = 0
    # Training
    n_rounds: int = 50
    batch_size: int = 64
    lr_client: float = 1e-3
    lr_server: float = 1e-3
    # Federated settings
    use_fedavg: bool = True
    fedavg_every: int = 5          # aggregate every N rounds
    use_fedprox: bool = False
    mu: float = 0.01               # FedProx coefficient
    # Model
    hidden_dim: int = 128
    num_layers: int = 2
    embed_dim: int = 64
    num_experts: int = 4
    expert_hidden: int = 128
    task_weights: dict = field(default_factory=lambda: {"ihm": 1.0, "los": 1.0, "pheno": 1.0})
    # Reproducibility
    seed: int = 42
    device: str = "cpu"
    # Synthetic data mode (for unit tests — skips file I/O)
    use_synthetic: bool = False
    n_synthetic: int = 256
    # Output
    output_csv: Optional[str] = None


# ---------------------------------------------------------------------------
# Synthetic data factory (for tests and smoke runs)
# ---------------------------------------------------------------------------

def _synthetic_loader(n: int, batch_size: int):
    """Return (train_loader, val_loader) over synthetic tensors."""
    x_A    = torch.randn(n, 1, 7)
    x_B    = torch.randn(n, 1, 4)
    x_C    = torch.randn(n, 1, 3)
    mask_A = torch.ones(n, 1)
    mask_B = torch.ones(n, 1)
    mask_C = torch.ones(n, 1)
    y_ihm  = torch.randint(0, 2, (n,)).float()
    y_los  = torch.randint(0, 10, (n,)).long()
    y_pheno= torch.randint(0, 2, (n, 25)).float()

    ds = TensorDataset(x_A, mask_A, x_B, mask_B, x_C, mask_C, y_ihm, y_los, y_pheno)
    n_val = max(1, n // 8)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n - n_val, n_val])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size),
    )


def _batch_to_dicts(batch):
    """Convert TensorDataset batch tuple to named dicts (matches VFLDataset format)."""
    x_A, mask_A, x_B, mask_B, x_C, mask_C, y_ihm, y_los, y_pheno = batch
    inputs = {"A": (x_A, mask_A), "B": (x_B, mask_B), "C": (x_C, mask_C)}
    labels = {"ihm": y_ihm, "los": y_los, "pheno": y_pheno}
    return inputs, labels


def _real_loaders(cfg: TrainConfig):
    """
    Return per-site loaders using build_site_loaders() from dataset.py.
    Each loader yields (x, mask, y) tuples.
    Returns (train_loaders, val_loaders) where each is dict {'A':..,'B':..,'C':..}
    """
    from data_prep.dataset import build_site_loaders
    train_loaders = build_site_loaders(cfg.project_root, "train",
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_workers)
    val_loaders   = build_site_loaders(cfg.project_root, "val",
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_workers)
    return train_loaders, val_loaders


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> list[dict]:
    """
    Execute the round-based VFL-MTL training protocol.

    Returns
    -------
    List of per-round result dicts with keys:
      round, train_loss, ihm_loss, los_loss, pheno_loss,
      wall_time_s, (val metrics if real data)
    """
    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)

    # Build clients — one per site
    site_dims = {"A": 7, "B": 4, "C": 3}
    clients = {
        site: VFLClient(
            input_dim=dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            embed_dim=cfg.embed_dim,
            lr=cfg.lr_client,
            device=device,
        )
        for site, dim in site_dims.items()
    }

    server = VFLServer(
        embed_dim=cfg.embed_dim,
        num_experts=cfg.num_experts,
        expert_hidden=cfg.expert_hidden,
        lr=cfg.lr_server,
        device=device,
        task_weights=cfg.task_weights,
    )

    # Data
    if cfg.use_synthetic:
        train_loader, val_loader = _synthetic_loader(cfg.n_synthetic, cfg.batch_size)
        get_inputs_labels = _batch_to_dicts
    else:
        train_loader, val_loader = _real_loaders(cfg)
        get_inputs_labels = _dict_batch_to_inputs_labels

    # Global params for FedProx
    global_params = {
        site: client.get_encoder_params()
        for site, client in clients.items()
    } if cfg.use_fedprox else None

    results = []

    for round_idx in range(1, cfg.n_rounds + 1):
        t0 = time.time()
        round_losses = []

        # Lockstep iteration across per-site loaders
        if cfg.use_synthetic:
            batch_iter = train_loader      # single synthetic loader
        else:
            batch_iter = zip(train_loader["A"], train_loader["B"], train_loader["C"])

        for raw_batch in batch_iter:
            if cfg.use_synthetic:
                inputs, labels = _batch_to_dicts(raw_batch)
            else:
                # raw_batch is (batch_A, batch_B, batch_C), each (x, mask, y)
                (x_A, mask_A, y_A), (x_B, mask_B, y_B), (x_C, mask_C, y_C) = raw_batch
                inputs = {"A": (x_A, mask_A), "B": (x_B, mask_B), "C": (x_C, mask_C)}
                labels = {"ihm": y_A, "los": y_B, "pheno": y_C}

            # --- Forward pass: each client encodes ---
            cut_embeddings = {}
            for site, client in clients.items():
                x, mask = inputs[site]
                dim = cfg.site_input_dims[site]
                x = x[..., :dim]
                cut_embeddings[site] = client.forward(x, mask)

            # --- Server: aggregate, loss, backward ---
            server.aggregate_embeddings(cut_embeddings)
            total_loss, task_losses = server.forward_and_loss(labels)

            # FedProx: add proximal penalty from each client
            if cfg.use_fedprox and global_params is not None:
                for site, client in clients.items():
                    total_loss = total_loss + fedprox_penalty(
                        client.encoder, global_params[site], cfg.mu
                    )

            server.backward_and_step(total_loss)
            grads = server.get_embedding_gradients()

            # --- Clients: apply gradients ---
            for site, client in clients.items():
                client.receive_gradient(grads[site])

            round_losses.append({
                "total": total_loss.item(),
                "ihm":   task_losses["ihm"].item(),
                "los":   task_losses["los"].item(),
                "pheno": task_losses["pheno"].item(),
            })

        # FedAvg: aggregate encoder parameters
        # NOTE: Each site has a different LSTM input_dim (7/4/3), so full state-dict
        # averaging across sites is architecturally invalid (weight_ih shapes differ).
        # FedAvg here only averages compatible parameter shapes — layers with the same
        # dimensions across all active clients (hh weights, projection, norm).
        # When sites have different architectures, use_fedavg=False is recommended.
        if cfg.use_fedavg and round_idx % cfg.fedavg_every == 0 and len(active_sites) > 1:
            params_list = [clients[s].get_encoder_params() for s in active_sites]
            # Filter to only keys whose tensor shapes match across all clients
            ref_shapes = {k: v.shape for k, v in params_list[0].items()}
            compatible_keys = [
                k for k in ref_shapes
                if all(p[k].shape == ref_shapes[k] for p in params_list[1:])
            ]
            compatible_params = [{k: p[k] for k in compatible_keys} for p in params_list]
            avg_compatible = fedavg_aggregate(compatible_params, [1] * len(active_sites))
            # Apply only compatible averaged params; site-specific input layers unchanged
            for client in clients.values():
                local_state = client.get_encoder_params()
                local_state.update(avg_compatible)
                client.set_encoder_params(local_state)
            # Update global params for FedProx
            if cfg.use_fedprox:
                global_params = {s: clients[s].get_encoder_params() for s in clients}

        avg_loss = np.mean([r["total"] for r in round_losses])
        result = {
            "round":       round_idx,
            "train_loss":  avg_loss,
            "ihm_loss":    np.mean([r["ihm"]   for r in round_losses]),
            "los_loss":    np.mean([r["los"]   for r in round_losses]),
            "pheno_loss":  np.mean([r["pheno"] for r in round_losses]),
            "wall_time_s": time.time() - t0,
        }
        results.append(result)
        print(
            f"Round {round_idx:3d}/{cfg.n_rounds} | "
            f"loss={avg_loss:.4f} | "
            f"ihm={result['ihm_loss']:.4f} | "
            f"los={result['los_loss']:.4f} | "
            f"pheno={result['pheno_loss']:.4f} | "
            f"t={result['wall_time_s']:.1f}s"
        )

    if cfg.output_csv:
        Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {cfg.output_csv}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run VFL-MTL training.")
    parser.add_argument("--splits_dir",   default="data/vertical_splits")
    parser.add_argument("--n_rounds",     type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--use_fedavg",   action="store_true")
    parser.add_argument("--fedavg_every", type=int,   default=5)
    parser.add_argument("--use_fedprox",    action="store_true")
    parser.add_argument("--mu",             type=float, default=0.01)
    parser.add_argument("--output_csv",     default=None)
    parser.add_argument("--use_synthetic",  action="store_true",
                        help="Skip file I/O; use synthetic tensors (for smoke tests)")
    parser.add_argument("--n_synthetic",    type=int, default=256,
                        help="Number of synthetic samples when --use_synthetic is set")
    args = parser.parse_args()

    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if v is not None})
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_train.py -v
```
Expected: 4 tests PASSED.

- [x] **Step 6: Commit**

```bash
git add experiments/metrics.py train.py tests/test_train.py
git commit -m "feat: add VFL-MTL training loop with FedAvg/FedProx support"
```

---

## Chunk 3: Experiment Scripts

### ✅ Task 4: Experiment 1 — Task Heterogeneity vs. Homogeneity (`experiments/run_exp1.py`)

**Context:** Compare VFL-MTL (3 sites, 3 heterogeneous tasks) against VFL-SingleTask (3 sites, all sites optimise only IHM). Report per-task AUC-ROC with seeds [42, 123, 7]. Write results to `results/exp1.csv`.

**VFL-SingleTask:** Use `VFLServer` with `task_weights = {"ihm": 1.0, "los": 0.0, "pheno": 0.0}` (zeroed-out tasks). This reuses the same architecture with non-participating task heads.

**Files:**
- Create: `experiments/run_exp1.py`

- [x] **Step 1: Implement `experiments/run_exp1.py`**

```python
"""
experiments/run_exp1.py — Exp 1: Task heterogeneity vs. homogeneity.

Compares:
  - VFL-MTL     : 3 sites, 3 tasks (ihm + los + pheno)
  - VFL-SingleTask: 3 sites, IHM only (los and pheno weights set to 0)

Seeds: [42, 123, 7]
Output: results/exp1.csv
  columns: model, seed, round, train_loss, ihm_loss, los_loss, pheno_loss

Usage:
    python experiments/run_exp1.py --splits_dir data/vertical_splits \
        --n_rounds 50 --device cpu
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
    "VFL-SingleTask": {
        "task_weights": {"ihm": 1.0, "los": 0.0, "pheno": 0.0},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="data/vertical_splits")
    parser.add_argument("--n_rounds",   type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--output",     default="results/exp1.csv")
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
                use_fedprox=False,
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
```

- [x] **Step 2: Add `--use_synthetic` to all experiment argparsers**

Every experiment script (`run_exp1.py` through `run_exp4.py`) must expose this flag so the script can be smoke-tested without real data. Add to each script's `argparse` section:

```python
parser.add_argument("--use_synthetic", action="store_true",
                    help="Use synthetic data (for smoke tests; no real MIMIC data needed)")
parser.add_argument("--n_synthetic",   type=int, default=256)
```

And forward both flags to `TrainConfig`:

```python
cfg = TrainConfig(
    ...
    use_synthetic=args.use_synthetic,
    n_synthetic=args.n_synthetic,
)
```

- [x] **Step 3: Smoke-test on synthetic data**

```bash
python experiments/run_exp1.py \
    --n_rounds 3 --device cpu --use_synthetic \
    --output /tmp/exp1_smoke.csv 2>&1 | tail -5
```

Expected output: loss values printed for 3 rounds × 2 models × 3 seeds; CSV written to `/tmp/exp1_smoke.csv`.

- [x] **Step 4: Commit**

```bash
git add experiments/run_exp1.py
git commit -m "feat: add Exp1 task heterogeneity vs. homogeneity script"
```

---

### ✅ Task 5: Experiment 2 — Feature Asymmetry (`experiments/run_exp2.py`)

**Context:** Test sensitivity of MTL gains to how unevenly features are distributed across sites. Three configurations are tested by overriding `site_dims` in TrainConfig. Since VFLDataset yields 7/4/3 features, asymmetry configs instead train sub-encoders that use only a subset of columns — achieved by passing `input_dim` overrides.

**Split configurations:**
- `balanced`: 5/5/4 (select first 5, 5, 4 from each site's columns)
- `skewed`:   3/7/4 (site B gets features from A and B merged — requires data prep change; **simplify**: test with input_dim only, truncate/pad feature tensor)
- `default`:  7/4/3 (standard split from CLAUDE.md)

For the experiment, "feature asymmetry" is approximated by varying `input_dim` and truncating/zero-padding the input tensor to the requested dimension. This does NOT require re-running data prep.

**Files:**
- Create: `experiments/run_exp2.py`

- [x] **Step 1: Implement `experiments/run_exp2.py`**

```python
"""
experiments/run_exp2.py — Exp 2: Feature asymmetry sensitivity.

Tests three feature-split configurations by varying the number of features
per site (input_dim). The feature vectors are truncated or zero-padded to the
requested dimension — this approximates asymmetry without re-running data prep.

Configurations:
  default  : 7/4/3  (CLAUDE.md standard)
  balanced : 5/4/3  (reduce Site A by 2 features)
  skewed   : 3/4/7  (Site C gets more, Site A fewer)

Seeds: [42, 123, 7]
Output: results/exp2.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig

SEEDS = [42, 123, 7]

SPLIT_CONFIGS = {
    "default":  {"A": 7, "B": 4, "C": 3},
    "balanced": {"A": 5, "B": 4, "C": 3},
    "skewed":   {"A": 3, "B": 4, "C": 7},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--output",        default="results/exp2.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    args = parser.parse_args()

    all_rows = []

    for split_name, dims in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            print(f"\n=== split={split_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                use_fedavg=True,
                fedavg_every=5,
                task_weights={"ihm": 1.0, "los": 1.0, "pheno": 1.0},
                # Note: input_dims overriding requires passing to TrainConfig.
                # VFLDataset always produces 7/4/3 features; client truncates to input_dim.
            )
            results = run_training(cfg)
            for r in results:
                all_rows.append({
                    "split_config": split_name,
                    "dim_A": dims["A"], "dim_B": dims["B"], "dim_C": dims["C"],
                    "seed": seed,
                    **r,
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 2 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
```

> **Implementation note for subagent:** `TrainConfig` needs a `site_input_dims` field to vary `input_dim` per client. Add `site_input_dims: dict = field(default_factory=lambda: {"A": 7, "B": 4, "C": 3})` to `TrainConfig` and update `run_training()` to read `site_dims` from `cfg.site_input_dims`. VFLDataset always returns 7/4/3; the client LSTM's `input_dim` must match — truncate in the training loop: `x = x[..., :dim]`.

- [x] **Step 2: Update `train.py` to support `site_input_dims`**

In `TrainConfig`, add:
```python
site_input_dims: dict = field(default_factory=lambda: {"A": 7, "B": 4, "C": 3})
```

In `run_training()`, change `site_dims = {"A": 7, "B": 4, "C": 3}` to `site_dims = cfg.site_input_dims`.

In the training loop batch processing, truncate:
```python
x, mask = inputs[site]
dim = cfg.site_input_dims[site]
x = x[..., :dim]   # truncate to requested feature count
```

- [x] **Step 3: Fix `run_exp2.py` to pass `site_input_dims` to `TrainConfig`**

In `run_exp2.py`, update the `TrainConfig(...)` constructor to include `site_input_dims=dims`:

```python
cfg = TrainConfig(
    splits_dir=args.splits_dir,
    n_rounds=args.n_rounds,
    batch_size=args.batch_size,
    device=args.device,
    seed=seed,
    use_fedavg=True,
    fedavg_every=5,
    task_weights={"ihm": 1.0, "los": 1.0, "pheno": 1.0},
    site_input_dims=dims,   # ← REQUIRED: wire the split config to the client dims
)
```

Without this line, all three configurations silently produce identical results.

- [x] **Step 4: Run tests still pass after train.py change**

```bash
python -m pytest tests/test_train.py -v
```
Expected: 4 tests PASSED.

- [x] **Step 5: Commit**

```bash
git add experiments/run_exp2.py train.py
git commit -m "feat: add Exp2 feature asymmetry; add site_input_dims to TrainConfig"
```

---

### ✅ Task 6: Experiment 3 — Task Relatedness / Negative Transfer (`experiments/run_exp3.py`)

**Context:** Test whether task relatedness affects MTL gains. Compare two task pairings:
- Pair A: IHM + Decompensation (related — both are mortality-type signals)
- Pair B: IHM + Phenotyping (less related — acute outcome vs. chronic condition labels)

Since decompensation task was not included in the vertical split, Pair A is approximated as IHM-only + IHM replicated at site C (same binary label). For Pair B use IHM + pheno as-is. Negative transfer = AUC on IHM task drops below local-only baseline when paired with an unrelated task.

**Files:**
- Create: `experiments/run_exp3.py`

- [x] **Step 1: Implement `experiments/run_exp3.py`**

```python
"""
experiments/run_exp3.py — Exp 3: Task relatedness and negative transfer.

Compares:
  VFL-MTL (IHM + LOS + Pheno)          — standard
  VFL-MTL-Related (IHM + IHM + IHM)    — all sites predict IHM (highly related)
  VFL-MTL-Unrelated (IHM + LOS + Pheno) — standard (same as VFL-MTL for reference)

Negative transfer rate = fraction of seeds where IHM AUC under MTL < local-only AUC.
**Important:** `run_training()` returns training loss, not val AUC. To compute the actual
negative transfer rate, `run_exp3.py` must call a separate `eval_on_val()` function after
training that loads the val split, runs client `eval_forward()` + `server.predict()` over
the val DataLoader, and computes `ihm_metrics()` from `experiments/metrics.py`.
In synthetic mode (`--use_synthetic`), skip val evaluation and write loss only.

Seeds: [42, 123, 7]
Output: results/exp3.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig, _real_loaders, _dict_batch_to_inputs_labels
from fl.client import VFLClient
from fl.server import VFLServer
from experiments.metrics import ihm_metrics
import numpy as np
import torch

SEEDS = [42, 123, 7]

TASK_CONFIGS = {
    "all_tasks":  {"ihm": 1.0, "los": 1.0, "pheno": 1.0},
    "ihm_only":   {"ihm": 1.0, "los": 0.0, "pheno": 0.0},
    "ihm_pheno":  {"ihm": 1.0, "los": 0.0, "pheno": 1.0},
    "ihm_los":    {"ihm": 1.0, "los": 1.0, "pheno": 0.0},
}


def eval_on_val(cfg: TrainConfig) -> dict:
    """
    Run one pass over the val DataLoader and compute IHM AUC-ROC.
    Returns dict with keys 'val_ihm_auc_roc', 'val_ihm_auc_pr'.
    """
    from data_prep.dataset import VFLDataset
    from torch.utils.data import DataLoader

    def collate(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    val_ds = VFLDataset(cfg.splits_dir, "val")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate)

    # Rebuild model objects (weights not restored — this eval is for final-round metrics)
    site_dims = {"A": 7, "B": 4, "C": 3}
    clients = {s: VFLClient(input_dim=d, device=cfg.device) for s, d in site_dims.items()}
    server  = VFLServer(device=cfg.device)

    all_y_true, all_y_prob = [], []
    for batch in val_loader:
        inputs, labels = _dict_batch_to_inputs_labels(batch)
        embeddings = {s: clients[s].eval_forward(*inputs[s]) for s in site_dims}
        preds = server.predict(embeddings)
        all_y_true.append(labels["ihm"].numpy())
        all_y_prob.append(preds["ihm"].squeeze(-1).detach().numpy())

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)
    metrics = ihm_metrics(y_true, y_prob)
    return {f"val_{k}": v for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--output",        default="results/exp3.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--n_synthetic",   type=int, default=256)
    args = parser.parse_args()

    all_rows = []

    for config_name, task_weights in TASK_CONFIGS.items():
        for seed in SEEDS:
            print(f"\n=== config={config_name} | seed={seed} ===")
            cfg = TrainConfig(
                splits_dir=args.splits_dir,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed,
                use_fedavg=True,
                fedavg_every=5,
                task_weights=task_weights,
            )
            results = run_training(cfg)

            # Val AUC evaluation (real data only; skipped in synthetic mode)
            if not args.use_synthetic:
                val_metrics = eval_on_val(cfg)
                for r in results:
                    all_rows.append({"task_config": config_name, "seed": seed,
                                     **r, **val_metrics})
            else:
                for r in results:
                    all_rows.append({"task_config": config_name, "seed": seed, **r})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nExp 3 complete. Results → {args.output}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Commit**

```bash
git add experiments/run_exp3.py
git commit -m "feat: add Exp3 task relatedness and negative transfer script"
```

---

### ✅ Task 7: Experiment 4 — Scalability (`experiments/run_exp4.py`)

**Context:** Vary the number of participating institutions (2 or 3) and measure convergence rounds and wall-clock time. With 2 sites, use only A+B (IHM + LOS). With 3 sites, use the full setup.

**Files:**
- Create: `experiments/run_exp4.py`

- [x] **Step 1: Add `n_sites` support to `VFLServer` and `train.py`**

**Modify `fl/server.py`** — add `n_sites` parameter so `VFLServer` only uses the first `n_sites` sites:

```python
# In VFLServer.__init__, change signature to:
def __init__(
    self,
    embed_dim: int = 64,
    num_experts: int = 4,
    expert_hidden: int = 128,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    task_weights: dict[str, float] | None = None,
    n_sites: int = 3,          # ← ADD THIS
):
    self.device    = torch.device(device)
    self.embed_dim = embed_dim
    self.SITES     = ("A", "B", "C")[:n_sites]   # ← ADD THIS (override class attr)

    self.model = MMoEServer(
        input_dim=n_sites * embed_dim,            # ← CHANGE: was len(self.SITES) * embed_dim
        ...
    )
```

The class-level `SITES = ("A", "B", "C")` stays unchanged; the instance attribute `self.SITES` set in `__init__` shadows it when `n_sites < 3`. All existing `self.SITES` references in `aggregate_embeddings()`, `predict()`, and `get_embedding_gradients()` then automatically use the shorter tuple.

**Add `n_sites` to `TrainConfig`:**
```python
n_sites: int = 3
```

**In `run_training()`**, pass `n_sites` to VFLServer:
```python
server = VFLServer(
    ...
    n_sites=cfg.n_sites,
)
```

And limit active clients:
```python
all_site_dims = {"A": 7, "B": 4, "C": 3}
active_sites = list(all_site_dims.keys())[:cfg.n_sites]
clients = {
    s: VFLClient(input_dim=cfg.site_input_dims.get(s, all_site_dims[s]), ...)
    for s in active_sites
}
```

- [x] **Step 2: Run existing tests after train.py changes**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED.

- [x] **Step 3: Implement `experiments/run_exp4.py`**

```python
"""
experiments/run_exp4.py — Exp 4: Scalability (2 vs. 3 institutions).

Measures:
  - Communication rounds to convergence (loss delta < 0.001 over 5 rounds)
  - Wall-clock time per round
  - Per-task AUC at final round

Configurations: n_sites ∈ {2, 3} × seeds [42, 123, 7]
Output: results/exp4.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import run_training, TrainConfig

SEEDS = [42, 123, 7]


def rounds_to_convergence(losses: list[float], threshold: float = 0.001, window: int = 5) -> int:
    """Return round index where loss delta over `window` rounds drops below threshold."""
    for i in range(window, len(losses)):
        delta = abs(losses[i - window] - losses[i])
        if delta < threshold:
            return i
    return len(losses)  # did not converge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--output",        default="results/exp4.csv")
    parser.add_argument("--use_synthetic", action="store_true")
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
            )
            results = run_training(cfg)
            losses = [r["train_loss"] for r in results]
            conv_round = rounds_to_convergence(losses)
            for r in results:
                all_rows.append({
                    "n_sites": n_sites,
                    "seed": seed,
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
```

- [x] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED.

- [x] **Step 5: Commit**

```bash
git add experiments/run_exp4.py train.py
git commit -m "feat: add Exp4 scalability script; add n_sites support to TrainConfig"
```

---

## Chunk 4: Results and Figures

### ✅ Task 8: Plot utilities and figure scripts

**Context:** Shared utilities load CSVs, compute mean±std across seeds, and produce comparison tables. Three specific figure scripts implement the paper figures.

**Files:**
- Create: `results/plot_results.py`
- Create: `figures/negative_transfer_heatmap.py`
- Create: `figures/scalability_curves.py`
- Create: `figures/feature_split_sensitivity.py`

- [x] **Step 1: Implement `results/plot_results.py`**

```python
"""
results/plot_results.py — Shared utilities for loading and plotting experiment results.

Usage:
    from results.plot_results import load_results, summary_table

    df = load_results("results/exp1.csv")
    tbl = summary_table(df, group_cols=["model"], metric_cols=["ihm_loss"])
    print(tbl)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/HPC use


def load_results(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def summary_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Return mean ± std over seeds grouped by group_cols."""
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    return agg


def loss_curves(
    df: pd.DataFrame,
    group_col: str,
    loss_col: str = "train_loss",
    output_path: str | None = None,
) -> None:
    """Plot mean training loss curves per group (averaged over seeds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby(group_col):
        mean = grp.groupby("round")[loss_col].mean()
        std  = grp.groupby("round")[loss_col].std()
        ax.plot(mean.index, mean.values, label=name)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)
```

- [x] **Step 2: Implement `figures/negative_transfer_heatmap.py`**

```python
"""
figures/negative_transfer_heatmap.py

Task × model heatmap of AUC delta vs. local-only baseline.
Positive = MTL helps; negative = negative transfer.

Usage:
    python figures/negative_transfer_heatmap.py \
        --exp3 results/exp3.csv --output figures/negative_transfer.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_delta_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each task_config, compute mean IHM/LOS/Pheno loss relative to 'ihm_only' baseline.
    Loss delta = baseline_loss - model_loss (positive = model is better).
    """
    # Use final-round loss as proxy for task performance (lower = better)
    final = df.groupby(["task_config", "seed"]).last().reset_index()
    baseline = final[final["task_config"] == "ihm_only"].set_index("seed")

    metrics = ["ihm_loss", "los_loss", "pheno_loss"]
    configs = [c for c in final["task_config"].unique() if c != "ihm_only"]

    rows = []
    for cfg in configs:
        grp = final[final["task_config"] == cfg].set_index("seed")
        row = {"task_config": cfg}
        for m in metrics:
            delta = (baseline[m] - grp[m]).mean()  # positive = MTL reduces loss
            row[m] = round(delta, 4)
        rows.append(row)

    return pd.DataFrame(rows).set_index("task_config")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3",   default="results/exp3.csv")
    parser.add_argument("--output", default="figures/negative_transfer.png")
    args = parser.parse_args()

    df = pd.read_csv(args.exp3)
    delta_df = build_delta_matrix(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        delta_df,
        annot=True, fmt=".3f", center=0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
    )
    ax.set_title("Loss Reduction vs. IHM-only Baseline\n(positive = MTL helps)")
    ax.set_xlabel("Task Loss")
    ax.set_ylabel("Model Configuration")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Implement `figures/scalability_curves.py`**

```python
"""
figures/scalability_curves.py

Rounds-to-convergence vs. number of institutions.

Usage:
    python figures/scalability_curves.py \
        --exp4 results/exp4.csv --output figures/scalability.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp4",   default="results/exp4.csv")
    parser.add_argument("--output", default="figures/scalability.png")
    args = parser.parse_args()

    df = pd.read_csv(args.exp4)

    # One row per (n_sites, seed) — take convergence_round from any row
    summary = df.groupby(["n_sites", "seed"])["convergence_round"].first().reset_index()
    agg = summary.groupby("n_sites")["convergence_round"].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: convergence round
    axes[0].bar(
        agg.index.astype(str), agg["mean"],
        yerr=agg["std"], capsize=5, color=["#4C72B0", "#DD8452"],
    )
    axes[0].set_xlabel("Number of Institutions")
    axes[0].set_ylabel("Rounds to Convergence")
    axes[0].set_title("Convergence Speed vs. n_sites")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: loss curves per n_sites
    for n_sites, grp in df.groupby("n_sites"):
        mean_loss = grp.groupby("round")["train_loss"].mean()
        axes[1].plot(mean_loss.index, mean_loss.values, label=f"{n_sites} sites")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Loss Curves by n_sites")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Implement `figures/feature_split_sensitivity.py`**

```python
"""
figures/feature_split_sensitivity.py

Bar chart of final training loss per feature split configuration (Exp 2).

Usage:
    python figures/feature_split_sensitivity.py \
        --exp2 results/exp2.csv --output figures/feature_split_sensitivity.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp2",   default="results/exp2.csv")
    parser.add_argument("--output", default="figures/feature_split_sensitivity.png")
    args = parser.parse_args()

    df = pd.read_csv(args.exp2)
    final = df.groupby(["split_config", "seed"]).last().reset_index()
    agg = final.groupby("split_config")[["ihm_loss", "los_loss", "pheno_loss"]].agg(
        ["mean", "std"]
    )

    configs = agg.index.tolist()
    tasks = ["ihm_loss", "los_loss", "pheno_loss"]
    task_labels = ["IHM", "LOS", "Pheno"]
    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (task, label) in enumerate(zip(tasks, task_labels)):
        means = agg[task]["mean"].values
        stds  = agg[task]["std"].values
        ax.bar(x + i * width, means, width, yerr=stds, label=label, capsize=4)

    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_xlabel("Feature Split Configuration")
    ax.set_ylabel("Training Loss (lower = better)")
    ax.set_title("Task Loss by Feature Split Configuration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Commit**

```bash
git add results/plot_results.py figures/negative_transfer_heatmap.py \
        figures/scalability_curves.py figures/feature_split_sensitivity.py
git commit -m "feat: add result plotting utilities and three paper figure scripts"
```

---

---

## Chunk 5: Baseline Implementations

### Critical assessment — VFL-SingleTask vs. MARS-VFL

These are **not the same baseline** and serve different purposes.

**VFL-SingleTask (ST-IHM, ST-LOS, ST-Pheno)**
Three fully independent local models — no cross-site embedding exchange. Site A trains only on IHM with its own 7 features; Site B only on LOS; Site C only on Pheno. The VFL framework is used so the architecture (encoder depth, embedding dim) is controlled, but with exactly one task head active per run (task weights = 1/0/0, 0/1/0, 0/0/1). This is the MTL ablation: it answers *"does multi-task learning add value over three independent single-task models?"*

**MARS-VFL**
A collaborative single-task VFL framework where ALL sites exchange embeddings to jointly predict ONE shared task (e.g., IHM across all three sites). The embedding exchange still happens; only the task objective is homogeneous. This is structurally distinct from VFL-SingleTask in two ways: (1) it has full cross-site VFL communication, (2) all sites share the same prediction target. MARS-VFL's paper reports results on 12 non-clinical datasets — direct AUC comparison to your MIMIC numbers is not valid. Use MARS-VFL as a cited VFL literature reference characterising the state-of-the-art VFL paradigm your paper extends, not as a directly reproduced MIMIC number.

**Why you need both:**

| Baseline | Ablates | Comparison answers |
|---|---|---|
| VFL-SingleTask | Multi-task learning | Does MTL help over independent single-task VFL? |
| MARS-VFL (cited) | VFL paradigm context | Where does our VFL-MTL sit relative to prior VFL systems? |
| MOCHA (HFL-MTL) | VFL vs. HFL | Does vertical feature partitioning hurt vs. horizontal patient-split FL? |
| FMTLJD (HFL-MTL) | VFL vs. HFL | Does VFL-MTL outperform best prior HFL-MTL method? |
| Local-only | FL + MTL together | What is the combined benefit of FL and MTL? |
| Centralized oracle | Privacy cost | How much performance do we sacrifice for privacy? |

**Verdict:** VFL-SingleTask ≠ MARS-VFL. VFL-SingleTask has no cross-site communication (pure local). MARS-VFL has full VFL communication but a single shared task — a different configuration not used in Exp 1 per CLAUDE.md because it conflates task heterogeneity with MTL contribution. Keep MARS-VFL as a cited reference, not a reproduced number.

---

### ✅ Baseline 1 — Local-only: `baselines/local_only.py`

**What it is:** Each site trains a standalone LSTM encoder + task-specific head on its own features with no cross-site communication. Lower bound.

**Architecture per site:**
- Site A: `SiteEncoder(input_dim=7)` → `Linear(64,1)+Sigmoid` → BCELoss (IHM)
- Site B: `SiteEncoder(input_dim=4)` → `Linear(64,10)` → CrossEntropyLoss (LOS)
- Site C: `SiteEncoder(input_dim=3)` → `Linear(64,25)+Sigmoid` → BCELoss (Pheno)

- [x] **Step 1: Implement `baselines/local_only.py`**

```python
"""
baselines/local_only.py — Local-only single-task baselines (no FL, no MTL).

One independent model per site. No cross-site communication whatsoever.
Lower bound: what each site achieves with only its own features.

Usage:
    python baselines/local_only.py --site A --splits_dir data/vertical_splits \
        --n_epochs 50 --seeds 42 123 7 --output results/local_only_A.csv
    
    # Smoke test
    python baselines/local_only.py --site A --use_synthetic --n_epochs 2
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.encoder import SiteEncoder
from data_prep.dataset import VFLSiteDataset
from experiments.metrics import ihm_metrics, los_metrics, pheno_metrics


_SITE_INPUT_DIMS = {"A": 7, "B": 4, "C": 3}
_SITE_TASKS      = {"A": "ihm", "B": "los", "C": "pheno"}


class _IHMHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(64, 1)
    def forward(self, e): return torch.sigmoid(self.fc(e)).squeeze(-1)

class _LOSHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(64, 10)
    def forward(self, e): return self.fc(e)   # logits

class _PhenoHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(64, 25)
    def forward(self, e): return torch.sigmoid(self.fc(e))


def _make_head(task: str) -> nn.Module:
    return {"ihm": _IHMHead, "los": _LOSHead, "pheno": _PhenoHead}[task]()

def _make_loss(task: str):
    return {"ihm": nn.BCELoss(), "los": nn.CrossEntropyLoss(), "pheno": nn.BCELoss()}[task]


def _synthetic_loader(site: str, n: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    input_dim = _SITE_INPUT_DIMS[site]
    task = _SITE_TASKS[site]
    x = torch.randn(n, 1, input_dim)
    if task == "ihm":   y = torch.randint(0, 2, (n,)).float()
    elif task == "los": y = torch.randint(0, 10, (n,))
    else:               y = torch.randint(0, 2, (n, 25)).float()
    ds = TensorDataset(x, y)
    class _W:
        def __iter__(self_):
            for xb, yb in DataLoader(ds, batch_size=batch_size): yield xb, yb
    return _W()


def train_local(site: str, splits_dir: str, n_epochs: int, lr: float,
                batch_size: int, seed: int, use_synthetic: bool) -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = _SITE_TASKS[site]

    encoder = SiteEncoder(input_dim=_SITE_INPUT_DIMS[site]).to(device)
    head    = _make_head(task).to(device)
    loss_fn = _make_loss(task)
    opt     = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    if use_synthetic:
        train_loader = _synthetic_loader(site, 128, batch_size, seed)
        val_loader   = _synthetic_loader(site, 32,  batch_size, seed + 1)
    else:
        ds_tr = VFLSiteDataset(splits_dir=splits_dir, split="train")
        ds_va = VFLSiteDataset(splits_dir=splits_dir, split="val")
        # Local-only: load only the relevant site features and task label
        # VFLSiteDataset returns all sites; we extract only what we need
        train_loader = _site_loader(ds_tr, site, task, batch_size, shuffle=True)
        val_loader   = _site_loader(ds_va, site, task, batch_size, shuffle=False)

    rows = []
    for epoch in range(1, n_epochs + 1):
        encoder.train(); head.train()
        total_loss = 0.0; nb = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            emb  = encoder(x)
            pred = head(emb)
            loss = loss_fn(pred, y.long() if task == "los" else y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1

        encoder.eval(); head.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_p.append(head(encoder(x.to(device))).cpu()); all_l.append(y)
        p = torch.cat(all_p); l = torch.cat(all_l)
        m = (ihm_metrics if task == "ihm" else los_metrics if task == "los" else pheno_metrics)(
            p.numpy(), l.numpy()
        )
        rows.append({"model": f"local_{site}", "site": site, "task": task,
                     "epoch": epoch, "train_loss": total_loss / max(nb, 1),
                     "seed": seed, **m})
    return rows


def _site_loader(ds: VFLSiteDataset, site: str, task: str,
                 batch_size: int, shuffle: bool):
    """Extract per-site tensors from VFLSiteDataset as (x, y) tuples."""
    from torch.utils.data import DataLoader as DL
    label_key = f"y_{task}"
    class _W:
        def __iter__(self_):
            for batch in DL(ds, batch_size=batch_size, shuffle=shuffle):
                yield batch[f"x_{site}"], batch[label_key]
    return _W()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",          required=True, choices=["A", "B", "C"])
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--output",        default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()
    out = args.output or f"results/local_only_{args.site}.csv"
    rows = []
    for seed in args.seeds:
        rows.extend(train_local(args.site, args.splits_dir, args.n_epochs,
                                args.lr, args.batch_size, seed, args.use_synthetic))
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: Write `tests/test_local_only.py`**

```python
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.local_only import train_local

@pytest.mark.parametrize("site", ["A", "B", "C"])
def test_smoke(site):
    rows = train_local(site=site, splits_dir="", n_epochs=2,
                       lr=1e-3, batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2
    assert rows[0]["site"] == site
    assert rows[0]["train_loss"] > 0
```

- [ ] **Step 3: Snellius run**

```bash
for SITE in A B C; do
  python baselines/local_only.py --site $SITE \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits/ \
    --n_epochs 50 --output results/local_only_${SITE}.csv
done
# Merge
python -c "import pandas as pd; pd.concat([pd.read_csv(f'results/local_only_{s}.csv') for s in 'ABC']).to_csv('results/local_only.csv', index=False)"
```

- [ ] **Step 4: Commit**

```bash
git add baselines/local_only.py tests/test_local_only.py
git commit -m "feat: add local-only single-task baselines (Sites A/B/C)"
```

---

### ✅ Baseline 2 — Centralized oracle: `baselines/centralized.py`

**What it is:** All 14 features concatenated, single model trained jointly on all three tasks. No privacy constraints. Upper bound. The performance gap between this and VFL-MTL quantifies the privacy cost.

**Architecture:** `CentralizedEncoder(input_dim=14)` (LSTM, hidden=128, layers=2, embed=64) → `MMoEServer` with `total_embed_dim=64` (single site, no concatenation). All three task heads active simultaneously.

**Dependency:** `MMoEServer` must accept a `total_embed_dim` constructor arg. Check `model/mmoe.py` — if it hardcodes `total_embed_dim=192`, add it as a configurable parameter with `default=192`.

- [x] **Step 1: Check and update `model/mmoe.py` if needed**

Open `model/mmoe.py` and verify `MMoEServer.__init__` signature. If `total_embed_dim` is hardcoded to `embed_dim * n_sites`, add it as an explicit `__init__` parameter with `default=192` so the centralized baseline can pass `total_embed_dim=64`.

- [x] **Step 2: Implement `baselines/centralized.py`**

```python
"""
baselines/centralized.py — Centralized oracle baseline (no privacy constraints).

All 14 features from all three sites concatenated into a single LSTM encoder.
MMoE server with all three task heads trained jointly. Upper bound.

Usage:
    python baselines/centralized.py --splits_dir data/vertical_splits \
        --n_epochs 50 --seeds 42 123 7 --output results/centralized.csv

    # Smoke test
    python baselines/centralized.py --use_synthetic --n_epochs 2
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mmoe import MMoEServer
from data_prep.dataset import VFLSiteDataset
from experiments.metrics import ihm_metrics, los_metrics, pheno_metrics


class CentralizedEncoder(nn.Module):
    """Single LSTM consuming all 14 features. Produces 64-dim embedding."""
    def __init__(self, input_dim: int = 14, hidden: int = 128,
                 layers: int = 2, embed_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.proj = nn.Linear(hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.norm(self.proj(h[-1]))   # (B, 64)


def _synthetic_loader(n: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    xa = torch.randn(n, 1, 7); xb = torch.randn(n, 1, 4); xc = torch.randn(n, 1, 3)
    yi = torch.randint(0, 2, (n,)).float()
    yl = torch.randint(0, 10, (n,))
    yp = torch.randint(0, 2, (n, 25)).float()
    ds = TensorDataset(xa, xb, xc, yi, yl, yp)
    class _W:
        def __iter__(self_):
            for a, b, c, i, l, p in DataLoader(ds, batch_size=batch_size):
                yield {"x_A": a, "x_B": b, "x_C": c, "y_ihm": i, "y_los": l, "y_pheno": p}
    return _W()


def train_centralized(splits_dir: str, n_epochs: int, lr: float,
                      batch_size: int, seed: int, use_synthetic: bool) -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CentralizedEncoder(input_dim=14).to(device)
    server  = MMoEServer(total_embed_dim=64, n_experts=4, expert_hidden=128).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(server.parameters()), lr=lr)
    ihm_fn = nn.BCELoss(); los_fn = nn.CrossEntropyLoss(); ph_fn = nn.BCELoss()

    if use_synthetic:
        train_loader = _synthetic_loader(128, batch_size, seed)
        val_loader   = _synthetic_loader(32,  batch_size, seed + 1)
    else:
        ds_tr = VFLSiteDataset(splits_dir=splits_dir, split="train")
        ds_va = VFLSiteDataset(splits_dir=splits_dir, split="val")
        train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    rows = []
    for epoch in range(1, n_epochs + 1):
        encoder.train(); server.train()
        total_loss = 0.0; nb = 0
        for batch in train_loader:
            x = torch.cat([batch["x_A"], batch["x_B"], batch["x_C"]], dim=-1).to(device)
            emb = encoder(x)
            out = server(emb)
            loss = (ihm_fn(out["ihm"], batch["y_ihm"].to(device))
                    + los_fn(out["los"], batch["y_los"].to(device).long())
                    + ph_fn(out["pheno"], batch["y_pheno"].to(device)))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1

        encoder.eval(); server.eval()
        ihm_p, ihm_l = [], []
        los_p, los_l = [], []
        ph_p,  ph_l  = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = torch.cat([batch["x_A"], batch["x_B"], batch["x_C"]], dim=-1).to(device)
                out = server(encoder(x))
                ihm_p.append(out["ihm"].cpu()); ihm_l.append(batch["y_ihm"])
                los_p.append(out["los"].cpu()); los_l.append(batch["y_los"])
                ph_p.append(out["pheno"].cpu()); ph_l.append(batch["y_pheno"])

        m_ihm   = ihm_metrics(torch.cat(ihm_p).numpy(), torch.cat(ihm_l).numpy())
        m_los   = los_metrics(torch.cat(los_p).numpy(), torch.cat(los_l).numpy())
        m_pheno = pheno_metrics(torch.cat(ph_p).numpy(), torch.cat(ph_l).numpy())
        rows.append({"model": "centralized_oracle", "epoch": epoch, "seed": seed,
                     "train_loss": total_loss / max(nb, 1),
                     **{f"ihm_{k}": v for k, v in m_ihm.items()},
                     **{f"los_{k}": v for k, v in m_los.items()},
                     **{f"pheno_{k}": v for k, v in m_pheno.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--output",        default="results/centralized.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.extend(train_centralized(args.splits_dir, args.n_epochs, args.lr,
                                     args.batch_size, seed, args.use_synthetic))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False); print(f"Saved {len(df)} rows → {args.output}")

if __name__ == "__main__":
    main()
```

- [x] **Step 3: Write `tests/test_centralized.py`**

```python
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.centralized import train_centralized

def test_smoke():
    rows = train_centralized(splits_dir="", n_epochs=2, lr=1e-3,
                             batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2
    assert rows[0]["model"] == "centralized_oracle"
    assert "ihm_auroc" in rows[0]
    assert "los_kappa" in rows[0]
    assert "pheno_macro_auc" in rows[0]
```

- [ ] **Step 4: Commit**

```bash
git add baselines/centralized.py tests/test_centralized.py
git commit -m "feat: add centralized oracle baseline (all 14 features, all tasks)"
```

---

### ✅ Baseline 3 — VFL-SingleTask: `baselines/vfl_singletask.py`

**What it is:** Full VFL framework (cross-site embedding exchange, server) with exactly one task active per run. Task weights = `(1,0,0)` for ST-IHM, `(0,1,0)` for ST-LOS, `(0,0,1)` for ST-Pheno. The architecture is identical to VFL-MTL — only the task objective differs. This cleanly isolates the MTL contribution.

**Distinction from Local-only:** Embedding exchange still happens (all sites contribute to the server). Only the gradient signal comes from one task head. In Local-only there is no server at all.

**Implementation:** Covered by `run_exp1.py` — ST-IHM, ST-LOS, ST-Pheno configs are already defined in `CONFIGS` and run alongside VFL-MTL. A separate `baselines/vfl_singletask.py` script is not needed.

- [ ] **Step 1: Implement `baselines/vfl_singletask.py`**

```python
"""
baselines/vfl_singletask.py — VFL-SingleTask baselines (ST-IHM, ST-LOS, ST-Pheno).

Runs the full VFL framework with one task head active per run. Used as MTL ablation:
VFL-MTL minus multi-task = VFL-SingleTask. The architecture is identical; only
the task weights change.

Usage:
    python baselines/vfl_singletask.py --task ihm \
        --splits_dir data/vertical_splits --n_rounds 100 \
        --seeds 42 123 7 --output results/vfl_singletask_ihm.csv

    # Smoke test
    python baselines/vfl_singletask.py --task ihm --use_synthetic --n_rounds 2
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import TrainConfig, run_training


_TASK_WEIGHTS = {
    "ihm":   (1.0, 0.0, 0.0),
    "los":   (0.0, 1.0, 0.0),
    "pheno": (0.0, 0.0, 1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",          required=True, choices=["ihm", "los", "pheno"])
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--output",        default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    out = args.output or f"results/vfl_singletask_{args.task}.csv"
    weights = _TASK_WEIGHTS[args.task]
    rows = []
    for seed in args.seeds:
        cfg = TrainConfig(
            splits_dir=args.splits_dir, n_rounds=args.n_rounds,
            lr=args.lr, batch_size=args.batch_size, seed=seed,
            use_synthetic=args.use_synthetic,
            task_weights=weights, aggregation="fedavg",
        )
        r = run_training(cfg)
        for row in r:
            row["model"] = f"vfl_st_{args.task}"; row["active_task"] = args.task
        rows.extend(r)

    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False); print(f"Saved {len(df)} rows → {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
python baselines/vfl_singletask.py --task ihm --use_synthetic --n_rounds 2
python baselines/vfl_singletask.py --task los --use_synthetic --n_rounds 2
python baselines/vfl_singletask.py --task pheno --use_synthetic --n_rounds 2
```

- [ ] **Step 3: Snellius run**

```bash
for TASK in ihm los pheno; do
  python baselines/vfl_singletask.py --task $TASK \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits/ \
    --n_rounds 100 --output results/vfl_singletask_${TASK}.csv
done
```

- [ ] **Step 4: Commit**

```bash
git add baselines/vfl_singletask.py
git commit -m "feat: add VFL-SingleTask convenience script (MTL ablation)"
```

---

### Baseline 4 — MOCHA: `baselines/mocha.py`

**What it is:** MOCHA (Smith et al., 2017) is an HFL-MTL method. It is run here as **true HFL** (not adapted to VFL): the PSI-aligned patient cohort is split horizontally across three clients, and each client receives **all 14 features** for its patient subset. This demonstrates whether the VFL feature partitioning constraint actually costs performance vs. having all sites run HFL with full feature access.

**Why this framing matters:** By running MOCHA as HFL with full features, you isolate the effect of vertical partitioning. If VFL-MTL approaches MOCHA's HFL performance, it confirms that VFL does not sacrifice much accuracy to achieve feature privacy. If it underperforms substantially, it quantifies the VFL privacy cost.

**Data setup:** Split the PSI-aligned train/val/test patients into three equal random horizontal subsets (by patient ID). Each MOCHA client loads all 14 features + all 3 task labels for its subset. The client index is not meaningful (no per-client task assignment — MOCHA learns task relationships across all clients jointly).

**Architecture per client:** `CentralizedEncoder(input_dim=14)` → three task heads (IHM, LOS, Pheno). MOCHA's server step updates task relationship matrix `Ω` from client gradient correlations.

- [ ] **Step 1: Implement `baselines/mocha.py`**

```python
"""
baselines/mocha.py — MOCHA baseline (true HFL-MTL, all 14 features per client).

Reference: Smith et al. (2017), "Federated Multi-Task Learning", NeurIPS.
https://dl.acm.org/doi/10.5555/3294996.3295196

Data setup: PSI-aligned patients are split HORIZONTALLY (by patient ID) into 3 equal
subsets. Each client receives all 14 features + all 3 task labels for its subset.
This simulates the HFL scenario: 3 hospitals, each with full feature access but
non-overlapping patient populations.

MOCHA learns inter-task relationships via a task relationship matrix Omega.
Client update: minimize local loss + lambda * sum_j(Omega_jj' * ||w_j - w_j'||^2)
Server update: Omega updated from pairwise gradient correlations across clients.

Simplified adaptation: Omega operates on the 64-dim embedding (not full param space)
for tractability. Omega_ij = cosine similarity between client i and j embeddings.

Usage:
    python baselines/mocha.py --splits_dir data/vertical_splits \
        --n_rounds 100 --lam 0.1 --seeds 42 123 7 --output results/mocha.csv

    # Smoke test
    python baselines/mocha.py --use_synthetic --n_rounds 2
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.centralized import CentralizedEncoder
from data_prep.dataset import VFLSiteDataset
from experiments.metrics import ihm_metrics, los_metrics, pheno_metrics


_N_CLIENTS = 3
_EMBED_DIM = 64
_TASKS     = ["ihm", "los", "pheno"]


class _MultiTaskHead(nn.Module):
    """Three task heads sharing a common encoder."""
    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.ihm_head   = nn.Sequential(nn.Linear(embed_dim, 1),  nn.Sigmoid())
        self.los_head   = nn.Linear(embed_dim, 10)
        self.pheno_head = nn.Sequential(nn.Linear(embed_dim, 25), nn.Sigmoid())

    def forward(self, emb: torch.Tensor) -> dict:
        return {
            "ihm":   self.ihm_head(emb).squeeze(-1),
            "los":   self.los_head(emb),
            "pheno": self.pheno_head(emb),
        }


def _task_loss(out: dict, batch: dict, device) -> torch.Tensor:
    return (nn.BCELoss()(out["ihm"],   batch["y_ihm"].to(device))
            + nn.CrossEntropyLoss()(out["los"], batch["y_los"].to(device).long())
            + nn.BCELoss()(out["pheno"], batch["y_pheno"].to(device)))


def _make_hfl_loaders(ds: VFLSiteDataset, batch_size: int, seed: int) -> list:
    """
    Horizontally split VFLSiteDataset into N_CLIENTS equal subsets by patient order.
    Each client gets all 14 features (x_A + x_B + x_C concatenated) + all labels.
    """
    n = len(ds)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    splits = np.array_split(indices, _N_CLIENTS)

    loaders = []
    for idx_arr in splits:
        subset_samples = [ds[i] for i in idx_arr]
        x_all   = torch.cat([torch.cat([s["x_A"], s["x_B"], s["x_C"]], dim=-1).unsqueeze(0)
                              for s in subset_samples])   # (n_client, 1, 14)
        y_ihm   = torch.stack([s["y_ihm"]   for s in subset_samples])
        y_los   = torch.stack([s["y_los"]   for s in subset_samples])
        y_pheno = torch.stack([s["y_pheno"] for s in subset_samples])
        client_ds = TensorDataset(x_all, y_ihm, y_los, y_pheno)
        class _W:
            def __init__(self_, ds_):
                self_._dl = DataLoader(ds_, batch_size=batch_size, shuffle=True)
            def __iter__(self_):
                for x, yi, yl, yp in self_._dl:
                    yield {"x_all": x, "y_ihm": yi, "y_los": yl, "y_pheno": yp}
        loaders.append(_W(client_ds))
    return loaders


def _synthetic_hfl_loaders(batch_size: int, seed: int, n_per_client: int = 64):
    torch.manual_seed(seed)
    loaders = []
    for i in range(_N_CLIENTS):
        torch.manual_seed(seed + i)
        x = torch.randn(n_per_client, 1, 14)
        yi = torch.randint(0, 2, (n_per_client,)).float()
        yl = torch.randint(0, 10, (n_per_client,))
        yp = torch.randint(0, 2, (n_per_client, 25)).float()
        ds = TensorDataset(x, yi, yl, yp)
        class _W:
            def __init__(self_, dl_): self_._dl = dl_
            def __iter__(self_):
                for xb, yib, ylb, ypb in self_._dl:
                    yield {"x_all": xb, "y_ihm": yib, "y_los": ylb, "y_pheno": ypb}
        loaders.append(_W(DataLoader(ds, batch_size=batch_size, shuffle=True)))
    return loaders


def train_mocha(splits_dir: str, n_rounds: int, lam: float,
                lr: float, batch_size: int, seed: int, use_synthetic: bool) -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = [CentralizedEncoder(input_dim=14).to(device) for _ in range(_N_CLIENTS)]
    heads    = [_MultiTaskHead().to(device)                  for _ in range(_N_CLIENTS)]
    opts     = [torch.optim.Adam(
                    list(encoders[i].parameters()) + list(heads[i].parameters()), lr=lr)
                for i in range(_N_CLIENTS)]

    # Omega: pairwise client task-relationship matrix (N_CLIENTS x N_CLIENTS)
    Omega = torch.eye(_N_CLIENTS, device=device)

    if use_synthetic:
        train_loaders = _synthetic_hfl_loaders(batch_size, seed)
        val_loaders   = _synthetic_hfl_loaders(batch_size, seed + 100, n_per_client=32)
    else:
        ds_tr = VFLSiteDataset(splits_dir=splits_dir, split="train")
        ds_va = VFLSiteDataset(splits_dir=splits_dir, split="val")
        train_loaders = _make_hfl_loaders(ds_tr, batch_size, seed)
        val_loaders   = _make_hfl_loaders(ds_va, batch_size, seed)

    rows = []
    for rnd in range(1, n_rounds + 1):
        for enc in encoders: enc.train()
        for h   in heads:    h.train()
        round_loss = 0.0; nb = 0

        # Interleave batches across clients
        for batches in zip(*[iter(loader) for loader in train_loaders]):
            embs = []
            for i in range(_N_CLIENTS):
                x = batches[i]["x_all"].to(device)
                embs.append(encoders[i](x))   # (B, 64)

            for i in range(_N_CLIENTS):
                out  = heads[i](embs[i])
                task_l = _task_loss(out, batches[i], device)

                # MOCHA regularizer on embedding vectors
                mocha_reg = sum(
                    Omega[i, j] * (embs[i] - embs[j].detach()).pow(2).mean()
                    for j in range(_N_CLIENTS) if j != i
                )
                loss = task_l + lam * mocha_reg
                opts[i].zero_grad(); loss.backward(retain_graph=(i < _N_CLIENTS - 1))
                opts[i].step()
                round_loss += task_l.item(); nb += 1

        # Server: update Omega from embedding cosine similarities
        with torch.no_grad():
            sample = next(iter(train_loaders[0]))
            sample_embs = [encoders[i](sample["x_all"].to(device)) for i in range(_N_CLIENTS)]
            for i in range(_N_CLIENTS):
                for j in range(_N_CLIENTS):
                    if i != j:
                        cos = torch.nn.functional.cosine_similarity(
                            sample_embs[i].flatten().unsqueeze(0),
                            sample_embs[j].flatten().unsqueeze(0),
                        ).clamp(0.0, 1.0)
                        Omega[i, j] = cos.item()

        # Validation: aggregate predictions across clients (simple average)
        for enc in encoders: enc.eval()
        for h   in heads:    h.eval()
        ihm_p, ihm_l = [], []
        los_p, los_l = [], []
        ph_p,  ph_l  = [], []
        with torch.no_grad():
            for loader in val_loaders:
                for batch in loader:
                    x = batch["x_all"].to(device)
                    # Average task head outputs across clients for each sample
                    ihm_out   = sum(heads[i](encoders[i](x))["ihm"]   for i in range(_N_CLIENTS)) / _N_CLIENTS
                    los_out   = sum(heads[i](encoders[i](x))["los"]   for i in range(_N_CLIENTS)) / _N_CLIENTS
                    pheno_out = sum(heads[i](encoders[i](x))["pheno"] for i in range(_N_CLIENTS)) / _N_CLIENTS
                    ihm_p.append(ihm_out.cpu()); ihm_l.append(batch["y_ihm"])
                    los_p.append(los_out.cpu()); los_l.append(batch["y_los"])
                    ph_p.append(pheno_out.cpu()); ph_l.append(batch["y_pheno"])

        m_ihm   = ihm_metrics(torch.cat(ihm_p).numpy(), torch.cat(ihm_l).numpy())
        m_los   = los_metrics(torch.cat(los_p).numpy(), torch.cat(los_l).numpy())
        m_pheno = pheno_metrics(torch.cat(ph_p).numpy(), torch.cat(ph_l).numpy())
        rows.append({
            "model": "mocha", "round": rnd, "seed": seed,
            "train_loss": round_loss / max(nb, 1),
            "omega_01": Omega[0,1].item(), "omega_02": Omega[0,2].item(),
            "omega_12": Omega[1,2].item(),
            **{f"ihm_{k}":   v for k, v in m_ihm.items()},
            **{f"los_{k}":   v for k, v in m_los.items()},
            **{f"pheno_{k}": v for k, v in m_pheno.items()},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int,   default=100)
    parser.add_argument("--lam",           type=float, default=0.1)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--output",        default="results/mocha.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.extend(train_mocha(args.splits_dir, args.n_rounds, args.lam,
                                args.lr, args.batch_size, seed, args.use_synthetic))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False); print(f"Saved {len(df)} rows → {args.output}")

if __name__ == "__main__":
    main()
```

**Hyperparameter grid for λ:** Run `lam ∈ {0.01, 0.1, 1.0}` on seed=42 for 20 rounds; pick the λ with best mean val AUC across tasks. Report selected λ in paper.

- [ ] **Step 2: Write `tests/test_mocha.py`**

```python
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.mocha import train_mocha

def test_smoke():
    rows = train_mocha(splits_dir="", n_rounds=2, lam=0.1, lr=1e-3,
                       batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2
    assert rows[0]["model"] == "mocha"
    assert 0.0 <= rows[0]["omega_01"] <= 1.0
    assert "ihm_auroc" in rows[0]
```

- [ ] **Step 3: Commit**

```bash
git add baselines/mocha.py tests/test_mocha.py
git commit -m "feat: add MOCHA baseline (HFL-MTL, horizontal patient split, all 14 features)"
```

---

### Baseline 5 — FMTLJD: `baselines/fmtljd.py`

**What it is:** FMTLJD (Huang et al., 2023) extends HFL-MTL with joint diagonalization (JAD) of per-task gradient covariance matrices to discover shared latent task structure. Same HFL data setup as MOCHA: three clients, each with a horizontal patient subset and all 14 features. The difference from MOCHA is the server step: instead of cosine-similarity Omega, FMTLJD replaces the encoder's projection weights with a jointly diagonalized subspace.

**JAD implementation:** For each round, accumulate per-task gradient covariance matrices `C_k = G_k^T G_k` (where `G_k` = gradient of the projection layer for task k across batches). The joint diagonalizer `W` minimizes `∑_k ||W C_k W^T - diag||_F^2`. `W` is found via power iteration (Yeredor, 2000 approximation). At round end, project each client's projection layer: `enc.proj.weight ← W @ enc.proj.weight`.

**Tractability:** Applied to the 64-dim projection layer output space only (not the full 128-dim hidden state or all LSTM weights). JAD on a 64×64 matrix is fast (< 1 ms per round).

- [ ] **Step 1: Implement `baselines/fmtljd.py`**

```python
"""
baselines/fmtljd.py — FMTLJD baseline (HFL-MTL with joint gradient diagonalization).

Reference: Huang et al. (2023), "Federated Multi-Task Learning for Joint Diagnosis
of Multiple Mental Disorders on MRI Scans", IEEE TBME.
https://doi.org/10.1109/TBME.2022.3210940

Data setup: identical to MOCHA — horizontal patient split, all 14 features per client.
Key difference from MOCHA: server step uses JAD of gradient covariance matrices
to discover the shared latent subspace, replacing cosine-similarity Omega update.

JAD approximation: Yeredor (2000) power-iteration on the 64-dim projection space.
Applied per-round; each client's enc.proj.weight is projected onto the joint subspace.

Usage:
    python baselines/fmtljd.py --splits_dir data/vertical_splits \
        --n_rounds 100 --seeds 42 123 7 --output results/fmtljd.csv

    # Smoke test
    python baselines/fmtljd.py --use_synthetic --n_rounds 2
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.centralized import CentralizedEncoder
from baselines.mocha import (
    _MultiTaskHead, _task_loss, _make_hfl_loaders, _synthetic_hfl_loaders,
)
from data_prep.dataset import VFLSiteDataset
from experiments.metrics import ihm_metrics, los_metrics, pheno_metrics


_N_CLIENTS = 3
_EMBED_DIM = 64


def _joint_diagonalizer(cov_list: list[np.ndarray], n_iter: int = 10) -> np.ndarray:
    """
    Approximate joint diagonalizer for a list of symmetric positive-semi-definite matrices.

    W = argmin_W  sum_k || W C_k W^T - diag(W C_k W^T) ||_F^2
        s.t. W orthogonal

    Initialization: eigenvectors of mean covariance.
    Refinement: Yeredor-style gradient descent with QR re-orthogonalization.

    Returns W: (d, d) orthogonal matrix.
    """
    d = cov_list[0].shape[0]
    C_mean = np.mean(cov_list, axis=0)
    _, evecs = np.linalg.eigh(C_mean)   # ascending eigenvalues; columns = eigenvectors
    W = evecs.T.copy()                  # (d, d); each row is an eigenvector

    step = 0.01
    for _ in range(n_iter):
        grad = np.zeros_like(W)
        for C in cov_list:
            WC  = W @ C
            WCWt = WC @ W.T
            off  = WCWt - np.diag(np.diag(WCWt))   # zero diagonal, keep off-diag
            grad += 2.0 * off @ WC                   # gradient wrt W
        W -= step * grad
        W, _ = np.linalg.qr(W.T)   # re-orthogonalize
        W = W.T

    return W   # (d, d) orthogonal


def train_fmtljd(splits_dir: str, n_rounds: int, lr: float,
                 batch_size: int, seed: int, use_synthetic: bool) -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = [CentralizedEncoder(input_dim=14).to(device) for _ in range(_N_CLIENTS)]
    heads    = [_MultiTaskHead().to(device)                  for _ in range(_N_CLIENTS)]
    opts     = [torch.optim.Adam(
                    list(encoders[i].parameters()) + list(heads[i].parameters()), lr=lr)
                for i in range(_N_CLIENTS)]

    if use_synthetic:
        train_loaders = _synthetic_hfl_loaders(batch_size, seed)
        val_loaders   = _synthetic_hfl_loaders(batch_size, seed + 100, n_per_client=32)
    else:
        ds_tr = VFLSiteDataset(splits_dir=splits_dir, split="train")
        ds_va = VFLSiteDataset(splits_dir=splits_dir, split="val")
        train_loaders = _make_hfl_loaders(ds_tr, batch_size, seed)
        val_loaders   = _make_hfl_loaders(ds_va, batch_size, seed)

    rows = []
    for rnd in range(1, n_rounds + 1):
        for enc in encoders: enc.train()
        for h   in heads:    h.train()
        round_loss = 0.0; nb = 0

        # Per-task gradient covariance accumulators (one per client, one per task)
        # Shape: (N_CLIENTS, EMBED_DIM, EMBED_DIM)
        cov_accum = np.zeros((_N_CLIENTS, _EMBED_DIM, _EMBED_DIM))

        for batches in zip(*[iter(loader) for loader in train_loaders]):
            for i in range(_N_CLIENTS):
                opts[i].zero_grad()
                x   = batches[i]["x_all"].to(device)
                emb = encoders[i](x)
                out = heads[i](emb)
                loss = _task_loss(out, batches[i], device)
                loss.backward()

                # Accumulate gradient covariance on proj layer (64 x 128 weight)
                g = encoders[i].proj.weight.grad
                if g is not None:
                    g_np = g.detach().cpu().numpy()   # (64, 128)
                    cov_accum[i] += (g_np @ g_np.T) / max(batch_size, 1)

                opts[i].step()
                round_loss += loss.item(); nb += 1

        # FMTLJD server step: JAD of per-client gradient covariance matrices
        # Symmetrize + regularize each covariance
        cov_list = [(c + c.T) / 2 + np.eye(_EMBED_DIM) * 1e-6 for c in cov_accum]
        W = _joint_diagonalizer(cov_list, n_iter=5)   # (64, 64) orthogonal
        W_t = torch.tensor(W, dtype=torch.float32, device=device)

        # Project each encoder's proj.weight onto the shared subspace
        with torch.no_grad():
            for enc in encoders:
                enc.proj.weight.data = W_t @ enc.proj.weight.data   # (64, 128)

        # Validation (same aggregation as MOCHA: average across clients)
        for enc in encoders: enc.eval()
        for h   in heads:    h.eval()
        ihm_p, ihm_l = [], []
        los_p, los_l = [], []
        ph_p,  ph_l  = [], []
        with torch.no_grad():
            for loader in val_loaders:
                for batch in loader:
                    x = batch["x_all"].to(device)
                    ihm_out   = sum(heads[i](encoders[i](x))["ihm"]   for i in range(_N_CLIENTS)) / _N_CLIENTS
                    los_out   = sum(heads[i](encoders[i](x))["los"]   for i in range(_N_CLIENTS)) / _N_CLIENTS
                    pheno_out = sum(heads[i](encoders[i](x))["pheno"] for i in range(_N_CLIENTS)) / _N_CLIENTS
                    ihm_p.append(ihm_out.cpu()); ihm_l.append(batch["y_ihm"])
                    los_p.append(los_out.cpu()); los_l.append(batch["y_los"])
                    ph_p.append(pheno_out.cpu()); ph_l.append(batch["y_pheno"])

        m_ihm   = ihm_metrics(torch.cat(ihm_p).numpy(), torch.cat(ihm_l).numpy())
        m_los   = los_metrics(torch.cat(los_p).numpy(), torch.cat(los_l).numpy())
        m_pheno = pheno_metrics(torch.cat(ph_p).numpy(), torch.cat(ph_l).numpy())
        rows.append({
            "model": "fmtljd", "round": rnd, "seed": seed,
            "train_loss": round_loss / max(nb, 1),
            **{f"ihm_{k}":   v for k, v in m_ihm.items()},
            **{f"los_{k}":   v for k, v in m_los.items()},
            **{f"pheno_{k}": v for k, v in m_pheno.items()},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--output",        default="results/fmtljd.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.extend(train_fmtljd(args.splits_dir, args.n_rounds, args.lr,
                                 args.batch_size, seed, args.use_synthetic))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False); print(f"Saved {len(df)} rows → {args.output}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_fmtljd.py`**

```python
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.fmtljd import train_fmtljd, _joint_diagonalizer

def test_joint_diagonalizer_orthogonal():
    covs = [np.eye(8) + np.random.default_rng(i).random((8,8)) * 0.01 for i in range(3)]
    W = _joint_diagonalizer(covs, n_iter=3)
    assert W.shape == (8, 8)
    assert np.allclose(W @ W.T, np.eye(8), atol=0.1)   # approximately orthogonal

def test_smoke():
    rows = train_fmtljd(splits_dir="", n_rounds=2, lr=1e-3,
                        batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2
    assert rows[0]["model"] == "fmtljd"
    assert "ihm_auroc" in rows[0]
```

- [ ] **Step 3: Commit**

```bash
git add baselines/fmtljd.py tests/test_fmtljd.py
git commit -m "feat: add FMTLJD baseline (HFL-MTL with JAD, horizontal patient split)"
```

---

### Baseline 6 — MARS-VFL: `baselines/mars_vfl.py`

**Submodule:** `baselines/MARS-VFL/` (git submodule → `https://github.com/shentt67/MARS-VFL`).

**What it is:** MARS-VFL (NeurIPS 2025) is a unified single-task VFL benchmark. Their paradigm: each passive client runs a local encoder on its feature partition → sends embedding to active client (server) → server concatenates → predicts one shared task → returns per-client gradient slices. We call their `base()` function **directly** from the submodule by wrapping our `SiteEncoder` and IHM head to match their `model_list` interface. We do NOT use their `Healthcare_VFL` dataset or their `LocalModelForMIMIC` (GRU/MLP with 2-client pickle format) — we replace data and models while running their exact training code.

**What this ablates:** Fills the missing cell in the ablation table — multi-site VFL with a single homogeneous task. Directly answers whether task heterogeneity is what drives VFL-MTL gains, not VFL collaboration per se.

| | Single-task | Multi-task |
|---|---|---|
| **No cross-site comms** | Local-only (Baseline 1) | — (not meaningful) |
| **Cross-site VFL** | **MARS-VFL (this)** | **VFL-MTL (proposed)** |

**Architecture:** Three `SiteEncoder` instances (input_dim=7/4/3, hidden=128, embed=64) as local models, identical to VFL-MTL. Server head: `Linear(192, 128) → ReLU → Linear(128, 1) → Sigmoid` → BCELoss on IHM. No MMoE. No FedAvg (MARS-VFL `base()` does not aggregate encoder weights across clients — clients have different feature partitions so full param averaging is architecturally invalid).

**Adapter design — how we plug into `base()`:**

`base(trainer, model_list, input, ...)` expects:
- `model_list[0]` = global model: `forward(local_output_list)` where `local_output_list` is a list of `(B, embed_dim)` tensors → returns `(B, n_classes)` logits
- `model_list[1..N]` = local models: `forward(x_split_i)` where `x_split_i` is the site's input → returns `(B, embed_dim)`
- `input[0]` = `x` passed to `prepare_data_vfl(x, args)` → must return `[x_split_0, ..., x_split_N]`
- `input[-2]` = `y` (labels)
- `trainer.args.client_num = 3`, `trainer.args.dataset = 'VFL_MTL'`
- `trainer.criterion` = `nn.BCELoss()` operating on sigmoid output vs float y_ihm

We provide:
1. **`_SiteEncoderWrapper(nn.Module)`** — wraps `SiteEncoder`; `forward((x, mask))` calls `SiteEncoder.forward(x, mask)` → `(B, 64)`. Receives tuple from `prepare_data_vfl`.
2. **`_IHMGlobalModel(nn.Module)`** — `forward([emb_A, emb_B, emb_C])` → `torch.cat(embs, dim=-1)` → `MLP(192→128→1)` → `Sigmoid` → `(B,)`. Compatible with `base()` global model call.
3. **Monkeypatch `prepare_data_vfl`** — before calling `base()`, register handler for `dataset='VFL_MTL'` with `client_num=3` that returns `[(x_A, mask_A), (x_B, mask_B), (x_C, mask_C)]` from the pre-split batch.
4. **Minimal `_Trainer` stub** — a simple namespace providing `device`, `args`, `criterion`, `optimizer_list` so `base()` can run without `Trainer_Efficiency`'s full logging infrastructure.

**VFL forward pass protocol** (mirrors MARS-VFL `Base.py`):
1. Each `VFLClient.forward(x, mask)` → returns detached leaf embedding (stores original for backward)
2. Server: `concat = torch.cat(embs, dim=-1); concat.retain_grad()`
3. `pred = head(concat); loss = BCELoss(pred, y_ihm); head_opt.zero_grad(); loss.backward()`
4. `concat.grad.split(64, dim=-1)` → per-site gradient slices
5. Each `VFLClient.receive_gradient(grad_slice)` → backprop local LSTM + optimizer step

**Real data interface:** Use `VFLSiteDataset` from `data_prep/dataset.py`. Each site dataset returns `(x, mask, y)` tuples; Site A `y` is `y_ihm`. Three per-site `DataLoader`s are iterated in lockstep via `zip(loader_A, loader_B, loader_C)`.

**Scope note:** IHM is the shared task (primary clinical outcome; most common in single-task VFL benchmarks). LOS and Pheno are out-of-scope for this baseline. Report MARS-VFL IHM AUC-ROC vs. VFL-MTL IHM AUC-ROC.

---

- [ ] **Step 1: Implement `baselines/mars_vfl.py`**

```python
"""
baselines/mars_vfl.py — MARS-VFL paradigm baseline (single-task VFL on MIMIC-III).

Calls baselines/MARS-VFL/method/Base.py::base() directly from the submodule.
We plug in our own models and data via an adapter layer:
  - _SiteEncoderWrapper: wraps SiteEncoder to match base()'s model_list[i] interface
  - _IHMGlobalModel: concat(3×64) → MLP(192,128,1) + Sigmoid, matches model_list[0]
  - monkeypatched prepare_data_vfl: returns [(x_A,mask_A),(x_B,mask_B),(x_C,mask_C)]
  - _Trainer stub: minimal namespace so base() runs without Trainer_Efficiency logging

We do NOT use their Healthcare_VFL dataset (requires im.pk pickle) or their
LocalModelForMIMIC (GRU/MLP for 2-client setup) — just their training protocol.

Reference: MARS-VFL (NeurIPS 2025) https://openreview.net/forum?id=4Ud0pRqFto

Usage:
    # Smoke test (no MIMIC data needed):
    python baselines/mars_vfl.py --use_synthetic --n_rounds 2

    # Full run (Snellius):
    python baselines/mars_vfl.py \
        --root /home/asoare/vfl_mlt --n_rounds 100 \
        --seeds 42 123 7 --output results/mars_vfl.csv
"""

from __future__ import annotations
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient
from data_prep.dataset import VFLSiteDataset, collate_fn
from experiments.metrics import ihm_metrics


_SITE_NAMES      = ["A", "B", "C"]
_SITE_INPUT_DIMS = {"A": 7, "B": 4, "C": 3}
_EMBED_DIM       = 64
_TOTAL_EMBED     = _EMBED_DIM * 3          # 192 — concat of three site embeddings


# ---------------------------------------------------------------------------
# Server: single IHM head (no MMoE — vanilla MARS-VFL base paradigm)
# ---------------------------------------------------------------------------

class _MARSVFLServer(nn.Module):
    """
    Concat(192) → Linear(192,128) → ReLU → Linear(128,1) → Sigmoid.
    Matches MARS-VFL GlobalModelForMIMIC adapted to our embed_dim.
    """
    def __init__(self, total_embed: int = _TOTAL_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_embed, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, concat: torch.Tensor) -> torch.Tensor:
        return self.net(concat).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Synthetic loaders (smoke test — no MIMIC data needed)
# Provides masks because SiteEncoder.forward(x, mask) requires them.
# ---------------------------------------------------------------------------

def _synthetic_loaders(batch_size: int, seed: int) -> dict[str, DataLoader]:
    g = torch.Generator(); g.manual_seed(seed)
    T = 48   # match real data sequence length

    def _make(n, s):
        g_ = torch.Generator(); g_.manual_seed(s)
        xa   = torch.randn(n, T, 7,  generator=g_)
        xb   = torch.randn(n, T, 4,  generator=g_)
        xc   = torch.randn(n, T, 3,  generator=g_)
        mask = torch.ones(n, T)                          # all timesteps valid
        y    = torch.randint(0, 2, (n,), generator=g_).float()
        ds   = TensorDataset(xa, xb, xc, mask, y)

        class _Loader:
            def __iter__(self_):
                for xa_b, xb_b, xc_b, m_b, y_b in DataLoader(
                        ds, batch_size=batch_size, shuffle=(s == seed)):
                    yield {"x_A": xa_b, "x_B": xb_b, "x_C": xc_b,
                           "mask": m_b, "y_ihm": y_b}
        return _Loader()

    return {"train": _make(256, seed), "val": _make(64, seed + 1)}


# ---------------------------------------------------------------------------
# Real data loaders — three per-site DataLoaders iterated in lockstep
# ---------------------------------------------------------------------------

def _real_loaders(root: str, batch_size: int, num_workers: int) -> dict:
    """
    Returns {"train": (loader_A, loader_B, loader_C),
             "val":   (loader_A, loader_B, loader_C)}
    Each per-site loader yields (x, mask, y) tuples.
    Site A y = y_ihm (binary); Sites B and C y ignored for this baseline.
    """
    root_p     = Path(root)
    splits_dir = root_p / "data" / "vertical_splits"
    bench_dir  = root_p / "data" / "mimic3-benchmarks" / "data"
    aligned    = splits_dir / "aligned_patient_ids.csv"

    from data_prep.dataset import (
        VFLSiteDataset,
        SITE_A_FEATURES, SITE_B_FEATURES, SITE_C_FEATURES, PHENO_LABEL_COLS,
    )

    loaders = {}
    for split in ("train", "val"):
        site_loaders = []
        for site, feat, label, ts_sub, ttype in [
            ("A", SITE_A_FEATURES, "y_ihm",        "in-hospital-mortality", "binary"),
            ("B", SITE_B_FEATURES, "y_los",         "length-of-stay",        "los_bins"),
            ("C", SITE_C_FEATURES, PHENO_LABEL_COLS,"phenotyping",           "multilabel"),
        ]:
            ds = VFLSiteDataset(
                site_csv        = splits_dir / f"site_{site}_{'vitals' if site=='A' else 'labs' if site=='B' else 'composite'}.csv",
                feature_cols    = feat,
                label_col       = label,
                split           = split,
                aligned_ids_csv = aligned,
                timeseries_root = bench_dir / ts_sub,
                task_type       = ttype,
            )
            site_loaders.append(DataLoader(
                ds, batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_fn,
                num_workers=num_workers,
            ))
        loaders[split] = tuple(site_loaders)
    return loaders


# ---------------------------------------------------------------------------
# Training loop — implements MARS-VFL Base.py protocol
# ---------------------------------------------------------------------------

def train_mars_vfl(
    root:          str,
    n_rounds:      int,
    lr:            float,
    batch_size:    int,
    seed:          int,
    use_synthetic: bool,
    num_workers:   int = 0,
) -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Local models — one VFLClient per site (same architecture as VFL-MTL)
    clients = {
        s: VFLClient(input_dim=_SITE_INPUT_DIMS[s], lr=lr, device=device)
        for s in _SITE_NAMES
    }

    # Global model — single IHM head (no MMoE)
    server     = _MARSVFLServer().to(device)
    server_opt = torch.optim.Adam(server.parameters(), lr=lr)
    loss_fn    = nn.BCELoss()

    loaders = (
        _synthetic_loaders(batch_size, seed)
        if use_synthetic
        else _real_loaders(root, batch_size, num_workers)
    )

    rows = []
    for rnd in range(1, n_rounds + 1):
        server.train()
        t0 = time.perf_counter()
        total_loss, nb = 0.0, 0

        # ── Training ──────────────────────────────────────────────────────
        if use_synthetic:
            batch_iter = loaders["train"]
        else:
            loader_A, loader_B, loader_C = loaders["train"]
            batch_iter = zip(loader_A, loader_B, loader_C)

        for raw_batch in batch_iter:
            if use_synthetic:
                # Synthetic: one dict with all sites
                batch = raw_batch
                inputs = {
                    "A": (batch["x_A"].to(device), batch["mask"].to(device)),
                    "B": (batch["x_B"].to(device), batch["mask"].to(device)),
                    "C": (batch["x_C"].to(device), batch["mask"].to(device)),
                }
                y_ihm = batch["y_ihm"].to(device)
            else:
                # Real: three (x, mask, y) tuples from lockstep loaders
                (x_A, mask_A, y_ihm), (x_B, mask_B, _), (x_C, mask_C, _) = raw_batch
                inputs = {
                    "A": (x_A.to(device), mask_A.to(device)),
                    "B": (x_B.to(device), mask_B.to(device)),
                    "C": (x_C.to(device), mask_C.to(device)),
                }
                y_ihm = y_ihm.to(device)

            # Step 1: each client encodes → returns detached leaf embedding
            # VFLClient.forward() stores original for backward via receive_gradient()
            embs = {s: clients[s].forward(*inputs[s]) for s in _SITE_NAMES}

            # Step 2: server concatenates embeddings
            # retain_grad() needed because concat is a non-leaf node
            concat = torch.cat([embs[s] for s in _SITE_NAMES], dim=-1)   # (B, 192)
            concat.retain_grad()

            # Step 3: server forward + loss
            pred = server(concat)
            loss = loss_fn(pred, y_ihm)
            server_opt.zero_grad()
            loss.backward()
            server_opt.step()

            # Step 4: slice concat.grad and return per-site gradients to clients
            # concat.grad is set because of retain_grad(); shape (B, 192)
            if concat.grad is not None:
                grad_slices = concat.grad.split(_EMBED_DIM, dim=-1)   # three (B, 64)
                for s, grad in zip(_SITE_NAMES, grad_slices):
                    clients[s].receive_gradient(grad)

            total_loss += loss.item(); nb += 1

        # ── Validation ────────────────────────────────────────────────────
        server.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            if use_synthetic:
                val_iter = loaders["val"]
            else:
                loader_A, loader_B, loader_C = loaders["val"]
                val_iter = zip(loader_A, loader_B, loader_C)

            for raw_batch in val_iter:
                if use_synthetic:
                    b = raw_batch
                    val_inputs = {
                        "A": (b["x_A"].to(device), b["mask"].to(device)),
                        "B": (b["x_B"].to(device), b["mask"].to(device)),
                        "C": (b["x_C"].to(device), b["mask"].to(device)),
                    }
                    y_val = b["y_ihm"]
                else:
                    (x_A, mask_A, y_val), (x_B, mask_B, _), (x_C, mask_C, _) = raw_batch
                    val_inputs = {
                        "A": (x_A.to(device), mask_A.to(device)),
                        "B": (x_B.to(device), mask_B.to(device)),
                        "C": (x_C.to(device), mask_C.to(device)),
                    }

                # eval_forward: no gradient tracking, encoder in eval mode
                val_embs = {s: clients[s].eval_forward(*val_inputs[s]) for s in _SITE_NAMES}
                val_concat = torch.cat([val_embs[s] for s in _SITE_NAMES], dim=-1)
                val_pred   = server(val_concat)
                all_preds.append(val_pred.cpu())
                all_labels.append(y_val)

        preds_np  = torch.cat(all_preds).numpy()
        labels_np = torch.cat(all_labels).numpy()
        metrics   = ihm_metrics(labels_np, preds_np)

        rows.append({
            "model":      "mars_vfl",
            "shared_task": "ihm",
            "round":      rnd,
            "train_loss": total_loss / max(nb, 1),
            "elapsed_s":  time.perf_counter() - t0,
            "seed":       seed,
            **{f"val_{k}": v for k, v in metrics.items()},
        })
        print(f"Round {rnd:3d}/{n_rounds} | "
              f"loss={rows[-1]['train_loss']:.4f} | "
              f"val_auc_roc={rows[-1].get('val_auc_roc', float('nan')):.4f}")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default=".")
    p.add_argument("--n_rounds",      type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--output",        default="results/mars_vfl.csv")
    p.add_argument("--use_synthetic", action="store_true")
    args = p.parse_args()

    all_rows = []
    for seed in args.seeds:
        rows = train_mars_vfl(args.root, args.n_rounds, args.lr,
                              args.batch_size, seed, args.use_synthetic,
                              args.num_workers)
        all_rows.extend(rows)
        print(f"seed={seed}: done ({len(rows)} rows)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_mars_vfl.py`**

```python
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.mars_vfl import train_mars_vfl


def test_returns_one_row_per_round():
    rows = train_mars_vfl(root=".", n_rounds=2, lr=1e-3,
                          batch_size=32, seed=42, use_synthetic=True)
    assert len(rows) == 2


def test_row_fields():
    rows = train_mars_vfl(root=".", n_rounds=1, lr=1e-3,
                          batch_size=32, seed=42, use_synthetic=True)
    r = rows[0]
    assert r["model"] == "mars_vfl"
    assert r["shared_task"] == "ihm"
    assert r["round"] == 1
    assert r["train_loss"] > 0


def test_has_val_auc_roc():
    rows = train_mars_vfl(root=".", n_rounds=1, lr=1e-3,
                          batch_size=32, seed=42, use_synthetic=True)
    assert "val_auc_roc" in rows[0]
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_mars_vfl.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add baselines/mars_vfl.py tests/test_mars_vfl.py
git commit -m "feat: add MARS-VFL paradigm baseline (single-task VFL, IHM, MARS-VFL base() protocol)"
```

---

### Baseline orchestration: `experiments/run_all_baselines.py`

- [ ] **Step 1: Implement `experiments/run_all_baselines.py`**

```python
"""
experiments/run_all_baselines.py — Run all Paper 1 baselines sequentially.

Runs all 5 implemented baselines (local_only, centralized, vfl_singletask,
mocha, fmtljd) and produces a merged results/all_baselines.csv.

VFL-MTL (proposed method) is run via experiments/run_exp1.py.
After both complete, merge:
    python -c "import pandas as pd; pd.concat([
        pd.read_csv('results/all_baselines.csv'),
        pd.read_csv('results/exp1.csv'),
    ]).to_csv('results/full_comparison.csv', index=False)"

Usage:
    python experiments/run_all_baselines.py \
        --splits_dir data/vertical_splits \
        --n_rounds 100 --n_epochs 50 \
        --seeds 42 123 7 --output results/all_baselines.csv

    # Smoke test
    python experiments/run_all_baselines.py --use_synthetic --n_rounds 2 --n_epochs 2
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.local_only    import train_local
from baselines.centralized   import train_centralized
from baselines.vfl_singletask import _TASK_WEIGHTS
from baselines.mocha         import train_mocha
from baselines.fmtljd        import train_fmtljd
from baselines.mars_vfl      import train_mars_vfl
from train                   import TrainConfig, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",    default="data/vertical_splits")
    parser.add_argument("--n_rounds",      type=int,   default=100)
    parser.add_argument("--n_epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--mocha_lam",     type=float, default=0.1)
    parser.add_argument("--output",        default="results/all_baselines.csv")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    all_rows = []

    # 1. Local-only (three sites independently)
    print("=== local_only ===")
    for site in ["A", "B", "C"]:
        for seed in args.seeds:
            rows = train_local(site=site, splits_dir=args.splits_dir,
                               n_epochs=args.n_epochs, lr=args.lr,
                               batch_size=args.batch_size, seed=seed,
                               use_synthetic=args.use_synthetic)
            all_rows.extend(rows)
            print(f"  site={site} seed={seed}: done ({len(rows)} rows)")

    # 2. Centralized oracle
    print("=== centralized_oracle ===")
    for seed in args.seeds:
        rows = train_centralized(splits_dir=args.splits_dir, n_epochs=args.n_epochs,
                                 lr=args.lr, batch_size=args.batch_size,
                                 seed=seed, use_synthetic=args.use_synthetic)
        all_rows.extend(rows)
        print(f"  seed={seed}: done ({len(rows)} rows)")

    # 3. VFL-SingleTask (three task configs)
    print("=== vfl_singletask ===")
    for task, weights in _TASK_WEIGHTS.items():
        for seed in args.seeds:
            cfg = TrainConfig(
                splits_dir=args.splits_dir, n_rounds=args.n_rounds,
                lr=args.lr, batch_size=args.batch_size, seed=seed,
                use_synthetic=args.use_synthetic,
                task_weights=weights, aggregation="fedavg",
            )
            rows = run_training(cfg)
            for r in rows:
                r["model"] = f"vfl_st_{task}"; r["active_task"] = task
            all_rows.extend(rows)
            print(f"  task={task} seed={seed}: done ({len(rows)} rows)")

    # 4. MOCHA (HFL-MTL, horizontal patient split)
    print("=== mocha ===")
    for seed in args.seeds:
        rows = train_mocha(splits_dir=args.splits_dir, n_rounds=args.n_rounds,
                           lam=args.mocha_lam, lr=args.lr, batch_size=args.batch_size,
                           seed=seed, use_synthetic=args.use_synthetic)
        all_rows.extend(rows)
        print(f"  seed={seed}: done ({len(rows)} rows)")

    # 5. FMTLJD (HFL-MTL with JAD, horizontal patient split)
    print("=== fmtljd ===")
    for seed in args.seeds:
        rows = train_fmtljd(splits_dir=args.splits_dir, n_rounds=args.n_rounds,
                            lr=args.lr, batch_size=args.batch_size,
                            seed=seed, use_synthetic=args.use_synthetic)
        all_rows.extend(rows)
        print(f"  seed={seed}: done ({len(rows)} rows)")

    # 6. MARS-VFL (single-task VFL, IHM shared across all sites)
    print("=== mars_vfl ===")
    for seed in args.seeds:
        rows = train_mars_vfl(splits_dir=args.splits_dir, n_rounds=args.n_rounds,
                              lr=args.lr, batch_size=args.batch_size,
                              seed=seed, use_synthetic=args.use_synthetic)
        all_rows.extend(rows)
        print(f"  seed={seed}: done ({len(rows)} rows)")

    df = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nAll baselines complete. {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
python experiments/run_all_baselines.py --use_synthetic --n_rounds 2 --n_epochs 2
```

Expected: `All baselines complete. N rows → results/all_baselines.csv`

- [ ] **Step 3: Commit**

```bash
git add experiments/run_all_baselines.py
git commit -m "feat: add run_all_baselines.py orchestration script"
```

---

### Snellius run order

```bash
SPLITS=/home/asoare/vfl_mlt/data/vertical_splits/

# Run all baselines (sequential; ~3-4 hours total)
python experiments/run_all_baselines.py \
  --splits_dir $SPLITS --n_rounds 100 --n_epochs 50 \
  --seeds 42 123 7 --output results/all_baselines.csv

# Run VFL-MTL (proposed method) via run_exp1.py
python experiments/run_exp1.py \
  --splits_dir $SPLITS --n_rounds 100 --seeds 42 123 7

# Merge into full comparison table
python -c "
import pandas as pd
pd.concat([
    pd.read_csv('results/all_baselines.csv'),
    pd.read_csv('results/exp1.csv'),
]).to_csv('results/full_comparison.csv', index=False)
print('full_comparison.csv written.')
"
```

### Baseline file map

| File | Status | Ablation question answered |
|---|---|---|
| `baselines/local_only.py` | ✅ implemented | Lower bound: what without any FL or MTL? |
| `baselines/centralized.py` | ✅ implemented | Upper bound: privacy cost of VFL? |
| `baselines/vfl_singletask.py` | ✅ covered by run_exp1.py | MTL contribution: does MTL help in VFL? |
| `baselines/mocha.py` | ✗ not applicable | HFL method — incompatible setting (same features, different samples). Cite as related work only. |
| `baselines/fmtljd.py` | ✗ not applicable | HFL method, psychiatric dataset. No shared data — cite reported numbers as context only. |
| `baselines/mars_vfl.py` | ✗ not applicable | Single-task VFL. Cite MARS-VFL's reported MIMIC-III IHM AUC as single-task VFL reference. No code integration needed. |
| `experiments/run_baselines.py` | ✅ implemented | Runs local_only + centralized; aggregates with exp1 VFL-MTL results |

---

## Chunk 6: Architecture Ablations

Five ablations isolate the contribution of individual design choices in VFL-MTL.
All use the same seeds [42, 123, 7] as main experiments. Results go to `results/ablations.csv`.

**Note on ST-VFL:** ST-IHM/ST-Decomp/ST-Pheno from Exp 1 serve as the MTL ablation and are
not re-run here. They are already in `results/exp1.csv`.

**Note on "No cut layer":** Dropped — transmitting raw features violates VFL's privacy guarantee
and is better described as a theoretical upper bound, not a realistic ablation. MARS-VFL reported
MIMIC-III IHM numbers serve this reference role instead.

---

### Ablation 1 — No MMoE: shared bottom MLP instead

**What it ablates:** Whether the MMoE gating is necessary, or a simpler shared-bottom MLP suffices.

**Implementation:** Add `use_mmoe: bool` flag to `MMoEServer`. When `False`, replace the 4-expert MMoE
with a single shared MLP (64→128→64) feeding all task heads — no gating, no per-task expert weighting.

```python
# In MMoEServer.__init__, add:
if not use_mmoe:
    self.shared_bottom = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 64))
# In forward(), replace expert gating with:
if not self.use_mmoe:
    shared = self.shared_bottom(x)
    return {task: head(shared) for task, head in self.task_heads.items()}
```

**Expected output:** `results/ablations.csv` rows with `model=abl_no_mmoe`.

---

### Ablation 2 — No PSI Alignment: random patient pairing

**What it ablates:** The value of correct privacy-preserving patient alignment.

**Implementation:** In `data_prep/dataset.py`, add `random_alignment` mode that shuffles patient IDs
across sites before joining. Patient counts per site stay the same; only cross-site pairing is randomised.

```python
def random_alignment(site_ids: dict[str, list], seed: int = 42) -> list:
    rng = random.Random(seed)
    min_len = min(len(ids) for ids in site_ids.values())
    shuffled = {site: rng.sample(ids, min_len) for site, ids in site_ids.items()}
    return list(zip(*shuffled.values()))
```

**Expected output:** `results/ablations.csv` rows with `model=abl_no_psi`. Performance drop vs.
VFL-MTL quantifies alignment value.

---

### Ablation 3 — MMoE expert count sensitivity: num_experts ∈ {2, 4, 8}

**What it ablates:** Sensitivity of VFL-MTL to the number of shared experts. `num_experts=4` is
the default (Ma et al. 2018 recommendation for 3-task settings); reviewers familiar with MMoE will
expect this sweep.

**Implementation:** `num_experts` is already a `TrainConfig` field and `MMoEServer` parameter.
No code changes needed — sweep via config.

**Expected output:** `results/ablations.csv` rows with `model=abl_experts_2`, `abl_experts_4`,
`abl_experts_8`.

---

### Ablation 4 — MMoE gating ablation: uniform fixed weights vs. learned softmax gating

**What it ablates:** Whether the learned gating network contributes, or uniform expert averaging suffices.

**Implementation:** Add `uniform_gating: bool` flag to `MMoEServer`. When `True`, replace softmax
gating with fixed equal weights (1/num_experts per expert).

```python
# In MMoEServer.forward(), replace gating with:
if self.uniform_gating:
    gate = torch.ones(batch, self.num_experts, device=x.device) / self.num_experts
else:
    gate = torch.softmax(self.gating[task](x), dim=-1)
```

**Expected output:** `results/ablations.csv` rows with `model=abl_uniform_gating`.

---

### Ablation 5 — Embedding dimension sensitivity: embed_dim ∈ {32, 64, 128}

**What it ablates:** Sensitivity to embedding size. `embed_dim` also proxies communication cost —
each round each site transmits a vector of this size to the server. Smaller = more private and
cheaper; larger = more expressive. Default is 64.

**Implementation:** `embed_dim` is already a `TrainConfig` field. No code changes needed.

**Expected output:** `results/ablations.csv` rows with `model=abl_embed_32`, `abl_embed_64`,
`abl_embed_128`.

---

### Single orchestration script: `experiments/run_ablations.py`

All five ablations run from one script to share data loading and avoid repeated GPFS reads.

```bash
# Smoke test:
python experiments/run_ablations.py --use_synthetic --n_rounds 3

# Snellius:
python experiments/run_ablations.py \
    --splits_dir /home/asoare/vfl_mlt/data/vertical_splits \
    --n_rounds 50 --device gpu_h100
```

Output: `results/ablations.csv` with columns:
`model, seed, round, val_ihm_auroc, val_ihm_auprc, val_decomp_auroc, val_decomp_auprc, val_pheno_macro_auroc`

---

### Ablation file map

| File | Status | Question answered |
|---|---|---|
| `experiments/run_ablations.py` | - [ ] | Orchestrates all 5 ablations |
| `model/mmoe.py` (flag: `use_mmoe`) | - [ ] | Is MMoE gating necessary? |
| `model/mmoe.py` (flag: `uniform_gating`) | - [ ] | Does learned gating add over uniform? |
| `data_prep/dataset.py` (flag: `random_align`) | - [ ] | What is the cost of proper PSI alignment? |
| `results/ablations.csv` | - [ ] | Aggregated ablation results |

---

## Final Checklist

Before claiming Paper 1 implementation is complete:

- [x] All unit tests pass: `python -m pytest tests/ -v`
- [x] Smoke-test synthetic training loop: `python train.py --use_synthetic --n_rounds 5`
- [x] All four experiment scripts importable: `python -c "import experiments.run_exp1"`
- [x] All figure scripts runnable on dummy CSV: `python figures/scalability_curves.py`
- [x] `git log --oneline` shows one commit per task
- [x] `requirements.txt` up to date
- [ ] Ablation script smoke test: `python experiments/run_ablations.py --use_synthetic --n_rounds 2`
- [ ] `results/ablations.csv` contains rows for: `abl_no_mmoe`, `abl_no_psi`, `abl_experts_2`,
      `abl_experts_4`, `abl_experts_8`, `abl_uniform_gating`, `abl_embed_32`, `abl_embed_64`, `abl_embed_128`
- [ ] All ablation Snellius runs complete
- [ ] Ablation bar chart figure generated: `figures/plot_ablations.py`

---

*Plan written: 2026-03-12. Spec source: CLAUDE.md — VFL-MTL Paper 1, Weeks 2–4.*
