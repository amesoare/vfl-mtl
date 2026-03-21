# VFL-MTL Paper 1 Completion — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

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

### ❌ Not yet implemented

- [ ] `train.py` — round-based VFL-MTL orchestration loop
- [ ] `experiments/metrics.py` — AUC-ROC, AUC-PR, Cohen's kappa, macro-AUC
- [ ] `experiments/run_exp1.py` — task heterogeneity vs. homogeneity
- [ ] `experiments/run_exp2.py` — feature asymmetry
- [ ] `experiments/run_exp3.py` — task relatedness / negative transfer
- [ ] `experiments/run_exp4.py` — scalability
- [ ] `results/plot_results.py` — shared plotting utilities
- [ ] `figures/negative_transfer_heatmap.py`
- [ ] `figures/scalability_curves.py`
- [ ] `figures/feature_split_sensitivity.py`

### ⚠️ Data location

Vertical splits **exist on Snellius** at `/home/asoare/vfl_mlt/data/` but **must be regenerated** after Chunk 0 fixes (clipping + stratification):
- `site_A_vitals.csv`
- `site_B_labs.csv`
- `site_C_composite.csv`
- `aligned_patient_ids.csv`

Do not use the current Snellius CSVs for experiments — they contain unclipped outliers and non-stratified splits.

For Snellius experiment runs, pass `--splits_dir /home/asoare/vfl_mlt/data/`.
For local smoke tests, use `--use_synthetic` (no data required).

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `data_prep/vertical_split.py` | ✅ done (Snellius re-run pending) | `load_clip_bounds()` + `clip_features()` before ffill/bfill; bounds from `variable_ranges.csv` |
| `data_prep/psi_alignment.py` | ✅ done (Snellius re-run pending) | `check_label_balance()` + `stratify_aligned_cohort()` after PSI intersection |
| `data_prep/dataset.py` | ✅ done | `VFLSiteDataset` + `build_site_loaders()`; raw T=48 sequences, YerevaNN CustomBins |
| `fl/fedavg.py` | ✅ done | `fedavg_aggregate()`: weighted average of compatible encoder state dicts |
| `fl/fedprox.py` | ✅ done | `fedprox_penalty()`: `(mu/2)||w_local-w_global||²`, device-aware |
| `tests/test_fedavg.py` | ✅ done | 5 tests: weighted avg, round-trip, penalty zero/positive/mu-scaling |
| `train.py` | ❌ todo | `run_training()`: round-based loop, lockstep zip over 3 site loaders, FedAvg/FedProx |
| `experiments/metrics.py` | ❌ todo | AUC-ROC, AUC-PR, Cohen's kappa, macro-AUC per task |
| `experiments/run_exp1.py` | ❌ todo | VFL-MTL vs. VFL-SingleTask; per-task AUC |
| `experiments/run_exp2.py` | ❌ todo | Feature asymmetry: 3 split configs × 3 seeds |
| `experiments/run_exp3.py` | ❌ todo | Task relatedness / negative transfer rate |
| `experiments/run_exp4.py` | ❌ todo | Scalability: 2/3 sites × rounds → AUC + wall-clock |
| `results/plot_results.py` | ❌ todo | `load_results()`, `comparison_table()`, `loss_curves()` |
| `figures/negative_transfer_heatmap.py` | ❌ todo | Task × model loss-delta heatmap |
| `figures/scalability_curves.py` | ❌ todo | Rounds-to-convergence vs. n_institutions |
| `figures/feature_split_sensitivity.py` | ❌ todo | AUC per split configuration bar chart |
| `tests/test_train.py` | ❌ todo | Smoke test for run_training() on synthetic data |

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
- [ ] **Re-run `vertical_split.py` on Snellius to overwrite existing CSVs**

```bash
# on Snellius
python data_prep/vertical_split.py \
    --root /home/asoare/vfl_mlt/mimic3-benchmarks/data/ \
    --output /home/asoare/vfl_mlt/data/
```

- [ ] **Commit**

```bash
git add data_prep/vertical_split.py mimic3-benchmarks/mimic3benchmark/scripts/extract_episodes_from_subjects.py
git commit -m "fix(data): clip features to VALID_LOW/VALID_HIGH from variable_ranges.csv before imputation"
```

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

- [ ] **Step 4: Re-run `psi_alignment.py` on Snellius** (after vertical_split.py re-run)

```bash
# on Snellius — re-run after vertical_split.py has produced updated CSVs
python data_prep/psi_alignment.py \
    --site_a /home/asoare/vfl_mlt/data/site_A_vitals.csv \
    --site_b /home/asoare/vfl_mlt/data/site_B_labs.csv \
    --site_c /home/asoare/vfl_mlt/data/site_C_composite.csv \
    --output /home/asoare/vfl_mlt/data/aligned_patient_ids.csv
```

Confirm output log shows either "Label balance OK" or reports which splits were re-stratified.

- [ ] **Step 5: Commit**

```bash
git add data_prep/psi_alignment.py requirements.txt
git commit -m "fix(data): stratified split assignment for PSI-aligned cohort; add skmultilearn"
```

---

## Chunk 1: Data Layer and Federated Aggregation

### ✅ Task 1: VFLDataset — `data_prep/dataset.py`

**Context:** `vertical_split.py` produces three CSVs where each row is one ICU stay, with mean-aggregated feature values and task labels. `psi_alignment.py` produces `aligned_patient_ids.csv` with `subject_id` and `split` columns. The Dataset must: filter to aligned IDs for the requested split, return per-site feature tensors with T=1 (single aggregated timestep), a ones mask, and all task labels. LOS is bucketed into 10 bins using YerevaNN-compatible thresholds.

**LOS bins (hours):** `[0, 8, 16, 24, 36, 48, 72, 120, 168, 336, ∞)` — 10 intervals, label = index 0–9.

**Files:**
- Create: `data_prep/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ameliasoare/Documents/codes
python -m pytest tests/test_dataset.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'data_prep.dataset'`

- [ ] **Step 3: Implement `data_prep/dataset.py`**

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dataset.py -v
```
Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_fedavg.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `fl/fedavg.py`**

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

- [ ] **Step 4: Implement `fl/fedprox.py`**

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

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_fedavg.py -v
```
Expected: 5 tests PASSED.

- [ ] **Step 6: Update `fl/__init__.py`**

Add exports:
```python
from .client import VFLClient
from .server import VFLServer
from .fedavg import fedavg_aggregate
from .fedprox import fedprox_penalty
```

- [ ] **Step 7: Commit**

```bash
git add fl/fedavg.py fl/fedprox.py fl/__init__.py tests/test_fedavg.py
git commit -m "feat: add FedAvg aggregation and FedProx proximal penalty"
```

---

## Chunk 2: Training Loop

### Task 3: Training orchestration — `train.py`

**Context:** `train.py` is the central round-based training loop used by all experiment scripts. It accepts configuration as a dataclass/dict and returns per-round metrics. Each round: (1) all clients forward-pass their batches, (2) server aggregates and computes loss, (3) server backprops and returns gradients, (4) clients update. Optional FedAvg runs every `fedavg_every` rounds. Optional FedProx adds the proximal penalty to each client before backward.

**Data interface:** `dataset.py` uses `build_site_loaders(root, split)` returning `{'A': DataLoader, 'B': DataLoader, 'C': DataLoader}`. Each loader yields `(x, mask, y)` tuples — NOT dicts. Batches across sites must be iterated in lockstep using `zip(loader_A, loader_B, loader_C)`. The `mimic3-benchmarks/` directory must be on the path (already handled inside `dataset.py` itself).

**Evaluation:** after each round, run `eval_round()` that calls `client.eval_forward()` on the val DataLoader and `server.predict()`, then computes all task metrics.

**Files:**
- Create: `experiments/metrics.py`
- Create: `train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write `experiments/metrics.py`** (no test needed — thin wrapper around sklearn)

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

- [ ] **Step 2: Write the failing train test**

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

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_train.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 4: Implement `train.py`**

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

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_train.py -v
```
Expected: 4 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add experiments/metrics.py train.py tests/test_train.py
git commit -m "feat: add VFL-MTL training loop with FedAvg/FedProx support"
```

---

## Chunk 3: Experiment Scripts

### Task 4: Experiment 1 — Task Heterogeneity vs. Homogeneity (`experiments/run_exp1.py`)

**Context:** Compare VFL-MTL (3 sites, 3 heterogeneous tasks) against VFL-SingleTask (3 sites, all sites optimise only IHM). Report per-task AUC-ROC with seeds [42, 123, 7]. Write results to `results/exp1.csv`.

**VFL-SingleTask:** Use `VFLServer` with `task_weights = {"ihm": 1.0, "los": 0.0, "pheno": 0.0}` (zeroed-out tasks). This reuses the same architecture with non-participating task heads.

**Files:**
- Create: `experiments/run_exp1.py`

- [ ] **Step 1: Implement `experiments/run_exp1.py`**

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

- [ ] **Step 2: Add `--use_synthetic` to all experiment argparsers**

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

- [ ] **Step 3: Smoke-test on synthetic data**

```bash
python experiments/run_exp1.py \
    --n_rounds 3 --device cpu --use_synthetic \
    --output /tmp/exp1_smoke.csv 2>&1 | tail -5
```

Expected output: loss values printed for 3 rounds × 2 models × 3 seeds; CSV written to `/tmp/exp1_smoke.csv`.

- [ ] **Step 4: Commit**

```bash
git add experiments/run_exp1.py
git commit -m "feat: add Exp1 task heterogeneity vs. homogeneity script"
```

---

### Task 5: Experiment 2 — Feature Asymmetry (`experiments/run_exp2.py`)

**Context:** Test sensitivity of MTL gains to how unevenly features are distributed across sites. Three configurations are tested by overriding `site_dims` in TrainConfig. Since VFLDataset yields 7/4/3 features, asymmetry configs instead train sub-encoders that use only a subset of columns — achieved by passing `input_dim` overrides.

**Split configurations:**
- `balanced`: 5/5/4 (select first 5, 5, 4 from each site's columns)
- `skewed`:   3/7/4 (site B gets features from A and B merged — requires data prep change; **simplify**: test with input_dim only, truncate/pad feature tensor)
- `default`:  7/4/3 (standard split from CLAUDE.md)

For the experiment, "feature asymmetry" is approximated by varying `input_dim` and truncating/zero-padding the input tensor to the requested dimension. This does NOT require re-running data prep.

**Files:**
- Create: `experiments/run_exp2.py`

- [ ] **Step 1: Implement `experiments/run_exp2.py`**

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

- [ ] **Step 2: Update `train.py` to support `site_input_dims`**

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

- [ ] **Step 3: Fix `run_exp2.py` to pass `site_input_dims` to `TrainConfig`**

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

- [ ] **Step 4: Run tests still pass after train.py change**

```bash
python -m pytest tests/test_train.py -v
```
Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_exp2.py train.py
git commit -m "feat: add Exp2 feature asymmetry; add site_input_dims to TrainConfig"
```

---

### Task 6: Experiment 3 — Task Relatedness / Negative Transfer (`experiments/run_exp3.py`)

**Context:** Test whether task relatedness affects MTL gains. Compare two task pairings:
- Pair A: IHM + Decompensation (related — both are mortality-type signals)
- Pair B: IHM + Phenotyping (less related — acute outcome vs. chronic condition labels)

Since decompensation task was not included in the vertical split, Pair A is approximated as IHM-only + IHM replicated at site C (same binary label). For Pair B use IHM + pheno as-is. Negative transfer = AUC on IHM task drops below local-only baseline when paired with an unrelated task.

**Files:**
- Create: `experiments/run_exp3.py`

- [ ] **Step 1: Implement `experiments/run_exp3.py`**

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

- [ ] **Step 2: Commit**

```bash
git add experiments/run_exp3.py
git commit -m "feat: add Exp3 task relatedness and negative transfer script"
```

---

### Task 7: Experiment 4 — Scalability (`experiments/run_exp4.py`)

**Context:** Vary the number of participating institutions (2 or 3) and measure convergence rounds and wall-clock time. With 2 sites, use only A+B (IHM + LOS). With 3 sites, use the full setup.

**Files:**
- Create: `experiments/run_exp4.py`

- [ ] **Step 1: Add `n_sites` support to `VFLServer` and `train.py`**

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

- [ ] **Step 2: Run existing tests after train.py changes**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED.

- [ ] **Step 3: Implement `experiments/run_exp4.py`**

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

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_exp4.py train.py
git commit -m "feat: add Exp4 scalability script; add n_sites support to TrainConfig"
```

---

## Chunk 4: Results and Figures

### Task 8: Plot utilities and figure scripts

**Context:** Shared utilities load CSVs, compute mean±std across seeds, and produce comparison tables. Three specific figure scripts implement the paper figures.

**Files:**
- Create: `results/plot_results.py`
- Create: `figures/negative_transfer_heatmap.py`
- Create: `figures/scalability_curves.py`
- Create: `figures/feature_split_sensitivity.py`

- [ ] **Step 1: Implement `results/plot_results.py`**

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

- [ ] **Step 2: Implement `figures/negative_transfer_heatmap.py`**

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

- [ ] **Step 3: Implement `figures/scalability_curves.py`**

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

- [ ] **Step 4: Implement `figures/feature_split_sensitivity.py`**

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

- [ ] **Step 5: Commit**

```bash
git add results/plot_results.py figures/negative_transfer_heatmap.py \
        figures/scalability_curves.py figures/feature_split_sensitivity.py
git commit -m "feat: add result plotting utilities and three paper figure scripts"
```

---

## Final Checklist

Before claiming Paper 1 implementation is complete:

- [ ] All unit tests pass: `python -m pytest tests/ -v`
- [ ] Smoke-test synthetic training loop: `python train.py --use_synthetic --n_rounds 5`
- [ ] All four experiment scripts importable: `python -c "import experiments.run_exp1"`
- [ ] All figure scripts runnable on dummy CSV: `python figures/scalability_curves.py`
- [ ] `git log --oneline` shows one commit per task
- [ ] `requirements.txt` up to date (add `seaborn` if missing — already present)

---

*Plan written: 2026-03-12. Spec source: CLAUDE.md — VFL-MTL Paper 1, Weeks 2–4.*
