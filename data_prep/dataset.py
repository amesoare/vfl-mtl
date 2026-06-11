"""
data_prep/dataset.py — VFL-MTL PyTorch Dataset for MIMIC-III and eICU vertical splits.

MIMIC-III (timeseries mode): Site CSVs index per-stay timeseries from YerevaNN task dirs.
eICU (tabular mode, timeseries_root=None): features pre-aggregated per patient; returns
  single-timestep (B, 1, F) sequences so the LSTM encoder is reused unchanged.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Feature and label column definitions
# (mirrors vertical_split.py — keep in sync if the split protocol changes)


SITE_A_FEATURES = [
    "Heart Rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Temperature",
    "Oxygen saturation",
    "Respiratory rate",
    "Glascow coma scale total",       # typo preserved from YerevaNN source
]

SITE_B_FEATURES = [
    "Glucose",
    "pH",
    "Fraction inspired oxygen",
    "Capillary refill rate",
]

SITE_C_FEATURES = [
    "Height",
    "Weight",
    "Mean blood pressure",
]

PHENO_LABEL_COLS = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]

assert len(PHENO_LABEL_COLS) == 25


# eICU feature and label column definitions
# (mirrors eicu_vertical_split.py — keep in sync if the split protocol changes)


EICU_SITE_A_FEATURES = [
    "Heart Rate", "Invasive BP Systolic", "Invasive BP Diastolic",
    "MAP (mmHg)", "O2 Saturation", "Respiratory Rate", "Temperature (C)",
]

EICU_SITE_B_FEATURES = ["glucose", "pH", "FiO2"]

EICU_SITE_C_FEATURES = ["GCS Total", "admissionheight", "admissionweight"]

EICU_PHENO_LABEL_COLS = [
    "Respiratory failure", "Essential hypertension", "Cardiac dysrhythmias",
    "Fluid disorders", "Septicemia", "Acute and unspecified renal failure",
    "Pneumonia", "Acute cerebrovascular disease", "CHF", "CKD", "COPD",
    "Acute myocardial infarction", "Gastrointestinal hem", "Shock",
    "lipid disorder", "DM with complications", "Coronary athe", "Pleurisy",
    "Other liver diseases", "lower respiratory", "Hypertension with complications",
    "Conduction disorders", "Complications of surgical", "upper respiratory",
    "DM without complication",
]

assert len(EICU_PHENO_LABEL_COLS) == 25



# Dataset


class VFLSiteDataset(Dataset):
    """PyTorch Dataset for one hospital site in the VFL-MTL setup."""

    def __init__(
        self,
        site_csv: Union[str, Path],
        feature_cols: list,
        label_col: Union[str, list],
        split: str,
        aligned_ids_csv: Union[str, Path],
        timeseries_root: Union[str, Path, None],
        max_seq_len: int = 48,
        task_type: str = "binary",
        id_col: str = "subject_id",
    ):
        assert split in ("train", "val", "test"), f"Unknown split: '{split}'"
        assert task_type in ("binary", "regression", "multilabel"), \
            f"Unknown task_type: '{task_type}'"

        self.feature_cols    = list(feature_cols)
        self.label_col       = label_col
        self.split           = split
        self.timeseries_root = Path(timeseries_root) if timeseries_root is not None else None
        self.max_seq_len     = max_seq_len
        self.task_type       = task_type

        aligned_df  = pd.read_csv(aligned_ids_csv)
        aligned_ids = set(aligned_df.loc[aligned_df["split"] == split, id_col])

        site_df = pd.read_csv(site_csv)
        site_df = site_df[
            (site_df["split"] == split) &
            (site_df[id_col].isin(aligned_ids))
        ].reset_index(drop=True)

        # Tabular mode (eICU): features already aggregated in the site CSV.
        # Returns (1, F) sequences so the LSTM encoder is reused unchanged.
        if self.timeseries_root is None:
            self._tabular    = True
            self._x_arr      = site_df[self.feature_cols].values.astype(np.float32)
        else:
            self._tabular    = False
            self.stays       = site_df["stay"].tolist()
            subdir = "test" if split == "test" else "train"
            self._cache: dict[str, tuple] = {}
            for stay in self.stays:
                if stay not in self._cache:
                    self._cache[stay] = self._load_timeseries_from_disk(
                        self.timeseries_root / subdir / stay
                    )

        if task_type == "multilabel":
            self.labels = site_df[label_col].values.astype(np.float32)
        else:  # binary or regression
            self.labels = site_df[label_col].values.astype(np.float32)


    def __len__(self) -> int:
        if self._tabular:
            return len(self._x_arr)
        return len(self.stays)

    def __getitem__(self, idx: int) -> tuple:
        if self._tabular:
            x    = torch.from_numpy(self._x_arr[idx : idx + 1])  # (1, F)
            mask = torch.ones(1, dtype=torch.float32)
        else:
            x_np, mask_np = self._load_timeseries(self.stays[idx])
            x    = torch.from_numpy(x_np)
            mask = torch.from_numpy(mask_np)

        raw = self.labels[idx]
        if self.task_type == "multilabel":
            y = torch.from_numpy(raw.copy())
        else:  # binary or regression
            y = torch.tensor(float(raw), dtype=torch.float32)

        return x, mask, y


    def _load_timeseries(self, stay_filename: str) -> tuple:
        return self._cache[stay_filename]

    def _load_timeseries_from_disk(self, path) -> tuple:
        """Load one timeseries CSV → hourly-binned, ffill/bfill imputed, padded to max_seq_len."""
        df = pd.read_csv(path)

        # Step 2 — bin fractional hours to integers
        df["_bin"] = (
            df["Hours"]
            .astype(float)
            .apply(lambda t: min(int(t), self.max_seq_len - 1))
        )

        # Record actual stay length before reindexing
        max_observed_bin = int(df["_bin"].max()) if len(df) > 0 else 0
        actual_len       = min(max_observed_bin + 1, self.max_seq_len)

        # Step 3 — last non-NaN value per bin per feature
        available = [c for c in self.feature_cols if c in df.columns]
        binned    = df.groupby("_bin")[available].last()

        # Step 4 — reindex to full range
        binned = binned.reindex(range(self.max_seq_len))

        # Add columns for any feature absent from this file
        for col in self.feature_cols:
            if col not in binned.columns:
                binned[col] = np.nan
        binned = binned[self.feature_cols]   # enforce column order

        # Steps 5–6 — impute
        binned = binned.ffill().bfill().fillna(0.0)

        x    = binned.values.astype(np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        return x, mask



# Collate function


def collate_fn(batch: list) -> tuple:
    xs, masks, ys = zip(*batch)
    return (
        torch.stack(xs),     # (B, T, F)
        torch.stack(masks),  # (B, T)
        torch.stack(ys),     # (B,) or (B, 25)
    )



# Convenience builder — constructs all three site loaders at once


def build_site_loaders(
    root: Union[str, Path],
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_seq_len: int = 48,
    dataset: str = "mimic",
) -> dict:
    """Build DataLoaders for all three sites for a given split ('train'/'val'/'test')."""
    root = Path(root)

    if dataset == "eicu":
        splits_dir = root / "data" / "eicu_vertical_splits"
        aligned    = splits_dir / "aligned_patient_ids_eicu.csv"
        configs = {
            "A": dict(
                site_csv        = splits_dir / "site_A_eicu.csv",
                feature_cols    = EICU_SITE_A_FEATURES,
                label_col       = "y_ihm",
                timeseries_root = None,
                task_type       = "binary",
                id_col          = "patientunitstayid",
            ),
            "B": dict(
                site_csv        = splits_dir / "site_B_eicu.csv",
                feature_cols    = EICU_SITE_B_FEATURES,
                label_col       = "y_rlos",
                timeseries_root = None,
                task_type       = "regression",
                id_col          = "patientunitstayid",
            ),
            "C": dict(
                site_csv        = splits_dir / "site_C_eicu.csv",
                feature_cols    = EICU_SITE_C_FEATURES,
                label_col       = EICU_PHENO_LABEL_COLS,
                timeseries_root = None,
                task_type       = "multilabel",
                id_col          = "patientunitstayid",
            ),
        }
    else:  # mimic
        splits_dir = root / "data" / "vertical_splits"
        bench_dir  = root / "data" / "mimic3-benchmarks" / "data"
        aligned    = splits_dir / "aligned_patient_ids.csv"
        configs = {
            "A": dict(
                site_csv        = splits_dir / "site_A_vitals.csv",
                feature_cols    = SITE_A_FEATURES,
                label_col       = "y_ihm",
                timeseries_root = bench_dir / "in-hospital-mortality",
                task_type       = "binary",
                id_col          = "subject_id",
            ),
            "B": dict(
                site_csv        = splits_dir / "site_B_labs.csv",
                feature_cols    = SITE_B_FEATURES,
                label_col       = "y_decomp",
                timeseries_root = bench_dir / "decompensation",
                task_type       = "binary",
                id_col          = "subject_id",
            ),
            "C": dict(
                site_csv        = splits_dir / "site_C_composite.csv",
                feature_cols    = SITE_C_FEATURES,
                label_col       = PHENO_LABEL_COLS,
                timeseries_root = bench_dir / "phenotyping",
                task_type       = "multilabel",
                id_col          = "subject_id",
            ),
        }

    loaders = {}
    for site_id, cfg in configs.items():
        ds = VFLSiteDataset(
            site_csv        = cfg["site_csv"],
            feature_cols    = cfg["feature_cols"],
            label_col       = cfg["label_col"],
            split           = split,
            aligned_ids_csv = aligned,
            timeseries_root = cfg["timeseries_root"],
            max_seq_len     = max_seq_len,
            task_type       = cfg["task_type"],
            id_col          = cfg["id_col"],
        )
        loaders[site_id] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            collate_fn  = collate_fn,
            num_workers = num_workers,
            drop_last   = True,
        )

    return loaders
