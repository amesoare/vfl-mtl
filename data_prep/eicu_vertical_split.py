#!/usr/bin/env python3
"""
data_prep/eicu_vertical_split.py

Splits eICU-CRD data (preprocessed by eICU_Benchmark) into three vertical
(feature) partitions representing three simulated hospital sites.

    Site A (7 vitals)    : HR, SBP, DBP, MAP, SpO2, RespRate, Temp
                           Task label: in-hospital mortality (binary)

    Site B (3 labs)      : Glucose, pH, FiO2 (normalised % → fraction)
                           Task label: remaining length of stay (regression, days)

    Site C (3 composite) : GCS Total, Height, Weight
                           Task label: phenotyping (25 binary ICD-9 codes)

13 features total. GCS subscores (Eyes, Motor, Verbal) excluded — same
cross-site reconstruction risk as MIMIC-III split (total = eye + motor + verbal).
FiO2 stored as % in eICU_Benchmark; divided by 100 to match MIMIC-III scale.

Prerequisites:
    Run eICU_Benchmark root extraction first:
      python data_prep/eicu_benchmark/data_extraction/data_extraction_root.py \\
          --eicu_dir /path/to/eicu-crd/2.0/ \\
          --output_dir /path/to/eicu_root_output/

Usage:
    python data_prep/eicu_vertical_split.py \\
        --root_dir /path/to/eicu_root_output/ \\
        --eicu_dir /path/to/eicu-crd/2.0/ \\
        --output   /path/to/eicu_vertical_splits/
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Feature definitions — eICU_Benchmark column names in all_data.csv
# ---------------------------------------------------------------------------

SITE_A_FEATURES = [
    "Heart Rate",
    "Invasive BP Systolic",
    "Invasive BP Diastolic",
    "MAP (mmHg)",
    "O2 Saturation",
    "Respiratory Rate",
    "Temperature (C)",
]

SITE_B_FEATURES = [
    "glucose",
    "pH",
    "FiO2",
]

SITE_C_FEATURES = [
    "GCS Total",
    "admissionheight",
    "admissionweight",
]

PHENO_LABELS = [
    "Respiratory failure", "Essential hypertension", "Cardiac dysrhythmias",
    "Fluid disorders", "Septicemia", "Acute and unspecified renal failure",
    "Pneumonia", "Acute cerebrovascular disease", "CHF", "CKD", "COPD",
    "Acute myocardial infarction", "Gastrointestinal hem", "Shock",
    "lipid disorder", "DM with complications", "Coronary athe", "Pleurisy",
    "Other liver diseases", "lower respiratory", "Hypertension with complications",
    "Conduction disorders", "Complications of surgical", "upper respiratory",
    "DM without complication",
]

# ---------------------------------------------------------------------------
# Clip bounds — PRISM tighter bounds (see CLAUDE.md CLIP_BOUNDS table)
# Applied after eICU_Benchmark extraction; FiO2 clipped before /100.
# ---------------------------------------------------------------------------

CLIP_BOUNDS = {
    "Heart Rate":            (0,    300),
    "Invasive BP Systolic":  (0,    300),
    "Invasive BP Diastolic": (0,    200),
    "Temperature (C)":       (25,    45),
    "O2 Saturation":         (0,    100),
    "Respiratory Rate":      (0,    100),
    "GCS Total":             (3,     15),
    "glucose":               (20,  2000),
    "pH":                    (6.5,  8.0),
    "FiO2":                  (21,   100),   # % — divided by 100 after clipping
    "MAP (mmHg)":            (0,    300),
    "admissionheight":       (50,   240),
    "admissionweight":       (10,   250),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_data(root_dir: Path) -> pd.DataFrame:
    path = root_dir / "eicu_all_data.csv"
    if not path.exists():
        print(f"Error: eicu_all_data.csv not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {path} ...")
    return pd.read_csv(path)


def prepare_cohort(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["gender"] != 0].copy()
    df = df[df["hospitaldischargestatus"] != 2].copy()
    df["RLOS"] = (df["unitdischargeoffset"] / 1440) - (df["itemoffset"] / 24)
    return df


def clip_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    if "FiO2" in df.columns:
        df["FiO2"] = df["FiO2"] / 100.0
    return df


def aggregate_per_patient(df: pd.DataFrame, feature_cols: list,
                           train_means: pd.Series = None) -> pd.DataFrame:
    rows = []
    for pid, group in df.groupby("patientunitstayid"):
        feat = group[feature_cols].ffill().bfill()
        row = {"patientunitstayid": pid}
        row.update(feat.mean().to_dict())
        rows.append(row)
    out = pd.DataFrame(rows)
    if train_means is not None:
        for col in feature_cols:
            out[col] = out[col].fillna(train_means.get(col, 0.0))
    return out


def make_splits(patient_ids: np.ndarray, labels: pd.Series,
                val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    ids = np.array(patient_ids)
    y = labels.reindex(ids).fillna(0).astype(int).values
    ids_tv, ids_test = train_test_split(ids, test_size=test_frac,
                                        stratify=y, random_state=seed)
    y_tv = labels.reindex(ids_tv).fillna(0).astype(int).values
    ids_train, ids_val = train_test_split(ids_tv,
                                          test_size=val_frac / (1 - test_frac),
                                          stratify=y_tv, random_state=seed)
    return set(ids_train), set(ids_val), set(ids_test)


# ---------------------------------------------------------------------------
# Site builders
# ---------------------------------------------------------------------------

def build_site_a(df: pd.DataFrame, train_ids, val_ids, test_ids, output: Path,
                 file_prefix: str = ""):
    print("Building Site A (vitals → IHM) ...")
    # IHM: restrict to first 48h window
    sub = df[df["itemoffset"] <= 48].copy()
    ihm_labels = sub.groupby("patientunitstayid")["hospitaldischargestatus"].max()

    train_means, frames = None, []
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        agg = aggregate_per_patient(
            sub[sub["patientunitstayid"].isin(split_ids)],
            SITE_A_FEATURES, train_means=train_means,
        )
        if split_name == "train":
            train_means = agg[SITE_A_FEATURES].mean()
            for col in SITE_A_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))
        agg["y_ihm"] = ihm_labels.reindex(agg["patientunitstayid"]).values
        agg["split"] = split_name
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    path = output / f"{file_prefix}site_A_eicu.csv"
    out.to_csv(path, index=False)
    print(f"  → {path}  ({len(out)} patients)")
    return out


def build_site_b(df: pd.DataFrame, train_ids, val_ids, test_ids, output: Path,
                 file_prefix: str = ""):
    print("Building Site B (labs → RLOS) ...")
    sub = df[(df["itemoffset"] > 0) & (df["RLOS"] > 0)].copy()
    rlos_labels = sub.groupby("patientunitstayid")["RLOS"].mean()

    train_means, frames = None, []
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        agg = aggregate_per_patient(
            sub[sub["patientunitstayid"].isin(split_ids)],
            SITE_B_FEATURES, train_means=train_means,
        )
        if split_name == "train":
            train_means = agg[SITE_B_FEATURES].mean()
            for col in SITE_B_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))
        agg["y_rlos"] = rlos_labels.reindex(agg["patientunitstayid"]).values
        agg["split"] = split_name
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    path = output / f"{file_prefix}site_B_eicu.csv"
    out.to_csv(path, index=False)
    print(f"  → {path}  ({len(out)} patients)")
    return out


def build_site_c(df: pd.DataFrame, eicu_dir: Path,
                 train_ids, val_ids, test_ids, output: Path,
                 file_prefix: str = ""):
    print("Building Site C (neuro+static → Phenotyping) ...")

    phen_code_path = Path(__file__).parent / "eicu_benchmark" / "phen_code.json"
    codes = json.loads(phen_code_path.read_text())

    diag = pd.read_csv(eicu_dir / "diagnosis.csv",
                       usecols=["patientunitstayid", "diagnosisoffset", "icd9code"])
    diag = diag[diag["diagnosisoffset"] > 0].dropna(subset=["icd9code"]).copy()
    diag["icd"] = diag["icd9code"].astype(str).str.split(",").str[0].str.replace(".", "", regex=False)

    label_code_map = [
        ("Septicemia",                        "septicemia"),
        ("Shock",                             "Shock"),
        ("Complications of surgical",         "Compl_surgical"),
        ("CKD",                               "ckd"),
        ("Acute and unspecified renal failure","renal_failure"),
        ("Gastrointestinal hem",              "Gastroint_hemorrhage"),
        ("Other liver diseases",              "Other_liver_dis"),
        ("upper respiratory",                 "upper_respiratory"),
        ("lower respiratory",                 "lower_respiratory"),
        ("Respiratory failure",               "Resp_failure"),
        ("Pleurisy",                          "Pleurisy"),
        ("COPD",                              "COPD"),
        ("Pneumonia",                         "Pneumonia"),
        ("Acute cerebrovascular disease",     "Acute_cerebrovascular"),
        ("CHF",                               "Congestive_hf"),
        ("Cardiac dysrhythmias",              "Cardiac_dysr"),
        ("Conduction disorders",              "Conduction_dis"),
        ("Coronary athe",                     "Coronary_ath"),
        ("Acute myocardial infarction",       "myocar_infarction"),
        ("Hypertension with complications",   "hypercomp"),
        ("Essential hypertension",            "essehyper"),
        ("Fluid disorders",                   "fluiddiso"),
        ("lipid disorder",                    "lipidmetab"),
        ("DM with complications",             "t2dmcomp"),
        ("DM without complication",           "t2dmwocomp"),
    ]
    for label, key in label_code_map:
        diag[label] = diag["icd"].isin(codes[key]).astype(float)

    pheno_labels = (
        diag.groupby("patientunitstayid")[PHENO_LABELS]
        .max().clip(0, 1).reset_index()
    )

    sub = df[df["itemoffset"] > 0].copy()
    valid_pids = set(pheno_labels["patientunitstayid"])

    train_means, frames = None, []
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_valid = split_ids & valid_pids
        agg = aggregate_per_patient(
            sub[sub["patientunitstayid"].isin(split_valid)],
            SITE_C_FEATURES, train_means=train_means,
        )
        if split_name == "train":
            train_means = agg[SITE_C_FEATURES].mean()
            for col in SITE_C_FEATURES:
                agg[col] = agg[col].fillna(train_means.get(col, 0.0))
        agg = agg.merge(pheno_labels, on="patientunitstayid", how="left")
        for lc in PHENO_LABELS:
            agg[lc] = agg[lc].fillna(0).astype(int)
        agg["split"] = split_name
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    path = output / f"{file_prefix}site_C_eicu.csv"
    out.to_csv(path, index=False)
    print(f"  → {path}  ({len(out)} patients)")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", required=True,
                        help="eICU_Benchmark output dir (contains all_data.csv)")
    parser.add_argument("--eicu_dir", required=True,
                        help="Raw eICU-CRD dir (contains diagnosis.csv)")
    parser.add_argument("--output", required=True,
                        help="Output directory for vertical split CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    eicu_dir = Path(args.eicu_dir).resolve()
    output   = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    df = load_all_data(root_dir)
    df = prepare_cohort(df)
    df = clip_and_normalize(df)

    ihm_labels = df.groupby("patientunitstayid")["hospitaldischargestatus"].max()
    train_ids, val_ids, test_ids = make_splits(
        ihm_labels.index.values, ihm_labels, seed=args.seed
    )
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients")

    build_site_a(df, train_ids, val_ids, test_ids, output)
    build_site_b(df, train_ids, val_ids, test_ids, output)
    build_site_c(df, eicu_dir, train_ids, val_ids, test_ids, output)

    print("\nVertical split complete.")
    print(f"Output: {output}")
    print("Next: run psi_alignment.py --dataset eicu to produce aligned_patient_ids.csv")


if __name__ == "__main__":
    main()
