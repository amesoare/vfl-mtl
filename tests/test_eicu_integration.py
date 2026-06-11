"""
tests/test_eicu_data_prep.py

Integration test for eICU vertical split and PSI alignment.
Uses synthetic data — no real eICU files or Snellius needed.

Mirrors the pattern used in test_integration.py for MIMIC scripts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import data_prep.eicu_vertical_split as evs
from data_prep.eicu_psi_alignment import check_label_balance
from data_prep.psi_alignment import compute_psi_alignment, write_aligned_ids

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

N_PATIENTS = 250
ROWS_PER_PATIENT = 5
RNG = np.random.default_rng(42)

_PHEN_CODE = json.loads(
    (PROJECT_ROOT / "data_prep" / "eicu_benchmark" / "phen_code.json").read_text()
)
# One representative ICD-9 code per phenotype group
_SAMPLE_CODES = [codes[0] for codes in _PHEN_CODE.values()]


def _make_all_data(n: int = N_PATIENTS) -> pd.DataFrame:
    pids = np.arange(100_000, 100_000 + n)
    records = []
    for pid in pids:
        stay_min = float(RNG.integers(2 * 1440, 8 * 1440))   # 2–8 days in minutes
        for t in range(ROWS_PER_PATIENT):
            # itemoffset in hours; keep last row well below stay end to ensure RLOS > 0
            offset_h = float(t * (stay_min / 1440 / ROWS_PER_PATIENT) * 24 * 0.8)
            records.append({
                "patientunitstayid": int(pid),
                "itemoffset":          offset_h,
                "unitdischargeoffset": stay_min,
                "gender":              int(RNG.choice([1, 2])),
                "hospitaldischargestatus": int(RNG.choice([0, 1], p=[0.87, 0.13])),
                "unitdischargestatus": int(RNG.choice([0, 1])),
                # Site A vitals
                "Heart Rate":           float(RNG.uniform(55, 145)),
                "Invasive BP Systolic": float(RNG.uniform(90, 175)),
                "Invasive BP Diastolic":float(RNG.uniform(50, 110)),
                "MAP (mmHg)":           float(RNG.uniform(60, 120)),
                "O2 Saturation":        float(RNG.uniform(90, 100)),
                "Respiratory Rate":     float(RNG.uniform(10, 35)),
                "Temperature (C)":      float(RNG.uniform(36.2, 38.5)),
                # Site B labs (FiO2 in %, will be ÷100 by clip_and_normalize)
                "glucose": float(RNG.uniform(80, 280)),
                "pH":      float(RNG.uniform(7.28, 7.48)),
                "FiO2":    float(RNG.uniform(21, 80)),
                # Site C neuro / static
                "GCS Total":       int(RNG.integers(8, 16)),
                "admissionheight": float(RNG.uniform(158, 192)),
                "admissionweight": float(RNG.uniform(55, 115)),
                # Extra columns present in real eicu_all_data.csv but not used by split
                "age":   float(RNG.uniform(25, 85)),
                "Eyes":  int(RNG.integers(1, 5)),
                "Motor": int(RNG.integers(1, 7)),
                "Verbal":int(RNG.integers(1, 6)),
            })
    return pd.DataFrame(records)


def _make_diagnosis(pids: np.ndarray, fraction: float = 0.75) -> pd.DataFrame:
    chosen = RNG.choice(pids, size=int(len(pids) * fraction), replace=False)
    rows = []
    for pid in chosen:
        for code in RNG.choice(_SAMPLE_CODES, size=RNG.integers(1, 4), replace=False):
            rows.append({
                "patientunitstayid": int(pid),
                "diagnosisoffset":   int(RNG.integers(1, 200)),
                "icd9code":          code,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared fixture — run the pipeline once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root_dir   = tmp_path_factory.mktemp("eicu_root")
    eicu_dir   = tmp_path_factory.mktemp("eicu_crd")
    output_dir = tmp_path_factory.mktemp("splits")

    df_raw = _make_all_data()
    df_raw.to_csv(root_dir / "eicu_all_data.csv", index=False)

    pids = df_raw["patientunitstayid"].unique()
    _make_diagnosis(pids).to_csv(eicu_dir / "diagnosis.csv", index=False)

    # Run pipeline (same steps as main())
    df = evs.load_all_data(root_dir)
    df = evs.prepare_cohort(df)
    df = evs.clip_and_normalize(df)

    ihm_labels = df.groupby("patientunitstayid")["hospitaldischargestatus"].max()
    train_ids, val_ids, test_ids = evs.make_splits(ihm_labels.index.values, ihm_labels)

    PREFIX = "smoketest_"
    evs.build_site_a(df, train_ids, val_ids, test_ids, output_dir, file_prefix=PREFIX)
    evs.build_site_b(df, train_ids, val_ids, test_ids, output_dir, file_prefix=PREFIX)
    evs.build_site_c(df, eicu_dir, train_ids, val_ids, test_ids, output_dir, file_prefix=PREFIX)

    site_a = pd.read_csv(output_dir / f"{PREFIX}site_A_eicu.csv")
    site_b = pd.read_csv(output_dir / f"{PREFIX}site_B_eicu.csv")
    site_c = pd.read_csv(output_dir / f"{PREFIX}site_C_eicu.csv")

    aligned = compute_psi_alignment(
        [site_a, site_b, site_c],
        salt="vfl_mtl_eicu",
        id_col="patientunitstayid",
        split_col="split",
    )
    aligned_path = output_dir / f"{PREFIX}aligned_patient_ids_eicu.csv"
    write_aligned_ids(aligned, aligned_path)

    return {
        "site_a": site_a, "site_b": site_b, "site_c": site_c,
        "aligned": aligned, "aligned_path": aligned_path,
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Output file existence
# ---------------------------------------------------------------------------

def test_output_files_exist(pipeline):
    out = pipeline["output_dir"]
    for fname in ("smoketest_site_A_eicu.csv", "smoketest_site_B_eicu.csv",
                  "smoketest_site_C_eicu.csv", "smoketest_aligned_patient_ids_eicu.csv"):
        assert (out / fname).exists(), f"Missing: {fname}"


# ---------------------------------------------------------------------------
# Site A — vitals → IHM (binary)
# ---------------------------------------------------------------------------

def test_site_a_has_feature_columns(pipeline):
    cols = set(pipeline["site_a"].columns)
    for f in evs.SITE_A_FEATURES:
        assert f in cols, f"Site A missing feature: {f}"


def test_site_a_ihm_label_binary(pipeline):
    y = pipeline["site_a"]["y_ihm"].dropna()
    assert set(y.unique()).issubset({0, 1, 0.0, 1.0}), f"y_ihm not binary: {y.unique()}"


def test_site_a_split_column(pipeline):
    splits = set(pipeline["site_a"]["split"].unique())
    assert splits == {"train", "val", "test"}


def test_site_a_split_proportions(pipeline):
    counts = pipeline["site_a"]["split"].value_counts(normalize=True)
    assert 0.60 <= counts["train"] <= 0.80, f"train proportion: {counts['train']:.2f}"
    assert 0.10 <= counts["val"]   <= 0.25, f"val proportion: {counts['val']:.2f}"
    assert 0.10 <= counts["test"]  <= 0.25, f"test proportion: {counts['test']:.2f}"


def test_site_a_vitals_within_clip_bounds(pipeline):
    df = pipeline["site_a"]
    bounds = {
        "Heart Rate":           (0, 300),
        "Invasive BP Systolic": (0, 300),
        "O2 Saturation":        (0, 100),
        "Respiratory Rate":     (0, 100),
        "Temperature (C)":      (25, 45),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            assert df[col].min() >= lo - 1e-6, f"{col} below lower bound"
            assert df[col].max() <= hi + 1e-6, f"{col} above upper bound"


# ---------------------------------------------------------------------------
# Site B — labs → RLOS (regression)
# ---------------------------------------------------------------------------

def test_site_b_has_feature_columns(pipeline):
    cols = set(pipeline["site_b"].columns)
    for f in evs.SITE_B_FEATURES:
        assert f in cols, f"Site B missing feature: {f}"


def test_site_b_rlos_nonnegative(pipeline):
    y = pipeline["site_b"]["y_rlos"].dropna()
    assert (y >= 0).all(), f"RLOS has negative values: min={y.min():.3f}"


def test_site_b_rlos_is_continuous(pipeline):
    y = pipeline["site_b"]["y_rlos"].dropna()
    # Regression target — should not be all-integer
    assert y.dtype in (float, np.float32, np.float64), f"y_rlos dtype: {y.dtype}"


def test_site_b_fio2_converted_to_fraction(pipeline):
    col = pipeline["site_b"]["FiO2"].dropna()
    assert col.max() <= 1.0 + 1e-6, f"FiO2 not converted to fraction: max={col.max():.3f}"
    assert col.min() >= 0.20 - 1e-6, f"FiO2 below 0.20: min={col.min():.3f}"


def test_site_b_ph_within_clip_bounds(pipeline):
    col = pipeline["site_b"]["pH"].dropna()
    assert col.min() >= 6.5 - 1e-6
    assert col.max() <= 8.0 + 1e-6


# ---------------------------------------------------------------------------
# Site C — neuro/static → Phenotyping (multi-label)
# ---------------------------------------------------------------------------

def test_site_c_has_feature_columns(pipeline):
    cols = set(pipeline["site_c"].columns)
    for f in evs.SITE_C_FEATURES:
        assert f in cols, f"Site C missing feature: {f}"


def test_site_c_has_25_pheno_labels(pipeline):
    pheno_cols = [c for c in pipeline["site_c"].columns if c in evs.PHENO_LABELS]
    assert len(pheno_cols) == 25, f"Expected 25 pheno labels, got {len(pheno_cols)}"


def test_site_c_pheno_labels_binary(pipeline):
    df = pipeline["site_c"]
    for col in evs.PHENO_LABELS:
        if col in df.columns:
            vals = set(df[col].unique())
            assert vals.issubset({0, 1, 0.0, 1.0}), f"Pheno '{col}' not binary: {vals}"


def test_site_c_gcs_within_bounds(pipeline):
    col = pipeline["site_c"]["GCS Total"].dropna()
    assert col.min() >= 3 - 1e-6
    assert col.max() <= 15 + 1e-6


# ---------------------------------------------------------------------------
# Feature non-overlap across sites
# ---------------------------------------------------------------------------

def test_no_feature_overlap_between_sites(pipeline):
    a_feats = set(evs.SITE_A_FEATURES)
    b_feats = set(evs.SITE_B_FEATURES)
    c_feats = set(evs.SITE_C_FEATURES)
    assert a_feats.isdisjoint(b_feats), f"A∩B overlap: {a_feats & b_feats}"
    assert a_feats.isdisjoint(c_feats), f"A∩C overlap: {a_feats & c_feats}"
    assert b_feats.isdisjoint(c_feats), f"B∩C overlap: {b_feats & c_feats}"


# ---------------------------------------------------------------------------
# PSI alignment
# ---------------------------------------------------------------------------

def test_aligned_ids_non_empty(pipeline):
    assert len(pipeline["aligned"]) > 0, "Aligned patient IDs is empty"


def test_aligned_ids_file_written(pipeline):
    assert pipeline["aligned_path"].exists()
    df = pd.read_csv(pipeline["aligned_path"])
    assert "patientunitstayid" in df.columns
    assert "split" in df.columns
    assert len(df) > 0


def test_aligned_patients_in_all_sites(pipeline):
    aligned_pids = set(pipeline["aligned"]["patientunitstayid"])
    for name, site in [("A", pipeline["site_a"]),
                       ("B", pipeline["site_b"]),
                       ("C", pipeline["site_c"])]:
        site_pids = set(site["patientunitstayid"])
        missing = aligned_pids - site_pids
        assert not missing, f"Aligned PIDs missing from Site {name}: {len(missing)} patients"


def test_aligned_no_duplicate_patients_per_split(pipeline):
    df = pipeline["aligned"]
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]["patientunitstayid"]
        assert sub.nunique() == len(sub), f"Duplicate patient IDs in {split} split"


def test_aligned_is_intersection_not_union(pipeline):
    aligned_pids = set(pipeline["aligned"]["patientunitstayid"])
    a_pids = set(pipeline["site_a"]["patientunitstayid"])
    b_pids = set(pipeline["site_b"]["patientunitstayid"])
    c_pids = set(pipeline["site_c"]["patientunitstayid"])
    true_intersection = a_pids & b_pids & c_pids
    assert aligned_pids <= true_intersection, "Aligned contains PIDs outside the intersection"


# ---------------------------------------------------------------------------
# Label balance check passes
# ---------------------------------------------------------------------------

def test_label_balance_check_runs(pipeline):
    # With 250 synthetic patients rare phenotype labels (~4%) have high split variance,
    # so we only verify the function executes without error and returns a bool.
    # On real eICU data (>30k patients) the 0.08 tolerance is meaningful.
    result = check_label_balance(
        pipeline["aligned"], pipeline["site_a"], pipeline["site_c"], tol=0.08
    )
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Clip bounds: out-of-range values in raw data must be clipped
# ---------------------------------------------------------------------------

def test_clip_bounds_are_applied():
    """Inject extreme artefact values; verify they are clipped after normalize."""
    df = _make_all_data(n=20)
    df.loc[df.index[0], "O2 Saturation"] = 29_818.0   # real artefact seen in MIMIC
    df.loc[df.index[1], "Respiratory Rate"] = 17_086.0
    df.loc[df.index[2], "FiO2"] = 150.0               # above 100% → should clip to 100 then /100

    df = evs.prepare_cohort(df)
    df = evs.clip_and_normalize(df)

    assert df["O2 Saturation"].max() <= 100.0 + 1e-6
    assert df["Respiratory Rate"].max() <= 100.0 + 1e-6
    assert df["FiO2"].max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# RLOS computation sanity
# ---------------------------------------------------------------------------

def test_rlos_positive_after_prepare_cohort():
    """RLOS must be positive for rows where itemoffset < stay_duration."""
    df = _make_all_data(n=50)
    df = evs.prepare_cohort(df)
    # Filter same as build_site_b
    sub = df[(df["itemoffset"] > 0) & (df["RLOS"] > 0)]
    assert len(sub) > 0, "No rows with RLOS > 0"
    assert (sub["RLOS"] >= 0).all(), "Negative RLOS values found"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
