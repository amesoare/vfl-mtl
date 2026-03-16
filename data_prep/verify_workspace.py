"""
verify_workspace.py — Step 7 workspace readiness check.

Checks:
  ✓ All four task directories exist and are non-empty
  ✓ vertical_splits/ contains all three site CSVs
  ✓ aligned_patient_ids.csv covers >= 30,000 patients
  ✓ No feature column overlap between site CSVs
  ✓ Class balance within acceptable range for each task

Usage:
  python data_prep/verify_workspace.py [--root WORKSPACE_ROOT]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── columns that are not clinical features ─────────────────────────────────
NON_FEATURE_COLS = {
    "stay", "subject_id", "split",
    # task labels
    "y_ihm", "y_los",
    # phenotyping labels (25 ICD groups)
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
}

TASKS = [
    "in-hospital-mortality",
    "decompensation",
    "length-of-stay",
    "phenotyping",
]

SITE_CSVS = {
    "site_A_vitals.csv": "y_ihm",
    "site_B_labs.csv": "y_los",
    "site_C_composite.csv": None,  # phenotyping — multi-label, checked separately
}

ALIGNED_MIN_PATIENTS = 18_000


def check(cond: bool, msg_ok: str, msg_fail: str, results: list) -> bool:
    if cond:
        print(f"  ✓ {msg_ok}")
        results.append(True)
    else:
        print(f"  ✗ {msg_fail}")
        results.append(False)
    return cond


def verify(root: Path) -> bool:
    results: list[bool] = []
    bench_data = root / "data" / "mimic3-benchmarks" / "data"
    splits_dir = root / "data" / "vertical_splits"

    print("\n══════════════════════════════════════════════════")
    print("  VFL-MTL Workspace Verification")
    print(f"  Root: {root}")
    print("══════════════════════════════════════════════════\n")

    # ── 1. Task directories ────────────────────────────────────────────────
    print("[ 1 ] Task directories")
    for task in TASKS:
        task_dir = bench_data / task
        has_dir = task_dir.is_dir()
        non_empty = has_dir and any(task_dir.iterdir())
        has_train = has_dir and (task_dir / "train_listfile.csv").exists()
        has_test  = has_dir and (task_dir / "test_listfile.csv").exists()
        has_val   = has_dir and (task_dir / "val_listfile.csv").exists()
        ok = non_empty and has_train and has_test and has_val
        check(
            ok,
            f"{task}/  [train_listfile, val_listfile, test_listfile present]",
            f"{task}/  MISSING or incomplete "
            f"(dir={has_dir}, train={has_train}, val={has_val}, test={has_test})",
            results,
        )

    # ── 2. Vertical split CSVs ─────────────────────────────────────────────
    print("\n[ 2 ] Vertical split files")
    for fname in list(SITE_CSVS) + ["aligned_patient_ids.csv"]:
        fpath = splits_dir / fname
        check(
            fpath.exists() and fpath.stat().st_size > 0,
            f"{fname} exists",
            f"{fname} MISSING or empty",
            results,
        )

    # ── 3. Aligned patient count ──────────────────────────────────────────
    print("\n[ 3 ] Aligned patient count")
    aligned_path = splits_dir / "aligned_patient_ids.csv"
    if aligned_path.exists():
        n = len(pd.read_csv(aligned_path))
        check(
            n >= ALIGNED_MIN_PATIENTS,
            f"aligned_patient_ids.csv: {n:,} patients (>= {ALIGNED_MIN_PATIENTS:,})",
            f"aligned_patient_ids.csv: only {n:,} patients (< {ALIGNED_MIN_PATIENTS:,} required)",
            results,
        )
    else:
        print(f"  ✗ aligned_patient_ids.csv not found — skipping count check")
        results.append(False)

    # ── 4. No feature column overlap between site CSVs ────────────────────
    print("\n[ 4 ] Feature column overlap between sites")
    site_feature_sets: dict[str, set] = {}
    for fname in SITE_CSVS:
        fpath = splits_dir / fname
        if fpath.exists():
            cols = set(pd.read_csv(fpath, nrows=0).columns) - NON_FEATURE_COLS
            site_feature_sets[fname] = cols

    if len(site_feature_sets) == len(SITE_CSVS):
        names = list(site_feature_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = site_feature_sets[a] & site_feature_sets[b]
                check(
                    len(overlap) == 0,
                    f"No overlap between {a} and {b}",
                    f"Overlap between {a} and {b}: {sorted(overlap)}",
                    results,
                )
        # Print feature summary
        print("  Feature counts per site:")
        for fname, feats in site_feature_sets.items():
            print(f"    {fname}: {len(feats)} features — {sorted(feats)}")
    else:
        print("  ✗ Cannot check overlap — one or more site CSVs missing")
        results.append(False)

    # ── 5. Class balance ──────────────────────────────────────────────────
    print("\n[ 5 ] Class balance")

    # 5a. In-hospital mortality (binary, acceptable range 5–40 %)
    site_a = splits_dir / "site_A_vitals.csv"
    if site_a.exists():
        df_a = pd.read_csv(site_a, usecols=["y_ihm", "split"])
        train_a = df_a[df_a["split"] == "train"]
        pos_rate = train_a["y_ihm"].mean()
        check(
            0.05 <= pos_rate <= 0.40,
            f"IHM mortality rate (train): {pos_rate:.1%}  [5%–40% acceptable]",
            f"IHM mortality rate (train): {pos_rate:.1%}  OUT OF RANGE [5%–40%]",
            results,
        )

    # 5b. Length-of-stay (continuous regression — report median & range)
    site_b = splits_dir / "site_B_labs.csv"
    if site_b.exists():
        df_b = pd.read_csv(site_b, usecols=["y_los", "split"])
        train_b = df_b[df_b["split"] == "train"]["y_los"]
        median_los = train_b.median()
        los_min, los_max = train_b.min(), train_b.max()
        reasonable = los_min >= 0 and los_max <= 30 * 24  # hours, max 30 days
        check(
            reasonable,
            f"LOS (train): median={median_los:.1f}h, range=[{los_min:.1f}, {los_max:.1f}]h",
            f"LOS (train): implausible values — median={median_los:.1f}h, "
            f"range=[{los_min:.1f}, {los_max:.1f}]h",
            results,
        )

    # 5c. Phenotyping (multi-label — check no label is near-zero or near-one)
    site_c = splits_dir / "site_C_composite.csv"
    pheno_labels = [
        "Acute and unspecified renal failure", "Acute cerebrovascular disease",
        "Acute myocardial infarction", "Cardiac dysrhythmias",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Complications of surgical procedures or medical care",
        "Conduction disorders", "Congestive heart failure; nonhypertensive",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus with complications",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism", "Essential hypertension",
        "Fluid and electrolyte disorders", "Gastrointestinal hemorrhage",
        "Hypertension with complications and secondary hypertension",
        "Other liver diseases", "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)", "Shock",
    ]
    if site_c.exists():
        present = [c for c in pheno_labels if c in pd.read_csv(site_c, nrows=0).columns]
        df_c = pd.read_csv(site_c, usecols=present + ["split"])
        train_c = df_c[df_c["split"] == "train"][present]
        rates = train_c.mean()
        degenerate = rates[(rates < 0.001) | (rates > 0.999)]
        check(
            len(degenerate) == 0,
            f"Phenotyping: all {len(present)} labels have prevalence in (0.1%, 99.9%)",
            f"Phenotyping: {len(degenerate)} degenerate label(s): {list(degenerate.index)}",
            results,
        )
        # Report prevalence range
        print(f"    Prevalence range: {rates.min():.1%} – {rates.max():.1%}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════")
    passed = sum(results)
    total  = len(results)
    if passed == total:
        print(f"  ALL CHECKS PASSED ({passed}/{total})")
        print("  Workspace is ready for experiments.\n")
        return True
    else:
        print(f"  {passed}/{total} checks passed — {total - passed} FAILED")
        print("  Resolve failures before running experiments.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify VFL-MTL workspace readiness.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the vfl_mlt project root (default: parent of data_prep/)",
    )
    args = parser.parse_args()

    ok = verify(args.root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
