"""
data_prep/eicu_psi_alignment.py — Simulated PSI for eICU vertical split alignment.

Mirrors psi_alignment.py for the eICU dataset. Differences from the MIMIC version:
  - Patient ID column: patientunitstayid (vs. subject_id)
  - Site B label y_rlos is a regression target — skipped in balance check
  - Phenotype label names match eicu_vertical_split.py (short strings, not YerevaNN names)
  - No skmultilearn re-stratification: eicu_vertical_split.py already stratifies by IHM

Usage:
    python data_prep/eicu_psi_alignment.py \\
        --site_a data/eicu_vertical_splits/site_A_eicu.csv \\
        --site_b data/eicu_vertical_splits/site_B_eicu.csv \\
        --site_c data/eicu_vertical_splits/site_C_eicu.csv \\
        --output data/eicu_vertical_splits/aligned_patient_ids_eicu.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_prep.psi_alignment import compute_psi_alignment, write_aligned_ids

ID_COL    = "patientunitstayid"
SPLIT_COL = "split"
SALT      = "vfl_mtl_eicu"

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
assert len(PHENO_LABELS) == 25


def check_label_balance(
    aligned_ids: pd.DataFrame,
    site_a: pd.DataFrame,
    site_c: pd.DataFrame,
    tol: float = 0.03,
) -> bool:
    """
    Check IHM (binary) and phenotype (binary) balance across splits.
    Site B (y_rlos) is a regression target — not checked here.
    """
    a_labels = site_a[[ID_COL, "y_ihm"]].drop_duplicates(subset=ID_COL)
    pheno_present = [c for c in PHENO_LABELS if c in site_c.columns]
    c_labels = site_c[[ID_COL] + pheno_present].drop_duplicates(subset=ID_COL)

    merged = (
        aligned_ids
        .merge(a_labels, on=ID_COL, how="left")
        .merge(c_labels, on=ID_COL, how="left")
    )

    train_ihm = merged.loc[merged[SPLIT_COL] == "train", "y_ihm"].mean()
    for split in ["val", "test"]:
        split_ihm = merged.loc[merged[SPLIT_COL] == split, "y_ihm"].mean()
        if abs(split_ihm - train_ihm) > tol:
            print(f"  IHM imbalance in {split}: train={train_ihm:.3f}, {split}={split_ihm:.3f}")
            return False

    for col in pheno_present:
        train_prev = merged.loc[merged[SPLIT_COL] == "train", col].mean()
        for split in ["val", "test"]:
            split_prev = merged.loc[merged[SPLIT_COL] == split, col].mean()
            if abs(split_prev - train_prev) > tol:
                print(f"  Pheno '{col}' imbalance in {split}: "
                      f"train={train_prev:.3f}, {split}={split_prev:.3f}")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site_a", required=True, help="Path to site_A_eicu.csv")
    parser.add_argument("--site_b", required=True, help="Path to site_B_eicu.csv")
    parser.add_argument("--site_c", required=True, help="Path to site_C_eicu.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for aligned_patient_ids_eicu.csv")
    parser.add_argument("--salt", default=SALT)
    parser.add_argument("--tol", type=float, default=0.03)
    args = parser.parse_args()

    site_dfs = {}
    for key, path in [("a", args.site_a), ("b", args.site_b), ("c", args.site_c)]:
        p = Path(path)
        if not p.exists():
            print(f"Error: not found: {p}", file=sys.stderr)
            sys.exit(1)
        site_dfs[key] = pd.read_csv(p)

    print("Running PSI alignment ...")
    aligned = compute_psi_alignment(
        [site_dfs["a"], site_dfs["b"], site_dfs["c"]],
        salt=args.salt,
        id_col=ID_COL,
        split_col=SPLIT_COL,
    )

    print("Checking label balance (IHM + phenotypes; y_rlos skipped — regression) ...")
    ok = check_label_balance(aligned, site_dfs["a"], site_dfs["c"], tol=args.tol)
    if not ok:
        print("  WARNING: imbalance exceeds tolerance. Keeping eicu_vertical_split.py splits.")
    else:
        print("  Label balance OK.")

    write_aligned_ids(aligned, Path(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
