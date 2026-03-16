"""
data_prep/psi_alignment.py — Simulated Private Set Intersection (PSI) for patient alignment.

Simulates a privacy-preserving PSI protocol:
  1. Each site hashes its patient IDs with SHA-256 (+ shared salt).
  2. The server computes the intersection of hashed ID sets.
  3. Returns the aligned patient index (original IDs) for training.

In production VFL the raw IDs never leave each site — only hashed tokens are
compared. This module simulates that protocol locally for benchmarking purposes.

Usage (module):
    from data_prep.psi_alignment import compute_psi_alignment, write_aligned_ids

Usage (CLI):
    python data_prep/psi_alignment.py \\
        --site_a data/vertical_splits/site_A_vitals.csv \\
        --site_b data/vertical_splits/site_B_labs.csv \\
        --site_c data/vertical_splits/site_C_composite.csv \\
        --output data/vertical_splits/aligned_patient_ids.csv \\
        [--salt my_shared_salt]
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Set

import pandas as pd

DEFAULT_SALT = "vfl_mtl_mimic3"


# ── Hashing ────────────────────────────────────────────────────────────────

def hash_id(patient_id: int, salt: str = DEFAULT_SALT) -> str:
    """SHA-256 hash of <salt>||<patient_id>."""
    token = f"{salt}:{patient_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def hash_id_set(patient_ids: Set[int], salt: str = DEFAULT_SALT) -> Dict[str, int]:
    """Return {hash → original_id} mapping for a set of patient IDs."""
    return {hash_id(pid, salt): pid for pid in patient_ids}


# ── PSI computation ────────────────────────────────────────────────────────

def compute_psi_alignment(
    site_dfs: list[pd.DataFrame],
    splits: list[str] = ("train", "val", "test"),
    salt: str = DEFAULT_SALT,
    id_col: str = "subject_id",
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Simulate PSI across N site DataFrames and return aligned patient index.

    Parameters
    ----------
    site_dfs  : list of DataFrames, each with at least `id_col` and `split_col`
    splits    : which split labels to process
    salt      : shared salt for SHA-256 hashing
    id_col    : name of the patient ID column
    split_col : name of the split column

    Returns
    -------
    DataFrame with columns [id_col, split_col] — one row per aligned patient per split
    """
    rows = []

    for split in splits:
        # Step 1 — each site hashes its patient IDs (only hashes are "shared")
        hashed_sets: list[Dict[str, int]] = []
        for df in site_dfs:
            subset = df.loc[df[split_col] == split, id_col]
            hashed_sets.append(hash_id_set(set(subset), salt))

        # Step 2 — server computes intersection of hashed token sets
        common_hashes: Set[str] = set(hashed_sets[0].keys())
        for hs in hashed_sets[1:]:
            common_hashes &= set(hs.keys())

        # Step 3 — recover original IDs from any site's reverse map (all identical for matching IDs)
        aligned_ids = sorted(hashed_sets[0][h] for h in common_hashes)

        for pid in aligned_ids:
            rows.append({id_col: pid, split_col: split})

        print(f"  PSI [{split}]: {len(aligned_ids):,} aligned patients "
              f"({', '.join(f'site{i+1}={len(hs):,}' for i, hs in enumerate(hashed_sets))})")

    return pd.DataFrame(rows)


def write_aligned_ids(aligned: pd.DataFrame, output_path: Path) -> None:
    """Write the aligned patient index to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    print(f"  → {output_path}  ({len(aligned):,} rows)")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site_a", required=True,
                        help="Path to site_A_vitals.csv")
    parser.add_argument("--site_b", required=True,
                        help="Path to site_B_labs.csv")
    parser.add_argument("--site_c", required=True,
                        help="Path to site_C_composite.csv")
    parser.add_argument("--output", required=True,
                        help="Output path for aligned_patient_ids.csv")
    parser.add_argument("--salt", default=DEFAULT_SALT,
                        help=f"Shared salt for SHA-256 hashing (default: '{DEFAULT_SALT}')")
    args = parser.parse_args()

    dfs = []
    for path in (args.site_a, args.site_b, args.site_c):
        p = Path(path)
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        dfs.append(pd.read_csv(p, usecols=["subject_id", "split"]))

    print("Running PSI alignment ...")
    aligned = compute_psi_alignment(dfs, salt=args.salt)
    write_aligned_ids(aligned, Path(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
