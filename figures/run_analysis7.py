"""
figures/run_analysis7.py  — standalone re-run of EDA Analysis 7 only.

Reads:
  data/vertical_splits/site_A_vitals.csv
  data/vertical_splits/site_B_labs.csv
  data/vertical_splits/site_C_composite.csv
  data/vertical_splits/aligned_patient_ids.csv

Writes:
  figures/fig7_featuretosignal.png
  figures/fig7_featuretosignal.svg  (vector, for paper)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/vertical_splits")
OUT      = Path("figures")
OUT.mkdir(exist_ok=True)

site_a  = pd.read_csv(DATA_DIR / "site_A_vitals.csv")
site_b  = pd.read_csv(DATA_DIR / "site_B_labs.csv")
site_c  = pd.read_csv(DATA_DIR / "site_C_composite.csv")
aligned = pd.read_csv(DATA_DIR / "aligned_patient_ids.csv")

print(f"Aligned cohort: {len(aligned):,} patients "
      f"(train={aligned['split'].eq('train').sum():,}, "
      f"val={aligned['split'].eq('val').sum():,}, "
      f"test={aligned['split'].eq('test').sum():,})")

# ── Feature / label column definitions ────────────────────────────────────────
FEATURES_A = [
    "Heart Rate", "Systolic blood pressure", "Diastolic blood pressure",
    "Temperature", "Oxygen saturation", "Respiratory rate",
    "Glascow coma scale total",
]
FEATURES_B = ["Glucose", "pH", "Fraction inspired oxygen", "Capillary refill rate"]
FEATURES_C = ["Height", "Weight", "Mean blood pressure"]

NON_LABEL_COLS_C = ["stay", "subject_id"] + FEATURES_C + ["split"]
PHENO_COLS = [c for c in site_c.columns if c not in NON_LABEL_COLS_C]

DISPLAY_NAMES_ALL = {
    "Heart Rate":                  "Heart Rate",
    "Systolic blood pressure":     "Systolic BP",
    "Diastolic blood pressure":    "Diastolic BP",
    "Temperature":                 "Temperature",
    "Oxygen saturation":           "SpO2",
    "Respiratory rate":            "Resp. Rate",
    "Glascow coma scale total":    "GCS Total",
    "Glucose":                     "Glucose",
    "pH":                          "pH",
    "Fraction inspired oxygen":    "FiO2",
    "Capillary refill rate":       "Cap. Refill",
    "Height":                      "Height",
    "Weight":                      "Weight",
    "Mean blood pressure":         "Mean BP",
}

PALETTE = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 12, "axes.labelsize": 10})

# ── Analysis 7: Feature-to-task signal ────────────────────────────────────────
train_ids = set(aligned.loc[aligned["split"] == "train", "subject_id"])

print(f"Training subjects used for signal analysis: {len(train_ids):,}")

def train_first_stay(df, cols):
    return (
        df[(df["subject_id"].isin(train_ids)) & (df["split"] == "train")]
        .drop_duplicates("subject_id")
        .set_index("subject_id")[cols]
    )

feats      = pd.concat([
    train_first_stay(site_a, FEATURES_A),
    train_first_stay(site_b, FEATURES_B),
    train_first_stay(site_c, FEATURES_C),
], axis=1)
y_ihm_s    = train_first_stay(site_a, ["y_ihm"])
y_decomp_s = train_first_stay(site_b, ["y_decomp"])
y_phe_s    = train_first_stay(site_c, PHENO_COLS)

common = sorted(
    set(feats.index)
    & set(y_ihm_s.index)
    & set(y_decomp_s.index)
    & set(y_phe_s.index)
)
feats      = feats.loc[common]
y_ihm_v    = y_ihm_s.loc[common, "y_ihm"].values
y_decomp_v = y_decomp_s.loc[common, "y_decomp"].values
y_phe_v    = y_phe_s.loc[common].values   # (N, 25)

print(f"Common subjects with all labels: {len(common):,}")

all_feats = FEATURES_A + FEATURES_B + FEATURES_C
rows = []
for feat in all_feats:
    x  = feats[feat].values
    ok = ~np.isnan(x)
    r_ihm    = abs(np.corrcoef(x[ok], y_ihm_v[ok])[0, 1])
    r_decomp = abs(np.corrcoef(x[ok], y_decomp_v[ok])[0, 1])
    r_pheno  = float(np.nanmean(
        [abs(np.corrcoef(x[ok], y_phe_v[ok, j])[0, 1]) for j in range(25)]
    ))
    rows.append({
        "Feature":             DISPLAY_NAMES_ALL.get(feat, feat),
        "IHM":                 r_ihm,
        "Decomp":              r_decomp,
        "Phenotyping\n(mean |r|)": r_pheno,
    })

signal = pd.DataFrame(rows).set_index("Feature")

print("\n── Feature-to-task |r| table ──")
print(signal.round(4).to_string())

# Site boundary positions (row indices in heatmap)
cuts = [len(FEATURES_A), len(FEATURES_A) + len(FEATURES_B)]

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    signal,
    annot=True, fmt=".3f",
    cmap="YlOrRd",
    vmin=0, vmax=0.3,
    linewidths=0.5,
    cbar_kws={"label": "|r|", "shrink": 0.6},
    ax=ax,
)

for b in cuts:
    ax.axhline(b, color="white", linewidth=2.5)

for pos, name, color in zip(
    [
        len(FEATURES_A) / 2,
        len(FEATURES_A) + len(FEATURES_B) / 2,
        len(FEATURES_A) + len(FEATURES_B) + len(FEATURES_C) / 2,
    ],
    ["Site A", "Site B", "Site C"],
    [PALETTE[0], PALETTE[4], PALETTE[6]],
):
    ax.text(
        -0.35, pos, name,
        va="center", ha="right", fontsize=9,
        color=color, fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

ax.set_title(
    "Analysis 7 — Feature-to-task Signal\n"
    f"|r| per feature–task pair  (aligned training, n={len(common):,}, no leakage)",
    fontsize=11,
)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()

for ext in ("png", "svg"):
    out_path = OUT / f"fig7_featuretosignal.{ext}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
