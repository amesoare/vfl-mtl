# EDA Scientific Rigour — Suggestions

Generated from scientific critical thinking analysis of the EDA notebook and data pipeline.

---

## Tier 1: Blocking — Fix Before Training

### 1. Outlier Clipping Before Imputation

**File:** `data_prep/vertical_split.py`

Implausible values exist in the raw data (SpO2 max = 29,818%, Temperature max = 285°C, RespRate max = 17,086, Glucose max = 15,356, pH max = 99, Weight max = 3,761,721). The current pipeline runs mean-imputation on columns containing these values, meaning every patient with a missing feature inherits a corrupted imputed mean.

**Fix:** Apply clinical range clipping *before* imputation. Example bounds:

| Feature | Min | Max |
|---------|-----|-----|
| Heart Rate | 0 | 300 |
| Systolic BP | 0 | 300 |
| Diastolic BP | 0 | 200 |
| Temperature (°C) | 25 | 45 |
| SpO2 (%) | 0 | 100 |
| Respiratory Rate | 0 | 100 |
| Glucose | 0 | 2000 |
| pH | 6.5 | 8.0 |
| Weight (kg) | 0 | 500 |

Document how many stays are affected per feature after clipping.

---

### 2. Stratified Train/Val/Test Splits

**File:** `data_prep/vertical_split.py`

IHM has a 13% positive rate. A random split can produce test sets with 8–18% positives, inflating variance in AUC-ROC across runs. The YerevaNN pipeline does not guarantee stratification across the vertical split CSVs.

**Fix:** Use `sklearn.model_selection.train_test_split(..., stratify=y_ihm)` when writing the site CSVs, and verify IHM positive rate is consistent (±2%) across train/val/test in `verify_workspace.py`.

---

## Tier 2: Paper Quality — Do Before Submission

### 3. PSI Alignment Attrition Table

PSI alignment reduces ~41,000 per-site stays to 18,094 aligned patients (~57% reduction). The EDA does not characterise who is dropped.

**Fix:** Add a CONSORT-style attrition table to the EDA notebook comparing aligned vs. non-aligned patients on: age, gender, mortality rate, median LOS, and phenotype prevalence. If the retained cohort is systematically different (e.g., longer stays, higher acuity), this must be acknowledged as a scope limitation.

---

### 4. Cross-Label Correlation Matrix

IHM, LOS, and phenotyping labels are never jointly examined. In MIMIC-III, mortality and several phenotypes (e.g., sepsis, respiratory failure) are strongly correlated. This directly affects interpretation of MTL results — high label correlation reduces expected negative transfer; low correlation increases it.

**Fix:** Add a cross-label correlation matrix to the EDA: φ-coefficient between IHM and each of the 25 phenotype labels; Spearman correlation between IHM and LOS bins. Use this to motivate expected task relatedness in the methods section.

---

### 5. Feature-to-Task Signal Analysis

Site C has 3 features (Height, Weight, Mean BP) and is assigned 25 phenotyping labels — the most complex task. Site A has 7 features and the simplest binary task. Any observed performance differential between sites could be explained by this feature-count/task-complexity co-variation rather than federated learning dynamics.

**Fix:** Add per-site univariate correlation between each feature and its assigned labels in the EDA. Provides baseline signal-to-task evidence and strengthens the experimental design justification.

---

## Tier 3: Acknowledge as Limitations (Thesis Methods/Discussion)

### 6. Missing Data Mechanism

The pipeline applies forward-fill → backward-fill → zero-fill without assessing *why* data is missing. In ICU settings, missingness is frequently MNAR (not measured because clinician already has enough information, or patient too unstable). The binary observation mask passed to the LSTM partially mitigates this but does not eliminate the assumption.

**Action:** State explicitly in the methods section that the imputation strategy assumes MAR and acknowledge MNAR as a limitation. No code change required.

---

### 7. Temporal Pattern Analysis

The observation rate heatmap is descriptive. It does not examine whether temporal missingness patterns differ between positive and negative cases, or whether informative measurement timing (e.g., labs drawn at admission vs. deterioration) affects representation.

**Action:** Optional addition to the EDA. At minimum, acknowledge in the discussion that temporal observation patterns were not used as features.

---

### 8. Cohort Definition

The cohort inherits YerevaNN MIMIC-III benchmark inclusion/exclusion criteria (ICU stay ≥ 48h, age ≥ 18, etc.) without restating them.

**Action:** Add a one-paragraph cohort definition to the EDA or methods section, citing Harutyunyan et al. (2019).

---

### 9. Pearson vs. Spearman Correlation

The within-site and cross-site correlation analyses use Pearson correlation, which is sensitive to the outliers already identified (e.g., Weight std = 18,377). Correlations may be distorted before outlier clipping is applied.

**Action:** After applying Tier 1 fixes, re-run correlation analyses. Consider reporting Spearman as a robustness check.

---

## Summary Table

| # | Issue | Severity | When to Fix |
|---|-------|----------|-------------|
| 1 | Outlier clipping before imputation | **Critical** | Before any training |
| 2 | Stratified train/val/test splits | **Critical** | Before any training |
| 3 | PSI attrition characterisation | Important | Before submission |
| 4 | Cross-label correlation matrix | Important | Before submission |
| 5 | Feature-to-task signal analysis | Important | Before submission |
| 6 | Missing data mechanism statement | Minor | Methods/discussion section |
| 7 | Temporal pattern analysis | Minor | Optional EDA addition |
| 8 | Cohort definition | Minor | Methods section |
| 9 | Pearson → Spearman robustness check | Minor | After Tier 1 fixes |
