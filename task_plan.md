# Task Plan: Switch Site B Task from LOS → Decompensation

## Goal
Replace Length-of-Stay (10-class multiclass) with Decompensation (binary) as Site B's
prediction task, with minimal code disruption. Archive current LOS results CSVs.

## Status
**Currently in Phase 1** — plan written, awaiting execution

---

## Why This Change Is Safe

Decompensation is structurally identical to IHM:
- Both are **binary** classification (`task_type="binary"` already exists)
- Both use **BCELoss** (already in `fl/server.py` as `_BCE`)
- Both use **AUC-ROC + AUC-PR** as metrics (same code as IHM)
- No new code paths needed — this is almost entirely a string rename + data swap

The change from `"los_bins"` (10-class CrossEntropy) to `"binary"` (BCE) eliminates the
only non-binary task type in Site B, making all three sites binary or multilabel.

---

## Phases

- [ ] Phase 0: Archive existing LOS results
- [ ] Phase 1: Data layer — vertical_split.py, dataset.py, psi_alignment.py
- [ ] Phase 2: Model core — mmoe.py, server.py
- [ ] Phase 3: Training loop — train.py
- [ ] Phase 4: Experiments — metrics.py, run_exp1–4.py
- [ ] Phase 5: Baselines — local_only.py, centralized.py
- [ ] Phase 6: Figures — 4 figure files
- [ ] Phase 7: Tests — 3 test files

---

## Phase 0 — Archive LOS Results (1 minute)

Locally rename the four CSVs so LOS results are preserved under a `_los` suffix
and new runs write fresh files. No git operations — just filesystem renames.

```
mv results/exp1.csv results/exp1_los.csv
mv results/exp2.csv results/exp2_los.csv
mv results/exp3.csv results/exp3_los.csv
mv results/exp4.csv results/exp4_los.csv
```

The `_los` files remain on disk and will show up as renamed/modified in git status
when you later commit, but **no git commands are part of this plan**.

---

## Phase 1 — Data Layer

### 1a. `data_prep/vertical_split.py`

**What changes:**
- `TASK_DIRS`: add `"decomp": "decompensation"` (keep `"los"` or remove — doesn't matter,
  `build_site_b` uses it directly). Cleanest: rename `"los"` → `"decomp"`.
- `build_site_b()`:
  - Change `task_dir = root / TASK_DIRS["los"]` → `root / TASK_DIRS["decomp"]`
  - Decompensation listfiles have: `stay, period_length, y_true` (binary per hourly slot)
  - Per-stay label: `y_decomp = max(y_true)` across all hourly rows for the same stay
    (i.e., "did this patient ever face an imminent decompensation event?")
  - Remove the `y_los = period_length + y_true` computation
  - Rename output column `y_los` → `y_decomp`
  - Change `deduplicate` strategy: `lf_dedup = lf.groupby("stay")["y_true"].max().rename("y_decomp")`
  - Update print string and output path comment (no rename of the file `site_B_labs.csv`)

**Decompensation listfile format (YerevaNN):**
Same as LOS: multiple rows per stay (one per hourly prediction window).
`y_true = 1` if patient decompensated in the next 24h from that hour.
Max across rows → 1 if any decompensation event occurred in the stay.

**Key invariant preserved:** `site_B_labs.csv` filename stays the same, only label
column changes from `y_los` to `y_decomp`.

### 1b. `data_prep/dataset.py`

**What changes:**
- `build_site_loaders()` Site B config block:
  ```python
  # OLD
  label_col="y_los", timeseries_root=bench_dir/"length-of-stay", task_type="los_bins"
  # NEW
  label_col="y_decomp", timeseries_root=bench_dir/"decompensation", task_type="binary"
  ```
- Remove the `los_bins` branch in `__init__` (lines ~184–196) — no longer needed since
  Site B is now `"binary"`. The assert can drop `"los_bins"` or keep it for safety.
  **Minimal choice: keep `"los_bins"` in the assert, just don't use it.**
- Remove (or keep as dead code) the `CustomBins`/`get_bin_custom` imports.
  **Minimal choice: leave them in — harmless, saves a line change.**
- Docstring update: Site B description.

### 1c. `data_prep/psi_alignment.py`

**What changes:**
- `check_label_balance()` (line ~117): rename `"y_los"` → `"y_decomp"`, update comment
- `stratify_aligned_cohort()` (lines ~161–196):
  - Load `y_decomp` (binary) instead of `y_los`
  - Remove `pd.get_dummies(merged["y_los"], prefix="los")` — replace with just
    `merged[["y_decomp"]]` as a single binary column in the combined label matrix
  - The iterative stratification then handles: [y_ihm | y_decomp | y_pheno×25]
    instead of [y_ihm | y_los_onehot×10 | y_pheno×25]

---

## Phase 2 — Model Core

### 2a. `model/mmoe.py`

**What changes:**
- `TASK_ORDER = ("ihm", "los", "pheno")` → `("ihm", "decomp", "pheno")`
- `heads` dict:
  ```python
  # OLD
  "los": TaskHead(expert_out, "los_bins", n_classes=10)
  # NEW
  "decomp": TaskHead(expert_out, "binary")
  ```
- Docstring comment: `LOS (Site B) → 10-bin cls` → `Decomp (Site B) → binary`
- Module docstring output shape comment: `'los': (B, 10)` → `'decomp': (B, 1)`

**`TaskHead` itself does NOT change** — `"binary"` and `"los_bins"` types both remain
in the class; we just instantiate `"binary"` for Site B now.

### 2b. `fl/server.py`

**What changes:**
- Default `task_weights`: `{"ihm": 1.0, "los": 1.0, "pheno": 1.0}` → `{"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}`
- `forward_and_loss()`:
  - Rename key `"los"` → `"decomp"` in `task_losses` dict
  - Change loss: `_CE(preds["los"], labels["los"].to(...))` →
    `_BCE(preds["decomp"].squeeze(-1), labels["decomp"].to(...))`
  - Docstring: update `'los': (B,) int64` → `'decomp': (B,) float32`
- `task_loss_sums` in logging comment (docstring only)

---

## Phase 3 — Training Loop (`train.py`)

**What changes — in order of appearance:**

1. `TrainConfig.task_weights` default: `"los"` → `"decomp"`
2. `make_synthetic_loaders()` Site B:
   - `torch.randint(0, 10, (N,)).long()` → `torch.randint(0, 2, (N,)).float()`
   - Comment update
3. `compute_metrics()`:
   - Rename `"los"` section → `"decomp"`
   - Replace kappa with AUC-ROC + AUC-PR (same code as IHM section):
     ```python
     p_decomp = np.concatenate(all_preds["decomp"])
     y_decomp = np.concatenate(all_labels["decomp"])
     metrics["decomp_auroc"] = float(roc_auc_score(y_decomp, p_decomp))
     metrics["decomp_auprc"] = float(average_precision_score(y_decomp, p_decomp))
     ```
   - Remove `cohen_kappa_score` import if unused (IHM doesn't use it); keep if defensive
4. `train_one_round()`:
   - `task_loss_sums`: `"los"` → `"decomp"`
   - `y_los` → `y_decomp` (variable name in batch unpacking)
   - `labels["los"]` → `labels["decomp"]`
   - Return dict key: `"los_loss"` → `"decomp_loss"`
5. `evaluate()` / `_evaluate_sites()`:
   - `all_preds["los"]` → `all_preds["decomp"]`
   - `all_labels["los"]` → `all_labels["decomp"]`
   - Padding: `torch.zeros(..., dtype=torch.long)` → `torch.zeros(..., dtype=torch.float)`
6. `run_training()`:
   - Row dict: `"los_loss"` → `"decomp_loss"`
   - `"val_los_kappa"` → `"val_decomp_auroc"` + `"val_decomp_auprc"`
7. `main()` (CLI):
   - `csv_fields`: `"los_loss"` → `"decomp_loss"`, `"los_kappa"` → `"decomp_auroc"`, `"decomp_auprc"`
   - `--w-los` arg → `--w-decomp`
   - Print string in training loop
8. `_train_one_round_sites()`:
   - `task_loss_sums`: `"los"` → `"decomp"`
   - `labels["los"]` → `labels["decomp"]`
   - Padding for missing task: `torch.long` → `torch.float`
   - Return dict: `"los_loss"` → `"decomp_loss"`

---

## Phase 4 — Experiments

### 4a. `experiments/metrics.py`
- Rename `los_metrics()` → `decomp_metrics()`
- Change signature: `y_pred_bin` → `y_prob` (probability, not bin index)
- New body: AUC-ROC + AUC-PR (same as `ihm_metrics()`, different key names)
- Remove `cohen_kappa_score` import
- `compute_all_metrics()`: rename `los_*` → `decomp_*`

### 4b. `experiments/run_exp1.py`
- `CONFIGS`: `"los"` key → `"decomp"` in all task_weights dicts
- `"ST-LOS"` config name → `"ST-Decomp"`
- Docstring: update task list, CSV column descriptions
- Output CSV column comment: `val_los_kappa` → `val_decomp_auroc`

### 4c. `experiments/run_exp3.py`
- `TASK_CONFIGS`: `"ihm_los"` → `"ihm_decomp"`, `"los"` → `"decomp"` in weights
- Docstring: `LOS` → `Decompensation`
- `compute_negative_transfer()`: no changes needed (uses "val_ihm_auroc" only)

### 4d. `experiments/run_exp4.py`
- Docstring comment only: `n_sites=2: Sites A + B only (IHM + LOS tasks)` →
  `(IHM + Decomp tasks)`
- Task weights in the 2-site config if `"los"` appears explicitly

### 4e. `experiments/run_exp2.py`
- Docstring comment only: no task-key changes needed (experiment tests feature dims)

---

## Phase 5 — Baselines

### 5a. `baselines/local_only.py`
- `_SITE_CONFIGS["B"]`:
  - `"label_col": "y_los"` → `"y_decomp"`
  - `"task_type": "los_bins"` → `"binary"`
  - `"ts_subdir": "length-of-stay"` → `"decompensation"`
- `_eval_site()` or wherever metrics are called:
  - `los_metrics(...)` → `decomp_metrics(...)`
  - Metric shape: argmax of logits → probability directly (sigmoid output)
- Loss: `nn.CrossEntropyLoss()` → `nn.BCELoss()` for Site B
- Import: `los_metrics` → `decomp_metrics`
- Docstring: Site B description

### 5b. `baselines/centralized.py`
- Site B dataset construction:
  - `label_col="y_los"` → `"y_decomp"`, `task_type="los_bins"` → `"binary"`
  - `timeseries_root=bench_dir/"length-of-stay"` → `bench_dir/"decompensation"`
- `CentralizedDataset.__getitem__`: `y_los` variable name → `y_decomp`
- Loss: `los_fn = nn.CrossEntropyLoss()` → `decomp_fn = nn.BCELoss()`
- Training loop: `los_fn(out["los"], y_los.long())` →
  `decomp_fn(out["decomp"].squeeze(-1), y_decomp.float())`
- Metrics: `los_metrics(...)` → `decomp_metrics(...)` (argmax → probability)
- Row dict keys: `val_los_*` → `val_decomp_*`
- Import: `los_metrics` → `decomp_metrics`

---

## Phase 6 — Figures

### 6a. `figures/negative_transfer_heatmap.py`
- `metrics` list: `"los_loss"` → `"decomp_loss"`
- `xticklabels`: `"LOS Loss"` → `"Decomp Loss"`

### 6b. `figures/feature_split_sensitivity.py`
- `TASKS`: `"los_loss"` → `"decomp_loss"`
- `LABELS`: `"LOS"` → `"Decomp"`

### 6c. `figures/plot_results_summary.py`
- Color map: `"ST-LOS"` → `"ST-Decomp"`, `"ihm_los"` → `"ihm_decomp"`
- Exp 1 LOS panel: `"ST-LOS"` → `"ST-Decomp"`, metric `"val_los_kappa"` → `"val_decomp_auroc"`
  axis label: `"LOS Cohen's κ"` → `"Decomp AUC-ROC"`
- Exp 2 panel: `"val_los_kappa"` → `"val_decomp_auroc"`

### 6d. `figures/plot_round1_results.py`
- Same as 6c — identical structure

---

## Phase 7 — Tests

### 7a. `tests/test_integration.py`
- Site B `"task": "los_bins"` → `"binary"`
- `make_batch` for Site B: `torch.randint(0, 10, ...).long()` → `torch.randint(0, 2, ...).float()`
- `labels["los"]` → `labels["decomp"]` in all test functions
- Shape assertion: `preds["los"].shape == (B, 10)` → `preds["decomp"].shape == (B, 1)`
- Print statements

### 7b. `tests/test_local_only.py`
- `test_site_b_has_los_metric()` → `test_site_b_has_decomp_metric()`
- Assert: `"kappa" in k or "mad" in k` → `"auroc" in k or "auprc" in k`

### 7c. `tests/test_centralized.py`
- Assert: `any(k.startswith("val_los_") for k in r)` →
  `any(k.startswith("val_decomp_") for k in r)`

---

## Key Invariants After Change

| What stays the same | What changes |
|---------------------|-------------|
| `site_B_labs.csv` filename | `y_los` column → `y_decomp` |
| Site B features (Glucose, pH, FiO2, CRR) | Task dir: length-of-stay → decompensation |
| `task_type="binary"` path in code | LOS-specific `"los_bins"` path unused |
| `BCELoss` (already used for IHM) | `CrossEntropyLoss` removed for Site B |
| 3-task structure (A=IHM, B=?, C=Pheno) | Middle task: LOS → Decomp |
| All embedding shapes (64-dim) | MMoE head: (64→10) → (64→1)+Sigmoid |
| `"pheno"` task unchanged | `"los"` key → `"decomp"` everywhere |

## Decisions Made

- **Per-stay decomp label**: `max(y_true)` over hourly rows — "ever decompensated during stay"
  This matches clinical intuition and is consistent with how LOS used deduplication.
- **Keep `"los_bins"` task_type in `TaskHead`**: leave as dead code, no removal needed.
- **Keep CustomBins import in dataset.py**: harmless dead code, saves 2 line changes.
- **Archive CSVs as `_los` suffix**: preserves provenance, new runs write fresh files.

## Errors Encountered
(none yet)
