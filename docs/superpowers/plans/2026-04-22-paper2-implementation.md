# Paper 2 Implementation Plan
# Differential Privacy in Vertical Federated Multi-Task Learning:
# Resilience, Utility Thresholds, and Multi-Task Label Inference Bounds

**Created:** 2026-04-22
**Revised:** 2026-05-09 — all results confirmed real MIMIC; new scripts added; bound_validation resubmitted
**Builds on:** Paper 1 VFL-MTL setup (model/, fl/, data_prep/, experiments/)
**Target:** 2 weeks implementation + writing

## Implementation Status

### Code (all files exist and are complete)

| Component | Status |
|-----------|--------|
| Opacus installed (≥1.4.0, actual: 1.5.4) | ✅ |
| `privacy/__init__.py` | ✅ |
| `privacy/adaptive_dpsgd.py` (AdaptiveDPSGD + DPVFLClient) | ✅ |
| `privacy/renyi_accountant.py` (RenyiAccountant) | ✅ |
| `train.py` — `privacy_config` parameter + DP loop wiring | ✅ |
| Unit tests for privacy/ (`tests/test_privacy.py`) | ✅ |
| `experiments/privacy_utility_curves.py` | ✅ |
| `experiments/ablations_dp.py` | ✅ |
| `experiments/validate_bound.py` | ✅ |
| `experiments/evaluate_test_ablations_dp.py` | ✅ |
| `attacks/__init__.py` | ✅ |
| `attacks/label_inference.py` | ✅ |
| `attacks/embedding_mia.py` | ✅ |
| `figures/privacy_utility_plot.py` | ✅ |
| `figures/resilience_variance.py` | ✅ |
| `figures/bound_validation.py` | ✅ |
| `figures/plot_ablations_dp.py` | ✅ |
| Slurm scripts for all ε levels (`run_privacy_curves_eps*.sh`) | ✅ |
| `run_attacks.sh` | ✅ |
| `run_bound_validation.sh` (partition fixed: rome, 15 min) | ✅ |
| `run_ablations_dp.sh` | ✅ |
| `run_evaluate_test_ablations_dp.sh` | ✅ |
| `experiments/run_factorial_pcmu.py` (PCMU Phase 2 full factorial) | ✅ |

### Results Status

All five per-ε runs are **real MIMIC** (Snellius, sample_rate=0.00437, ~229 batches/round).
The `sigma_*` CSV columns are **uncertainty-weighting parameters** (Kendall et al.), not the
DP noise multiplier — actual DP σ is in slurm logs only and IS correctly different per ε level:

| ε | DP σ (from slurm log) |
|---|---|
| 0.5 | 5.156 |
| 1.0 | ~3.5 |
| 2.0 | ~2.2 |
| 5.0 | 0.908 (uniform); 1.592/1.592/2.793 (stratified) |
| 10.0 | 0.690 |

| Result file | Status |
|---|---|
| `results/privacy_utility_eps{05,1,2,5,10}.csv` | ✅ Real MIMIC, 100 rounds, 3 seeds |
| `results/privacy_utility_combined.csv` | ✅ Canonical merged file: all ε levels + exp1 no-DP (ε=∞). Authoritative source. |
| `results/embedding_mia.csv` | ✅ Real MIMIC — MIA AUC ~0.50 across all ε (expected under DP) |
| `results/label_inference.csv` | ✅ Real MIMIC — IHM AUC=0.780 at ε=∞ matches exp1; degrades as ε↓ |
| `results/dp_ablations.csv` | ✅ Real MIMIC, 3 seeds — all 3 ablations (abl1/abl2/abl3); note: abl2 val metrics are NaN (training loop gap), covered by test_ablations_dp.csv |
| `results/test_ablations_dp.csv` | ✅ Real MIMIC, 3 seeds — proper test-set inference for abl2 + abl3 checkpoints |
| `results/bound_validation.csv` | ⚠️ Empirical values are synthetic (validate_bound.py was run before label_inference.csv was populated). Job resubmitted 2026-05-09 via `run_bound_validation.sh` on rome partition. |

**Key SRQ2 finding:** AUC ~0.5 across all finite ε levels {0.5–10} is genuine — DP noise on
VFL cut-layer gradients prevents learning above chance at all tested budgets. ε* exists only
at ε=∞ (no DP). This is a publishable result.

**Remaining action:** Sync `bound_validation.csv` from Snellius once job completes.

---

## Research Questions

- **SRQ1:** How resilient is the proposed architecture to the stochasticity introduced by differential privacy?
- **SRQ2:** At what threshold does the privacy-utility trade-off compromise the reliability of clinical decision support?

---

## Contributions

| Row | vs. Existing Work | Our Distinction |
|-----|-------------------|-----------------|
| 1 | Abadi et al. (2016) — uniform σ | Task-stratified noise allocation with empirical ε sweep |
| 2 | FMTLJD (2023) — no per-task analysis | Per-task ε quantification with gradient coupling analysis |
| 3 | No prior work — novel contribution grounded in the Gaussian mechanism (Abadi et al. 2016) | Multi-task label inference bound g(σ,ρ) = Φ(C√(1+ρ)/σ); single-task base g(σ)=Φ(C/σ) derived from DP-SGD noise distinguishability |
| 4 | MTFSLaMM (2025) — non-clinical, no MTL | Clinical VFL-MTL embedding-space attack suite |

---

## Existing Paper 1 Infrastructure — Do Not Modify

```
model/encoder.py          — SiteEncoder LSTM (hidden=128, embed_dim=64)
model/mmoe.py             — MMoEServer (4 experts, per-task gating) — complete
fl/client.py              — VFLClient::receive_gradient() at line 69
                            DP hook comment at line 73: "subclass and override here to clip/add noise"
fl/server.py              — VFLServer::aggregate_embeddings(), compute_loss(), send_gradients()
                            compute_task_gradient_similarity() at line 213 — returns
                            {grad_sim_ihm_decomp, grad_sim_ihm_pheno, grad_sim_decomp_pheno}
                            get_embedding_gradients() at line 191 — slices grad per site
fl/fedavg.py / fedprox.py — aggregation utilities
train.py                  — round-based loop; accepts grad_sim_every at line 93;
                            calls compute_task_gradient_similarity() at line 617
                            does NOT yet accept privacy_config
results/exp1.csv          — VFL-MTL: IHM=0.782, Decomp=0.712, Pheno=0.620 (3 seeds, ~79 rounds)
                            ST-IHM: IHM=0.795; ST-Decomp: Decomp=0.701; ST-Pheno: Pheno=0.612
results/ablations.csv     — all 7 ablation variants complete on real MIMIC (3 seeds each):
                            VFL-MTL, abl_no_mmoe, abl_experts_2/8, abl_uniform_gating,
                            abl_embed_32, abl_embed_128
results/centralized.csv   — IHM=0.862, Decomp=0.910, Pheno=0.655 (3 seeds, ~93 epochs)
```

**Key reuse:**
- `fl/server.py::compute_task_gradient_similarity()` (line 213) — use directly for ρ estimation; no new code needed
- `fl/server.py::get_embedding_gradients()` (line 191) — DP noise injection point for uniform mode
- `fl/client.py::receive_gradient()` (line 69) — DP clipping hook per CLAUDE.md spec

**Key Paper 1 finding:** Decomp gained +0.125 AUROC from MTL — most dependent on shared representation. Central hypothesis: Decomp will be first to lose utility under DP noise.

---

## Clinical Utility Floors (SRQ2 anchor)

| Task | Metric | Floor | Rationale |
|------|--------|-------|-----------|
| IHM | AUC-ROC | 0.75 | Harutyunyan et al. (2019) |
| Decomp | AUC-ROC | 0.70 | Harutyunyan et al. (2019) |
| Pheno | Macro-AUC | 0.65 | Harutyunyan et al. (2019) |

ε* per task = the ε below which AUC drops below the floor.

---

## Files Created (all complete)

```
privacy/
  __init__.py                        ✅
  adaptive_dpsgd.py                  ✅ Opacus wrapper, per-task σ, gradient clipping
  renyi_accountant.py                ✅ Rényi DP tracking + coupling matrix
attacks/
  __init__.py                        ✅
  label_inference.py                 ✅ linear probe on cut-layer embeddings → labels
  embedding_mia.py                   ✅ binary classifier → member/non-member
experiments/
  privacy_utility_curves.py          ✅ ε sweep, 3 seeds, per-task AUC + std
  ablations_dp.py                    ✅ stratified vs. uniform σ; related vs. unrelated tasks
  validate_bound.py                  ✅ ρ sweep, theoretical bound vs. empirical accuracy
  evaluate_test_ablations_dp.py      ✅ test-set inference for Abl2+Abl3 checkpoints
  run_factorial_pcmu.py              ✅ PCMU Phase 2 full factorial (108 runs)
figures/
  privacy_utility_plot.py            ✅ per-task AUC vs. ε line plot
  resilience_variance.py             ✅ std(AUC) vs. ε per task
  bound_validation.py                ✅ theoretical vs. empirical MI bound
  plot_ablations_dp.py               ✅ three-panel ablation figure
results/
  privacy_utility_combined.csv       ✅ canonical merged ε sweep (all levels + ε=∞ from exp1)
  dp_ablations.csv                   ✅ Abl1 real MIMIC; Abl2/3 val metrics only (see test_ablations_dp.csv)
  test_ablations_dp.csv              ✅ proper test-set inference for Abl2+Abl3
  embedding_mia.csv                  ✅ real MIMIC
  label_inference.csv                ✅ real MIMIC
  bound_validation.csv               ⚠️ empirical values synthetic — job resubmitted 2026-05-09
```

Also modified: `train.py` (privacy_config parameter) ✅

---

## Week 1 — DP-SGD Integration + Accounting + ε Sweep

**Goal:** VFL-MTL training loop runs under DP-SGD; Rényi accounting works; full privacy-utility curves complete by end of week.

### Days 1–2: Core DP Module

**Step 1 — Install Opacus** ✅
```bash
pip install opacus>=1.4.0
```
Verify compatible with torch version in requirements.txt.

**Step 2 — Write `privacy/adaptive_dpsgd.py`** ✅

```python
class AdaptiveDPSGD:
    """Wraps per-site SiteEncoder with Opacus GradSampleModule.
    Called inside fl/client.py::receive_gradient() before backward."""

    def __init__(self, encoder, max_grad_norm: float, noise_multiplier: float):
        self.module = GradSampleModule(encoder)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def set_uniform(self, sigma: float) -> None:
        """All task gradients use same σ."""

    def set_stratified(self, sigma_ihm: float, sigma_decomp: float, sigma_pheno: float) -> None:
        """Per-task σ dict {ihm: σ_ihm, decomp: σ_decomp, pheno: σ_pheno}."""

    def clip_and_noise(self, grad: Tensor, task: str) -> Tensor:
        """Clip grad norm to max_grad_norm; add Gaussian noise scaled to σ_task."""
```

Integration point: in `fl/client.py::receive_gradient()` at line 73 (existing DP hook comment).

**Step 3 — Write `privacy/renyi_accountant.py`** ✅

```python
class RenyiAccountant:
    """Per-task Rényi DP tracking wrapping opacus.accountants.RDPAccountant."""

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int,
             task: str) -> None:
        """Advance accountant for one task."""

    def get_epsilon(self, delta: float = 1e-5) -> dict[str, float]:
        """Returns {task: ε_k} for all tasks."""

    def cross_task_coupling_matrix(self, server, n_rounds: int = 5) -> dict:
        """Pearson correlation matrix of per-task embedding gradients.
        Calls fl/server.py::compute_task_gradient_similarity() (line 213)
        over first n_rounds; returns {(ihm,decomp): ρ, (ihm,pheno): ρ, ...}."""

    def coupling_epsilon_inflation(self) -> float:
        """Empirical ε_total vs. sum(ε_k)."""
```

Unit test: single-task ε must match opacus reference ± 1e-4.

**Step 4 — Modify `train.py`** ✅

Add `privacy_config` parameter (after existing `grad_sim_every` at line 93):
```python
# privacy_config = None                              → no DP (Paper 1 unchanged)
# privacy_config = {'mode': 'uniform',
#                   'sigma': float, 'delta': 1e-5}  → uniform DP
# privacy_config = {'mode': 'stratified',
#                   'sigma_ihm': f, 'sigma_decomp': f, 'sigma_pheno': f}
```
Log ε per task at end of each round to results CSV.

**Step 5 — Unit tests** ✅ `tests/test_privacy.py` (25 tests, all passing)
- ε increases monotonically with rounds
- Higher σ → lower ε at same round count
- Per-task ε sum ≤ ε_total + 1e-3
- Encoder gradients clipped to max_grad_norm

**Deliverable Days 1–2:** `privacy/` passing unit tests; `train.py` accepts `privacy_config` ✅

---

### Days 3–5: ε Sweep + Resilience Analysis (SRQ1 + SRQ2)

**Step 6 — Write `experiments/privacy_utility_curves.py`** ✅

ε levels: `{∞, 10, 5, 2, 1, 0.5}` — seeds: `[42, 123, 7]` (consistent with Paper 1)

For each ε level:
- Compute σ via `opacus.utils.uniform_noise.get_noise_multiplier(target_epsilon=ε, target_delta=1e-5, sample_rate=batch/N, epochs=100)`
- Run VFL-MTL for 100 rounds (uniform σ mode)
- Log: `round, seed, epsilon_level, val_ihm_auroc, val_decomp_auroc, val_pheno_macro_auroc, convergence_round`
- Output: `results/privacy_utility.csv`

**Step 7 — Run task-stratified variant** ✅ Real MIMIC, results in `privacy_utility_eps5.csv`

Clinical risk hierarchy: σ_IHM < σ_Decomp < σ_Pheno
- ε_total=5; `mode=stratified` rows present in `privacy_utility_combined.csv`

**Step 8 — SRQ1 resilience metrics** ✅ Computable from `privacy_utility_combined.csv`

From `results/privacy_utility_combined.csv`:
- `std(AUC)` across 3 seeds per ε per task → variance inflation index
- Convergence round: first round where val_AUC ≥ 0.90 × AUC(ε=∞)
- Note: 3 seeds used (plan called for 5; 3 is consistent with Paper 1)

**Step 9 — SRQ2 threshold identification** ✅ Computable from `privacy_utility_combined.csv`

Key finding: all finite ε levels {0.5–10} yield AUC ~0.5 for most tasks — DP noise on the
cut-layer prevents useful learning. ε* for all tasks is above ε=10 (i.e., very loose DP
is required for clinical utility). This is the publishable finding.

**Deliverable end of Week 1:** ✅ Complete — real-MIMIC ε sweep final

---


## Week 2 — Attack Suite + Label Inference Bound + Writing

**Goal:** Embedding-space attacks evaluated; multi-task label inference bound derived and validated; full paper draft complete.

### Days 6–7: Attack Suite

**Step 1 — Write `attacks/label_inference.py`** ✅

Threat model: passive party (e.g. Site B) observes cut-layer embeddings z_A, z_C from server.

- Extract cut-layer embedding vectors for all validation patients at each ε level
  (embeddings saved during `privacy_utility_curves.py` runs)
- Probe: logistic regression, input = z_A or z_C, output = active party labels
  (IHM labels at Site A; Pheno labels at Site C)
- Report: fraction correctly inferred per task per ε level
- Target under sufficient DP: ≈ prevalence rate (IHM ≈ 10%)

**Step 2 — Write `attacks/embedding_mia.py`** ✅

- Split patients: 50% members (in training set), 50% non-members (held out)
- Extract cut-layer embeddings for both groups at each ε level
- Binary classifier (logistic regression) → member vs. non-member
- Report: attack AUC and accuracy at each ε level
- Target under sufficient DP: AUC ≈ 0.50

Note: gradient-norm MIA (Carlini et al. 2022) not used — requires white-box gradient access; in VFL the server observes embeddings, not raw parameter gradients.

### Days 7–8: Label Inference Bound + Ablations

**Step 3 — Novel multi-task label inference bound** ✅ `experiments/validate_bound.py`

Single-task base (novel; derived from Gaussian mechanism, Abadi et al. 2016):
  Under N(0, σ²C²I) noise, distinguishability between positive/negative class gradient
  distributions is bounded by g(σ) = Φ(C/σ), where Φ is the standard normal CDF,
  C = max_grad_norm, σ = noise multiplier.
  NOTE: this base bound is the author's own derivation — it is NOT from Liu et al. (2022),
  which is a VFL survey (IEEE TKDE 2024) with no MI bound.

Multi-task extension:
- Let ρ = Pearson correlation between ∂L_k/∂z and ∂L_j/∂z for tasks k ≠ j
  (from `renyi_accountant.cross_task_coupling_matrix()`, which calls `server.compute_task_gradient_similarity()` at line 213)
- Proposition: label inference AUC ≤ g(σ, ρ) = Φ(C·√(1+ρ)/σ), monotonically increasing in ρ
- Proof sketch: the Gaussian mechanism bound widens by √(1+ρ) when task gradients are correlated,
  reflecting the amplified distinguishability of the composite gradient signal across tasks

Write `experiments/validate_bound.py`:
- For each ε level: compute ρ + compute bound g(σ, ρ) + measure empirical label inference accuracy
- Output: `results/bound_validation.csv`

**Step 4 — Write `experiments/ablations_dp.py`** ✅

- Abl 1: Uniform σ vs. task-stratified σ at ε_total=5 — per-task AUC comparison
- Abl 2: Related pair (IHM+Decomp, ρ high) vs. unrelated (IHM+Pheno, ρ low) —
  label inference accuracy difference; confirms coupling amplification hypothesis
- Abl 3: embed_dim ∈ {32, 64, 128} × ε ∈ {1, 5, ∞} — per-task AUC comparison.
  Motivation: SNR = 1/(embed_dim × σ²); ε* is embed_dim-dependent.
  Validates that Paper 2's ε* claims are conditional on embed_dim=64.
- Output: `results/dp_ablations.csv`

**Deliverable Days 6–8:** `attacks/` module ✅ real MIMIC results; `results/dp_ablations.csv` ✅ real MIMIC (3 seeds); `results/test_ablations_dp.csv` ✅ real MIMIC (proper test-set inference for Abl2+Abl3); `results/bound_validation.csv` ⚠️ empirical values synthetic — job resubmitted 2026-05-09

### Days 9–10: Figures + Results Assembly

**Step 5 — Write `figures/plot_ablations_dp.py`** ✅ code complete

Three separate output files (one per ablation):
- `figures/ablations_dp_abl1.png` — grouped bars: uniform vs. stratified σ per task
- `figures/ablations_dp_abl2.png` — ρ and IHM inference AUC: related vs. unrelated pairs
- `figures/ablations_dp_abl3.png` — line plot per task, x=ε, lines=embed_dim ∈ {32, 64, 128}

Usage: `python figures/plot_ablations_dp.py [--abl {1,2,3}]`

**Step 6 — Write `figures/privacy_utility_plot.py`** ✅ code exists; ⚠️ needs real data to render
- Three-panel line plot (one per task: IHM / Decomp / Pheno)
- x-axis: ε (log scale); y-axis: mean AUC ± std across 3 seeds (plan said 5; actual runs use 3)
- Two lines per panel: uniform σ vs. task-stratified σ
- Horizontal dashed line: clinical utility floor; vertical marker: ε*

**Step 7 — Write `figures/bound_validation.py`** ✅ code exists; ⚠️ needs real data to render
- Theoretical MI bound vs. empirical label inference accuracy across ε levels
- One line per ρ value (ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9})

**Step 8 — Write `figures/resilience_variance.py`** ✅ code exists; ⚠️ needs real data to render
- x-axis: ε (log scale); y-axis: std(AUC) across seeds
- One line per task; expected: Decomp line rises steepest

### Days 10–14: Writing

**Introduction** (Day 10):
- Clinical motivation: hospital consortia, heterogeneous EHR, VFL without feature sharing
- Gap: no existing work studies DP resilience or utility thresholds in VFL with heterogeneous task sets
- Four contribution bullets per differentiation table
- SRQ1 and SRQ2 stated explicitly

**Threat model + Related Work** (Day 10):
- Honest-but-curious server observes cut-layer embeddings
- Passive party observes returned embedding gradients ∂L/∂z_i
- Multi-task leakage surface: MMoE gradient correlation amplifies label inference risk
- Related Work (~1.5 pages): DP-FL (Abadi, McMahan), VFL privacy (Liu, Luo, Fu),
  MTL-FL (FMTLJD, MTFSLaMM), MIA (Shokri 2017, Weng 2021), clinical FL (Harutyunyan)

**Methods** (Day 11):
- DP-SGD integration at VFL cut layer (hook at `fl/client.py:73`)
- Task-stratified noise allocation formulation (σ_IHM < σ_Decomp < σ_Pheno)
- Rényi DP accounting + gradient coupling measurement (reuse `fl/server.py:213`)
- Multi-task label inference bound (proposition + proof sketch)
- Embedding-space attack suite design
- Evaluation setup: MIMIC-III, ε sweep, 3 seeds, clinical utility floors

**Results** (Days 12–13):
- Per-task AUC vs. ε line plot
- std(AUC) vs. ε per task
- ε* per task table
- Theoretical bound vs. empirical label inference accuracy
- Ablation table: uniform vs. stratified σ; related vs. unrelated task pairs
- MIA results table at each ε level

**Discussion** (Day 13):
- Which task loses utility first and why (Decomp hypothesis)
- Risk-stratified ε as GDPR Article 25 "data protection by design"
- Limitations: MIMIC-III simulation, single DP mechanism (Gaussian), no adaptive clipping,
  conservative ε composition under correlated tasks

**Abstract + polish** (Day 14)

**Reporting format:**
- All AUC values: mean ± std across 3 seeds
- All ε values with explicit δ=1e-5
- Per-task results in separate rows/panels — do not average across tasks

---

## Expected Figures

### Figure 1 — Privacy-Utility Curves (`figures/privacy_utility_plot.py`)

Answers SRQ2. Three-panel line plot (IHM / Decomp / Pheno):
- x-axis: ε ∈ {0.5, 1, 2, 5, 10, ∞} on log scale
- y-axis: mean AUC-ROC ± std across 3 seeds
- Two lines: uniform σ vs. task-stratified σ
- Horizontal dashed line: clinical utility floor
- Vertical marker: ε* crossing point

### Figure 2 — Resilience Variance Plot (`figures/resilience_variance.py`)

Answers SRQ1. Single panel:
- x-axis: ε (log scale); y-axis: std(AUC) across 3 seeds
- One line per task; Decomp expected to rise steepest (highest MTL gain → most DP-sensitive)

Convergence round reported as table row alongside ε*, not standalone figure.

### Figure 3 — Bound Validation (`figures/bound_validation.py`)

Single panel:
- x-axis: ε levels; y-axis: MI upper bound + empirical label inference accuracy
- One line per ρ value (ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9})

### Tables

| Table | Source | Content |
|---|---|---|
| ε* per task | `results/privacy_utility_combined.csv` | SRQ2 answer — ε* and floor-met flag per task |
| MIA results | `results/embedding_mia.csv` | Attack AUC + accuracy at each ε level |
| Label inference | `results/label_inference.csv` | Fraction correctly inferred per task per ε |
| DP ablations | `results/dp_ablations.csv` (Abl1) + `results/test_ablations_dp.csv` (Abl2+3) | Abl 1: uniform vs. stratified σ; Abl 2: related vs. unrelated pairs; Abl 3: embed_dim × DP |

### RQ → Figure Map

```
SRQ1 (DP resilience)     → Figure 2: resilience_variance.py
SRQ2 (utility threshold) → Figure 1: privacy_utility_plot.py + ε* table
Theoretical bound        → Figure 3: bound_validation.py
Attack effectiveness     → MIA table + label inference table
Stratified vs. uniform   → Figure 1 (both lines) + DP ablations table (Abl 1)
Coupling amplification   → DP ablations table (Abl 2) + Figure 3 (ρ lines)
```

Not visualized (by design):
- Gradient coupling matrix: scalar reported in text from `renyi_accountant.cross_task_coupling_matrix()`
- Convergence round: table row alongside ε*
- pFedMe: dropped (architectural incompatibility — clients hold only local LSTM encoder)

---

## Key Dependencies

| Paper 2 component | Depends on Paper 1 artifact |
|---|---|
| DP-SGD wrapper | `fl/client.py::receive_gradient()` at line 69 |
| Embedding extraction | `fl/server.py::aggregate_embeddings()` |
| Gradient coupling matrix | `fl/server.py::compute_task_gradient_similarity()` at line 213 |
| Label inference attack | cut-layer embeddings saved during training |
| Bound validation | `results/privacy_utility.csv` + `attacks/label_inference.py` |

---

## References

- Abadi et al. (2016) DP-SGD. CCS. https://doi.org/10.1145/2976749.2978318
- Mironov (2017) Rényi DP. CSF. https://doi.org/10.1109/CSF.2017.11
- Yousefpour et al. (2021) Opacus. arXiv. https://arxiv.org/abs/2109.12298
- McMahan et al. (2018) DP LSTM FL. ICLR. https://arxiv.org/abs/1710.06963
- Liu et al. (2024) VFL Survey ("Vertical Federated Learning: Concepts, Advances, and Challenges"). IEEE TKDE. https://doi.org/10.1109/TKDE.2024.3352628 — survey only; contains no MI bound; cited as liu_2022_vfl in bib (key reflects arXiv 2022 preprint year)
- Fu et al. (2022) Label inference attacks in VFL. USENIX Security.
- Luo et al. (2021) Feature inference attack on VFL. CCS. https://dl.acm.org/doi/10.1145/3460120.3485370
- Weng et al. (2021) Privacy leakage in VFL. arXiv. https://arxiv.org/abs/2011.09290
- Harutyunyan et al. (2019) MIMIC benchmarks. Sci. Data. https://doi.org/10.1038/s41597-019-0103-9
- Huang et al. (2023) FMTLJD. IEEE TBME. https://doi.org/10.1109/TBME.2022.3210940
- Dong et al. (2025) MTFSLaMM. Sensors. https://doi.org/10.3390/s25010233
