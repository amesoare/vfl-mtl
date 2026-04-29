---
title: "Methodology — VFL-MTL: Vertical Federated Multi-Task Learning with Heterogeneous Task Sets for Clinical Prediction"
author: Amelia Soare
date: April 2026
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.4
header-includes:
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \captionsetup{font=small}
---

# 3. Methodology

## 3.1 Dataset

### 3.1.1 MIMIC-III

The dataset used in this study is MIMIC-III (Medical Information Mart for Intensive Care III, version 1.4) [@johnson2016], a publicly available clinical database containing de-identified health records from over 40,000 patients admitted to the intensive care unit (ICU) at Beth Israel Deaconess Medical Center in Boston between 2001 and 2012. Because all personally identifying information has been removed, the database is freely available to researchers worldwide through PhysioNet, provided they complete mandatory data privacy training and sign a data use agreement. All use of MIMIC-III in this study complies with the PhysioNet Credentialled Health Data Use Agreement 1.5.0.

For each ICU stay, MIMIC-III records measurements taken at the bedside approximately once per hour: vital signs such as heart rate, blood pressure, and oxygen saturation; laboratory test results such as blood glucose and pH; body measurements such as height and weight; and outcome information such as whether the patient died during the hospital stay or which diseases were diagnosed at discharge. In total, the database contains over 330 million individual recorded measurements, stored in a table called `CHARTEVENTS`. Each entry in this table specifies the patient, the measurement type, the time it was taken, and its value.

MIMIC-III is the standard benchmark dataset for ICU clinical prediction. The majority of methods this study compares against have been evaluated on this database, making it the natural choice for fair comparison.

### 3.1.2 Prediction Tasks

Raw MIMIC-III data were preprocessed using the YerevaNN mimic3-benchmarks pipeline [@harutyunyan2019], an open-source tool that converts the raw database tables into structured time-series files and defines four standard clinical prediction tasks. This study uses three of those four tasks:

- **In-hospital mortality (IHM):** predicting whether a patient will die during their hospital stay, based on the first 48 hours of ICU measurements. This is a binary task (died / did not die).

- **Decompensation:** predicting whether a patient's condition will suddenly and seriously worsen — for example, a sudden drop in blood pressure or oxygen levels that requires urgent intervention — at any point during the ICU stay. This is also a binary task (experienced decompensation / did not).

- **Phenotyping:** classifying each patient into one or more of 25 disease groups based on their diagnoses at hospital discharge. Each patient can belong to multiple groups simultaneously, making this a multi-label task.

The fourth task, length-of-stay prediction, is not included in this study. That task requires predicting which of 10 duration categories a stay falls into, which creates an imbalanced learning problem when combined with the two binary tasks above. It is noted as a candidate for future extension.

---

## 3.2 Data Preparation

### 3.2.1 Cohort Selection

The YerevaNN pipeline applies the following inclusion criteria, which are adopted without modification in this study. Patients must be at least 18 years old. Each stay must be at least 48 hours long, because the in-hospital mortality task requires the first 48 hours of measurements. Stays without a valid outcome label are excluded. After applying these filters, approximately 33,798 unique patients and 42,276 ICU stays remain.

The decompensation task in the YerevaNN pipeline originally produces one label per hour of a stay — for each hour, it records whether the patient was at risk of deteriorating in the next few hours. Since this study requires a single label per stay, we convert this into one binary value: if the patient was ever at risk of deteriorating at any point during the 48-hour window, the stay is labelled as positive for decompensation. Exploratory data analysis confirmed that the resulting positive rate of 10.1% is clinically plausible.

### 3.2.2 Simulating a Multi-Hospital Setup

This study simulates a scenario in which three hospitals collaborate to train a shared model while each keeping their own patient data private. In practice, different hospitals collect different types of measurements: one hospital might routinely record vital signs, a second might specialise in laboratory analyses, and a third might focus on general patient assessments. To reflect this, the available clinical variables are divided across three simulated hospital sites. Each site holds a different, non-overlapping set of features and is responsible for predicting a different clinical outcome.

The YerevaNN pipeline provides 17 clinical variables in total. Three of these — the sub-components of the Glasgow Coma Scale (GCS eye opening, GCS motor response, and GCS verbal response) — are excluded before splitting. The Glasgow Coma Scale measures a patient's level of consciousness by scoring eye, verbal, and motor responses; the total GCS score is simply the sum of the three sub-scores. If one site held the sub-scores and another held the total, either site could reconstruct the other's values by addition or subtraction, which would leak information across sites. To avoid this, only GCS total is retained and assigned to Site A.

The remaining 14 variables are distributed across the three simulated sites as follows:

| Site | Features | Assigned Task |
|------|----------|---------------|
| Site A | Heart rate, systolic BP, diastolic BP, temperature, SpO$_2$, respiratory rate, GCS total (7 variables) | In-hospital mortality |
| Site B | Glucose, pH, FiO$_2$, capillary refill rate (4 variables) | Decompensation |
| Site C | Height, weight, mean arterial pressure (3 variables) | Phenotyping |

: Feature assignment and prediction task for each simulated hospital site.

The assignment reflects clinical practice. Vital signs such as heart rate and blood pressure (Site A) are the most direct real-time indicators of survival risk. Laboratory and respiratory measurements (Site B) reflect the metabolic state of a patient over time and are natural predictors of acute deterioration. Anthropometric and haemodynamic summary values (Site C) provide broader physiological context relevant to disease classification.

The unequal number of features across sites (7, 4, 3) is intentional: it reflects the realistic situation where hospitals differ in how much data they collect. The effect of this imbalance on model performance is directly studied in Experiment 2 (Section 3.4).

Exploratory data analysis showed that measurements across sites are correlated. For example, systolic blood pressure (Site A) and mean arterial pressure (Site C) are related, and 61 non-zero cross-site feature pairs were found in the aligned training cohort. Site B laboratory values also carry predictive signal for the IHM outcome (|Pearson *r*| up to 0.12 for pH). These correlations are the empirical motivation for sharing information across sites during training.

> **[FIGURE 1 — Cross-site feature correlation diagram. Circos chord diagram showing the 61 non-zero cross-site feature pairs in the aligned training cohort. Chord width is proportional to the absolute Pearson correlation. Site A: blue, Site B: orange, Site C: green.]**

### 3.2.3 Identifying Shared Patients Across Sites

For the three simulated sites to train together, they must agree on which patients to include. In a real deployment, hospitals cannot share patient names or identifiers directly — this would violate patient privacy. The standard technique for solving this problem is Private Set Intersection (PSI): each site converts its patient identifiers into anonymous codes using a cryptographic hash function (SHA-256), then only the patients whose codes appear in all three sites are included. No site learns which specific patients the other sites excluded. This procedure is implemented in `data_prep/psi_alignment.py`.

After applying this procedure, the final aligned cohort contains 18,094 ICU stays, a reduction of approximately 46% from the per-site totals. This reduction is driven by Site A's 48-hour eligibility requirement: Sites B and C each contribute around 33,600 patients, but roughly half of these patients do not have a corresponding IHM-eligible stay in Site A. Exploratory data analysis confirmed that this reduction is not random — patients who remain in the cohort tend to be more severely ill, with a higher rate of decompensation (10.1% vs. 6.6% among excluded patients; χ² = 128.93, *p* < 0.0001) and higher prevalence across the 25 phenotype labels (18.8% vs. 13.1%). Results should therefore be interpreted with this context in mind: they apply to patients with ICU stays of at least 48 hours, which skews toward higher-acuity cases.

> **[FIGURE 2 — Patient flowchart. Tracks ICU stay counts from raw MIMIC-III (42,276) through inclusion criteria, per-site totals, and final aligned cohort (18,094 stays). CONSORT format.]**

The aligned cohort is split into 70% training, 15% validation, and 15% test sets. Label balance is checked across all three tasks: if any split deviates by more than 3 percentage points from the training set's positive rate, the split is reassigned to restore balance across all labels simultaneously (Sechidis et al., 2011, via `scikit-multilearn`).

### 3.2.4 Preprocessing

Each ICU stay is represented as a matrix of 48 rows (one per hour) and $d_k$ columns (one per feature at site $k$). Three preprocessing steps are applied in order.

**Step 1 — Removing implausible values.** ICU databases sometimes contain obvious data entry errors — for example, an oxygen saturation of 29,818% or a patient weight of 3,761,721 kg. Before any other processing, each feature is clipped to a plausible clinical range. The bounds are taken from the YerevaNN reference file `variable_ranges.csv` [@harutyunyan2019], with tighter limits applied where exploratory data analysis identified surviving artefacts. Clipping is applied per row before filling in missing values, so that a corrupted measurement does not propagate forward in time.

**Step 2 — Filling in missing values.** ICU measurements are not taken at perfectly regular intervals: some hours have no recorded value. Missing values are first filled forward in time (the last known value is carried forward) and then backward (the next available value is filled in). Any values that remain missing after this step are replaced with the feature's mean value, calculated on the training set. Exploratory data analysis showed that temporal coverage varies across sites: on average, stays had 47.8 observed hours at Site A, 38.8 hours at Site B, and 38.2 hours at Site C. Site B laboratory measurements are sparser than Site A vital signs, reflecting the lower frequency of blood draws compared to bedside monitoring.

> **[APPENDIX FIGURE A — Temporal coverage per site. Distribution of observed sequence lengths across 300 training stays per site.]**

**Step 3 — Input format.** The preprocessed 48-hour time series is used directly as model input. No further aggregation is applied, so the model can learn from hour-by-hour changes in a patient's condition rather than only from summary statistics.

---

## 3.3 Model Implementation

### 3.3.1 Baseline Models

Five comparison conditions are included to isolate the contribution of the proposed framework.

**Local-only (lower bound).** Each site trains a model using only its own measurements and its own prediction task, with no communication to any other site. This represents the current situation in a hospital that works entirely in isolation. The difference between this baseline and the proposed method quantifies the benefit of cross-site information sharing and multi-task learning. The encoder architecture is identical to the one used in the proposed method; the only difference is the absence of the shared server.

**Centralised oracle (upper bound).** A single model is trained on all 14 features from all three sites simultaneously, with no privacy restrictions. This represents what would be achievable if all hospitals could freely pool their raw data. The gap between the proposed method and this oracle shows the performance cost of the federated and privacy-preserving constraints.

**VFL-SingleTask variants (ST-IHM, ST-Decomp, ST-Pheno).** Three baselines that use the same federated infrastructure as the proposed method — three sites communicating through a shared server — but each site trains on its own task only. These baselines allow us to separate the contribution of multi-task learning from the contribution of the federated architecture: if the proposed method outperforms each ST variant on the corresponding task, the improvement is due specifically to training on multiple tasks simultaneously.

**MOCHA** [@smith2017] and **FMTLJD** [@huang2023] are federated multi-task learning methods designed for the horizontal setting, where all hospitals hold the same types of measurements for different patients. They cannot be directly applied to a setup where hospitals hold different features, so they are included as reported number comparisons rather than code re-runs. They establish whether the proposed method is competitive with existing federated MTL methods that operate without the vertical partitioning constraint.

**MARS-VFL** [@shen2025] is the most recent vertical federated learning benchmark, evaluated across 12 datasets under a single shared prediction task. Its results provide a reference point for what single-task VFL achieves.

### 3.3.2 Proposed Method

The proposed framework, VFL-MTL, has three main components: a local encoder at each site, a shared server that combines information across sites, and task-specific output layers. Training is organised into communication rounds in which sites and the server exchange information without sharing raw patient data.

**Per-site encoder.** Each site runs a local neural network — called an encoder — that processes only its own measurements. The encoder is a two-layer Long Short-Term Memory network (LSTM), a type of recurrent neural network that processes sequences step by step and is well suited to time-series data [@harutyunyan2019]. The LSTM reads the 48-hour measurement sequence and produces a single fixed-size summary vector of 64 numbers, called an embedding:

$$\mathbf{h}_k = \text{LayerNorm}\left(\mathbf{W}_k \cdot \text{LSTM}_k(\mathbf{x}_k)\right), \quad \mathbf{h}_k \in \mathbb{R}^{64}$$

where $\mathbf{x}_k \in \mathbb{R}^{48 \times d_k}$ is the measurement sequence at site $k$. This 64-number summary is the only information sent from each site to the server. Raw measurements never leave the site, which enforces the privacy guarantee: the server can only observe a compressed representation, not the original clinical data.

**Shared server.** The server receives the three 64-number embeddings from Sites A, B, and C and concatenates them into a single vector of 192 numbers. This combined vector is then processed by a Mixture-of-Experts (MMoE) module [@ma2018], a neural network design that can handle multiple prediction tasks at once. The MMoE consists of four shared expert sub-networks and one gating network per task. Each expert is a small two-layer network that transforms the 192-dimensional input. For each task, the gating network learns a set of weights over the four experts, producing a task-specific representation as a weighted combination of the expert outputs:

$$\mathbf{z}_t = \sum_{i=1}^{4} g_t^{(i)} \cdot f_i\!\left([\mathbf{h}_A; \mathbf{h}_B; \mathbf{h}_C]\right)$$

where $g_t^{(i)}$ is the weight assigned to expert $i$ for task $t$ (with all four weights summing to one), and $f_i$ is the output of expert $i$.

The key advantage of this design is that each task can learn to rely on different experts. If one task's learning signal conflicts with another's, the gating networks can route each task through the experts that are most useful for it, reducing the risk that one task's gradient harms another. A simpler alternative would be to pass the same combined representation to all tasks — this is tested in Ablation 1 (Section 3.4).

**Output layers.** Each task has its own output layer applied to its task-specific representation $\mathbf{z}_t$:

- **IHM (Site A):** one output neuron with sigmoid activation, producing a probability of in-hospital death. Trained with binary cross-entropy loss.
- **Decompensation (Site B):** one output neuron with sigmoid activation. Trained with binary cross-entropy loss on the stay-level label.
- **Phenotyping (Site C):** 25 output neurons with sigmoid activations, each producing an independent probability for one disease group. Trained with binary cross-entropy loss per label.

The total training loss is the sum of the three task losses with equal weights:

$$\mathcal{L} = \mathcal{L}_\text{ihm} + \mathcal{L}_\text{decomp} + \mathcal{L}_\text{pheno}$$

Equal weighting is a standard starting point in multi-task learning [@caruana1997].

**Training procedure.** Training is divided into communication rounds, where each round processes the full training set once. Within each batch, the following steps are executed:

1. Each site runs its encoder on its local feature batch and sends the resulting embedding to the server.
2. The server concatenates the three embeddings, runs them through the MMoE and output layers, and computes the combined training loss. It then calculates how much each part of the combined embedding contributed to the loss (the gradient) and sends each site back its corresponding portion.
3. Each site uses its received gradient to update its encoder.

This exchange allows each site's encoder to improve based on the prediction errors of all three tasks, even though each site only has access to its own measurements. Server parameters are updated after each batch using the Adam optimiser with a learning rate of $1\times10^{-3}$.

Every five rounds, the three site encoders are averaged together (weighted by training set size per site) and each site receives the updated averaged encoder. This synchronisation step, known as Federated Averaging (FedAvg) [@mcmahan2017], encourages the encoders to learn a representation that is useful across all sites, rather than each encoder specialising entirely to its local features.

---

## 3.4 Experimental Design

Four experiments evaluate the proposed framework. Each is run with three random seeds (42, 123, 7) and results are reported as mean ± standard deviation. All experiments are run on Snellius (SURF HPC, NVIDIA H100 GPU). Results are saved to CSV files.

**Experiment 1 — Does multi-task training help?** VFL-MTL (three tasks trained simultaneously) is compared against the three single-task baselines (ST-IHM, ST-Decomp, ST-Pheno). This tests whether training on multiple tasks at once improves each site's individual prediction performance.

**Experiment 2 — Does the number of features per site matter?** Three alternative feature splits are evaluated: (5, 6, 6), (3, 7, 7), and (7, 6, 4). The goal is to determine whether the default split (7, 4, 3) produces results that are specific to the number of features assigned to each site, or whether the framework is robust to different distributions.

**Experiment 3 — Do related tasks help each other more?** Two task combinations are compared: in-hospital mortality with decompensation (both predict acute deterioration and share correlated physiological drivers) versus in-hospital mortality with phenotyping (less clinically related). We measure how often multi-task training makes a task perform *worse* than single-task training — a phenomenon known as negative transfer.

**Experiment 4 — How does the framework scale?** VFL-MTL with two sites is compared against three sites, measuring the number of training rounds needed to converge, computing time per round, and prediction performance at convergence.

**Ablations.** Three additional experiments isolate the contribution of individual design choices. Ablation 1 replaces the MMoE server with a single shared network (no per-task expert weighting), testing whether the MMoE design is necessary. Ablation 2 replaces the privacy-preserving patient matching step with random pairing across sites, measuring the cost of proper patient alignment. Ablation 3 transmits raw features to the server instead of compressed embeddings, providing an upper bound on performance when the privacy constraint at the encoder is removed.

**Experimental configuration.**

| Hyperparameter | Value |
|----------------|-------|
| LSTM hidden dimension | 128 |
| LSTM layers | 2 |
| LSTM dropout | 0.1 |
| Embedding size | 64 |
| Number of experts (MMoE) | 4 |
| Expert network hidden size | 128 |
| Expert output size | 64 |
| Task loss weights | 1.0 each |
| Optimiser | Adam |
| Learning rate | 1$\times$10$^{-3}$ |
| Batch size | 64 |
| Training rounds | 50 |
| Encoder averaging frequency | Every 5 rounds |
| Observation window | 48 hours |
| Train / val / test split | 70% / 15% / 15% |
| Random seeds | 42, 123, 7 |
| Hardware | SURF Snellius, NVIDIA H100 GPU |
| Software | PyTorch 2.2, scikit-learn 1.4, pandas 2.2 |

: Complete configuration for all main experiments.

---

## 3.5 Evaluation Metrics

The research question asks whether a privacy-preserving federated framework can support heterogeneous clinical prediction tasks *while maintaining competitive performance compared to a centralised baseline*. This framing implies three distinct evaluative purposes, and the metrics below are organised accordingly: (1) quantifying the cost of federated constraints relative to the centralised upper bound; (2) quantifying the contribution of multi-task training within the federated setting; and (3) detecting cases where that contribution is negative.

Metric selection follows the YerevaNN benchmark specification [@harutyunyan2019], the authoritative reference for MIMIC-III clinical prediction evaluation, to ensure comparability with existing results in the literature.

### 3.5.1 Task-Specific Metrics

**In-hospital mortality and decompensation.** Both tasks are evaluated using AUC-ROC (area under the receiver operating characteristic curve). AUC-ROC measures the probability that a randomly drawn positive case is ranked higher than a randomly drawn negative case; it does not depend on any particular classification threshold, which makes it suitable for ranking-oriented evaluation in settings where the operating threshold is not fixed in advance. Accuracy is not reported: positive rates are 13% for IHM and 10.1% for decompensation, making accuracy uninformative — a trivial model that always predicts "negative" would achieve 87–90% accuracy without learning anything useful. AUC-ROC is insensitive to class imbalance in this respect.

AUC-PR (area under the precision-recall curve) is reported as a secondary metric. Whereas AUC-ROC measures separation across all thresholds weighted equally, AUC-PR concentrates on the region where the model must trade off false positives against false negatives at high positive predictive value — the operating regime that matters when a positive prediction triggers a clinical intervention. In highly imbalanced tasks, AUC-PR is more sensitive to improvements on the minority class than AUC-ROC, and therefore provides a stronger test of whether the proposed framework genuinely captures the signal distinguishing patients who die or decompensate [@harutyunyan2019].

**Phenotyping.** This task is evaluated using macro-averaged AUC-ROC: AUC-ROC is computed separately for each of the 25 disease groups and then averaged with equal weight per group:

$$\text{Macro-AUC} = \frac{1}{25} \sum_{k=1}^{25} \text{AUC-ROC}_k$$

Macro-averaging is required because disease prevalences range from under 4% to over 40%. Micro-averaging would give disproportionate weight to the most common conditions and could mask failures on clinically important but rare phenotypes. Equal weighting per label makes the metric sensitive to performance across the full diagnostic scope, not only the highest-frequency diagnoses. This is the standard evaluation choice for multi-label clinical phenotyping [@harutyunyan2019].

### 3.5.2 Framework Evaluation Metrics

The task-specific metrics above are used not as absolute quality indicators but as inputs to two comparative quantities that directly answer the research question.

**Performance gap to centralised oracle.** The centralised oracle trains on all features from all three sites simultaneously, with no privacy or federation constraints. It represents the theoretical upper bound achievable with the same model architecture when data pooling is unrestricted. The performance gap for task $t$ is:

$$\delta_t^{\text{fed}} = \text{Oracle}_t - \text{VFL-MTL}_t$$

A small gap ($\delta_t^{\text{fed}} < 0.02$ AUC points, following the convention of Harutyunyan et al., 2019) indicates that the privacy-preserving federated framework retains nearly all of the information available to the unconstrained system. A larger gap quantifies the information cost of keeping patient data local and communicating through compressed embeddings. This comparison is the primary operationalisation of "maintaining competitive performance" in the research question.

**Multi-task gain.** The multi-task gain $\Delta_t$ measures whether training on heterogeneous tasks simultaneously improves over training on each task in isolation within the same federated infrastructure. For each task $t$, the corresponding single-task baseline (ST-IHM, ST-Decomp, or ST-Pheno) uses the same federated setup but without the other tasks:

$$\Delta_t = \text{VFL-MTL}_t - \text{ST-}t$$

A positive $\Delta_t$ means that access to the other sites' learning signals — via the shared MMoE server — improved task $t$'s predictions beyond what it could achieve alone. A negative $\Delta_t$ means the opposite: the other tasks interfered with task $t$'s learning, a phenomenon called negative transfer [@caruana1997]. Multi-task gain is the primary operationalisation of the "heterogeneous task sets" contribution of the framework.

### 3.5.3 Negative Transfer

Negative transfer is declared for a task when $\Delta_t < 0$ across all three random seeds. Requiring consistency across all three seeds reduces the probability of a false declaration caused by variance in model initialisation, rather than a genuine structural conflict between tasks [@standley2020]. The rate and pattern of negative transfer is the focus of Experiment 3 (Section 3.4), which varies task relatedness to determine when multi-task training helps and when it does not.

---

## 3.6 Statistical Analysis

Comparisons are based on mean AUC-ROC across three random seeds (42, 123, 7). Three seeds is a deliberate choice to keep compute requirements feasible on Snellius; full confidence intervals are therefore not reported. Mean ± standard deviation are reported for all metrics across seeds. A difference larger than 0.02 AUC points is treated as practically meaningful, following the threshold used by Harutyunyan et al. (2019). Differences below this threshold are not interpreted as evidence of framework superiority or inferiority, since they fall within the expected noise of this evaluation regime.

---

*Prepared for Paper 1 — VFL-MTL. References marked [@bai2025] and [@standley2020] require DOI verification before submission.*
