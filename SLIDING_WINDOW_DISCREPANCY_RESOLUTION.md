# Forensic Resolution: Sliding-Window Length and Feature-Count Discrepancies

**Investigation:** Second Comprehensive Scientific Audit  
**Artifact:** Research Paper — *"Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing"*  
**Repository:** `cloud-failure-detection`  
**Date:** 2026-11-11  
**Status:** Completed & Empirically Reconciled  

---

## Executive Summary & Final Audit Verdict

### Final Verdict: **NO — The final research paper does NOT accurately describe the exact sliding-window and feature configuration used to produce the reported experimental results.**

### Forensic Summary:
1. **Window-Length Discrepancy ($W=15$ vs. $L=5$):**
   - The paper draft explicitly claims a window length of **$W=15$ (75 minutes)** in the Problem Formulation and Table III footnote.
   - The actual final model checkpoints (`models/checkpoints_phase7d/`), the Candidate A validation data (`data/processed/phase7c/X_val.npy`), the final evaluation metrics, and the dashboard inference pipeline all operate strictly on **$L=5$ (25 minutes)**.
   - The $W=15$ configuration was used only in early exploratory phases (Phases 2–5). In Phase 6.2, an independent empirical audit proved that $L=15$ caused severe sequence dropout (over 60% of cloud tasks terminate in under 75 minutes), so the protocol was officially amended to $L=5$. The paper draft was never updated to reflect this critical protocol evolution.
2. **Feature-Count Discrepancy ($d=42$ vs. $d=31$):**
   - The paper text explicitly claims a feature dimension of **$d=42$** in Section IV.
   - All deployed final models, the Candidate A scaler, and the live dashboard operate strictly on **31 canonical features** (or 63 dimensions for the representation hybrid).
   - Forensic tracing reveals that the number **42** originated from **Phase 7K** ([`scripts/run_phase7k_ablation.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase7k_ablation.py)), where model **$M_4$** combined the 31 baseline features with 11 additional candidate features (peak telemetry, scheduler metadata, and preemption history). Phase 7K concluded that $M_4$ provided no operational benefit and failed all acceptance gates (`NO_OPERATIONAL_WINNER`), so the project permanently retained the 31-feature baseline ($M_0$). The paper draft erroneously copied the exploratory $M_4$ feature dimension ($d=42$) into the main text.
3. **Metric & Dataset Divergence:**
   - The paper reports an LSTM Macro-F1 of **0.87** on the Google Borg 2019 trace with zero-shot transfer to Alibaba and Azure.
   - In the actual codebase, Borg 2019 was labeled `DATA_ACCESS_BLOCKED` in Phase 7Y; the actual local models were trained and evaluated on the **Google Borg 2011 trace**, where real class imbalance ($98:1:1$) yielded honest Macro-F1 scores of **$0.3068 - 0.4479$**, decisively exposing that all models fail operational false-alarm ceilings.

---

## 1. Resolution of the Window-Length Discrepancy

### A. Core Questions & Concrete Answers

| Question | Verified Finding | Evidence Source |
| :--- | :--- | :--- |
| **Window length used to train final checkpoints?** | **$L = 5$ (25 minutes)** | `models/checkpoints_phase7d/best_transformer_checkpoint_phase7d.pt` has `max_seq_len=5`. `scripts/train_and_evaluate_phase7d.py` trains on shape `(3278, 5, 31)`. |
| **Window length used to generate Candidate A validation data?** | **$L = 5$ (25 minutes)** | `data/processed/phase7c/X_val.npy` has shape **`(657, 5, 31)`**. |
| **Window length used for reported Macro-F1 and operational metrics?** | **$L = 5$ (25 minutes)** | Phase 7D validation metrics (`results/phase7d/validation_metrics.json`) and Phase 8 audit report (`results/phase8_research_integrity/`). |
| **Window length used by dashboard inference pipeline?** | **$L = 5$ (25 minutes)** | `dashboard/utils/inference_pipeline.py` lines 86–105 slices sliding windows of size 5. |
| **Does the paper describe the final protocol?** | **NO** | The paper specifies $W=15$ (Table III note: *"Validation-selected $W=15$"*), which reflects the superseded Phase 2–5 protocol. |

---

### B. Phase-Wise Window-Length Evolution

The repository contains an exhaustive audit trail explaining why the sequence length changed:

| Project Phase | Focus & Objective | Window Length ($L$) | Time Span | Key Artifact / Script | Empirical Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | Data Pipeline Ingestion & Initial Windows | **$L = 15$** | $75\text{ min}$ | [`scripts/run_phase2_pipeline.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase2_pipeline.py), [`data/processed/X_train.npy`](file:///C:/AntiGravity%20Files/cloud-failure-detection/data/processed/X_train.npy) | Shape `(4798, 15, 31)`. Strict 15-step requirement established. |
| **Phase 3** | Baselines & Initial Sequential LSTM | **$L = 15$** | $75\text{ min}$ | [`scripts/phase3b_lstm.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/phase3b_lstm.py), `best_lstm_checkpoint.pt` | Evaluates 15-step inputs. |
| **Phase 4** | Transformer & Representation Hybrid | **$L = 15$** | $75\text{ min}$ | [`scripts/phase4_transformer_hybrid.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/phase4_transformer_hybrid.py) | Checkpoint `exp3_lstm_2l_h32_balanced.pt` trained on $L=15$. |
| **Phase 5** | Confirmatory Holdout Protocol 1.0 | **$L = 15$** | $75\text{ min}$ | [`scripts/extract_confirmatory_data.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/extract_confirmatory_data.py) | Generates `X_confirmatory.npy` under $L=15$. |
| **Phase 6.1–6.2** | Feasibility Audit & Protocol Amendment | **$L = 15 \to 5$** | **Pivot: $75 \to 25\text{ min}$** | [`scripts/audit_phase6_feasibility.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/audit_phase6_feasibility.py), [`scripts/build_protocol_l5.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/build_protocol_l5.py) | **CRITICAL DISCOVERY:** Over 60% of tasks in Google Borg terminate in $< 75$ min. Demanding $L=15$ caused sequence dropout, leaving only 33 failure episodes. Protocol Amendment 2.1 ratified **$L=5$**, recovering 141 episodes. |
| **Phase 6.3–6.4** | L=5 Model Retraining & Confirmatory Run | **$L = 5$** | $25\text{ min}$ | [`scripts/train_phase6_3_l5_models.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/train_phase6_3_l5_models.py), [`scripts/extract_confirmatory_data_l5.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/extract_confirmatory_data_l5.py) | First models trained on $L=5$. Confirmatory holdout evaluated (Random Forest Macro-F1 0.6111 vs LSTM 0.4425). |
| **Phase 7A–7D** | Candidate A Development & Final Checkpoints | **$L = 5$** | $25\text{ min}$ | [`scripts/reconstruct_phase7c_raw_features.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/reconstruct_phase7c_raw_features.py), [`scripts/train_and_evaluate_phase7d.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/train_and_evaluate_phase7d.py) | Input tensors sliced to `(N, 5, 31)`. Deployed checkpoints saved in `models/checkpoints_phase7d/`. |
| **Phase 7K** | Controlled Feature Ablation Study | **$L = 5$** | $25\text{ min}$ | [`scripts/run_phase7k_ablation.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase7k_ablation.py) | Evaluated $M_0$ (31 features) against $M_4$ (42 features) at $L=5$. |
| **Phase 8** | Research Integrity & Dataset Audit | **$L = 5$** | $25\text{ min}$ | [`scripts/run_phase8_research_integrity_audit.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase8_research_integrity_audit.py) | Cryptographic freeze on Candidate A validation arrays `(657, 5, 31)`. |
| **Production UI** | Dashboard & Live CPU Inference | **$L = 5$** | $25\text{ min}$ | [`dashboard/utils/inference_pipeline.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/inference_pipeline.py) | Enforces 5-step sliding window formatting. |

---

## 2. Resolution of the Feature-Count Discrepancy

### A. The Origin of "42 Features"
In Section IV of the research paper, the text states:
> *"Causal rolling features (moving average, rate of change, slope) are computed over an $r = 6$ step (30 min) trailing period, yielding a feature dimension $d = 42$."*

**Forensic Discovery:** The number **42** is NOT an arbitrary typographical error; it represents the **exact feature dimension of Model $M_4$ from the Phase 7K ablation experiment** ([`scripts/run_phase7k_ablation.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase7k_ablation.py#L208-L228)).

In Phase 7K, the researchers tested whether expanding the 31-feature telemetry baseline by adding 11 external features could overcome the sensitivity-specificity barrier:
- **Baseline $M_0$:** 31 features (pure telemetry).
- **Group J1 (5 features):** Peak usage & headroom (`max_cpu_rate`, `cpu_burstiness`, `max_memory`, `memory_headroom`, `total_page_cache`).
- **Group J2 (4 features):** Scheduler metadata (`priority`, `scheduling_class`, `req_cpu`, `req_ram`).
- **Group J3 (2 features):** Preemption history (`eviction_count_prior`, `time_since_last_schedule`).
- **Model $M_4$:** $31 + 5 + 4 + 2 =$ **42 features**.

**The Phase 7K Audit Finding:** Phase 7K evaluated 30 models across 6,030 threshold points and proved that Model $M_4$ ($d=42$) did **not** pass the pre-registered operational gates (`NO_OPERATIONAL_WINNER`). The project therefore **rejected $M_4$ and retained the clean 31-feature baseline ($M_0$)** for all subsequent production models. The paper author mistakenly copied the feature count of the exploratory $M_4$ trial ($d=42$) into the manuscript.

---

### B. Complete Feature Schema Comparison

#### 1. Canonical 31-Feature Schema (Used by All Final Models, Scalers & Dashboard)
Source: [`src/features/registry.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/features/registry.py), [`configs/phase7c/fitted_scaler_phase7c.json`](file:///C:/AntiGravity%20Files/cloud-failure-detection/configs/phase7c/fitted_scaler_phase7c.json)

| Index | Feature Name | Category | Description |
| :--- | :--- | :--- | :--- |
| 1 | `cpu_utilization` | Group A: Raw Utilization | Average CPU core fraction consumed |
| 2 | `memory_utilization` | Group A: Raw Utilization | Canonical resident memory usage |
| 3 | `assigned_memory` | Group A: Raw Utilization | Memory allocation limit |
| 4 | `disk_io_time` | Group A: Raw Utilization | Cumulative disk I/O time |
| 5 | `cycles_per_instruction` | Group A: Raw Utilization | Hardware execution efficiency (CPI) |
| 6 | `memory_accesses_per_instruction` | Group A: Raw Utilization | Memory bus traffic intensity |
| 7–10 | `cpu_utilization_lag_1, 2, 3, 6` | Group B: Lags | Historical CPU utilization at $t-1, t-2, t-3, t-6$ steps |
| 11–14 | `memory_utilization_lag_1, 2, 3, 6` | Group B: Lags | Historical memory utilization at $t-1, t-2, t-3, t-6$ steps |
| 15 | `cpu_diff_1` | Group C: First Differences | First-difference discrete derivative ($\Delta \text{CPU}$) |
| 16 | `cpu_rel_change_1` | Group C: First Differences | Normalized relative rate of CPU change |
| 17 | `memory_diff_1` | Group C: First Differences | First-difference memory change |
| 18–20 | `cpu_rolling_mean_3`, `std_3`, `range_3` | Group D: Rolling Moments | Trailing 3-step ($15\text{ min}$) CPU rolling statistics |
| 21 | `memory_rolling_mean_3` | Group D: Rolling Moments | Trailing 3-step ($15\text{ min}$) memory rolling mean |
| 22–24 | `cpu_rolling_mean_6`, `std_6`, `range_6` | Group D: Rolling Moments | Trailing 6-step ($30\text{ min}$) CPU rolling statistics |
| 25 | `memory_rolling_mean_6` | Group D: Rolling Moments | Trailing 6-step ($30\text{ min}$) memory rolling mean |
| 26 | `cpu_trend_slope_6` | Group E: Trend & Slope | 6-step OLS linear regression trend slope |
| 27 | `cpu_exceed_85` | Group F: Threshold Indicators | Binary indicator ($\text{CPU} > 0.85$) |
| 28 | `cpu_exceed_fraction_6` | Group F: Threshold Indicators | Trailing 6-step fraction of time exceeding 85% CPU |
| 29 | `memory_to_assigned_ratio` | Group G: Allocation Ratios | Dynamic memory headroom utilization ratio |
| 30 | `task_age_minutes` | Group H: Lifecycle Dynamics | Cumulative runtime age in minutes |
| 31 | `cpi_missing_fraction_6` | Group I: Quality & Tracking | Missingness tracking fraction for hardware counters |

#### 2. The 11 Additional Features in Phase 7K ($M_4$, 42 Features)
Source: [`scripts/run_phase7k_ablation.py#L224-L226`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase7k_ablation.py#L224-L226)

| Additional Feature | Origin Group | Physical Meaning | Reason Excluded from Final Models |
| :--- | :--- | :--- | :--- |
| `max_cpu_rate` | Group J1: Peak Usage | Maximum instantaneous CPU spike | Did not improve warning separation; high collinearity with `cpu_utilization`. |
| `cpu_burstiness` | Group J1: Peak Usage | Difference: `max_cpu_rate - cpu_rate` | Marginally boosted Macro-F1 but caused Critical Recall to collapse to 4.1%. |
| `max_memory` | Group J1: Peak Usage | Maximum instantaneous memory spike | Redundant with `canonical_memory`. |
| `memory_headroom` | Group J1: Peak Usage | Difference: `assigned_memory - max_memory` | Highly correlated with `memory_to_assigned_ratio`. |
| `total_page_cache` | Group J1: Peak Usage | Kernel page cache allocation | Incomplete logging across non-standard tasks. |
| `priority` | Group J2: Scheduler | Borg task scheduling priority integer | Static attribute; provides zero temporal gradient. |
| `scheduling_class` | Group J2: Scheduler | Latency sensitivity class ($0–3$) | Static attribute; failed to separate failure trajectories. |
| `req_cpu` | Group J2: Scheduler | Requested CPU core fraction | Static reservation parameter. |
| `req_ram` | Group J2: Scheduler | Requested memory fraction | Static reservation parameter. |
| `eviction_count_prior` | Group J3: Preemption | Prior evictions on same task index | Slightly boosted recall but inflated pure-normal Entity FAR from 16% to 25.2%. |
| `time_since_last_schedule` | Group J3: Preemption | Time elapsed since last schedule event | Inflated false alarms on long-running batch jobs. |

#### 3. Representation Hybrid Feature Count (63 Features)
Source: [`src/models/hybrid.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/models/hybrid.py), [`models/checkpoints_phase7d/representation_hybrid_phase7d.joblib`](file:///C:/AntiGravity%20Files/cloud-failure-detection/models/checkpoints_phase7d/representation_hybrid_phase7d.joblib)
- **Latent Embedding ($32\text{ dims}$):** Extracted from the final hidden state of the LSTM backbone (`exp3_lstm_2l_h32_balanced.pt`).
- **Instantaneous Telemetry ($31\text{ dims}$):** The canonical 31 features at the anchor time-step ($X[:, -1, :]$).
- **Total Input to Random Forest Head:** $32 + 31 =$ **63 dimensions** (`n_features_in_ = 63`).

---

## 3. Verified Checkpoint Provenance

We inspected the actual binary weight matrices and scikit-learn descriptors in `models/checkpoints_phase7d/` and `models/checkpoints/`:

| Checkpoint Path | Architecture | Training Phase | Window Length ($L$) | Input Features | Training Dataset | Preprocessing Config | Verified Validation Metrics | Provenance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `models/checkpoints_phase7d/random_forest_phase7d.joblib` | `RandomForestClassifier` (100 trees, balanced) | Phase 7D | Instantaneous ($L=1$, anchor step) | **31** | Candidate A Train ($N=3,278$) | `configs/phase7c/fitted_scaler_phase7c.json` | Macro-F1: `0.3653`<br>Accuracy: `73.06%` | **VERIFIED** |
| `models/checkpoints_phase7d/best_lstm_checkpoint_phase7d.pt` | `TemporalLSTMClassifier` (2 layers, hidden=64) | Phase 7D | **$L = 5$** (25 min) | **31** | Candidate A Train ($N=3,278$, shape `[3278, 5, 31]`) | `configs/phase7c/fitted_scaler_phase7c.json` | Macro-F1: `0.3068`<br>Accuracy: `85.24%` | **VERIFIED** |
| `models/checkpoints_phase7d/best_transformer_checkpoint_phase7d.pt` | `CompactTemporalTransformer` (2 layers, $d_{\text{model}}=32$, $n_{\text{head}}=4$) | Phase 7D | **$L = 5$** (25 min) | **31** | Candidate A Train ($N=3,278$, shape `[3278, 5, 31]`) | `configs/phase7c/fitted_scaler_phase7c.json` | Macro-F1: `0.3328`<br>Accuracy: `79.30%` | **VERIFIED** |
| `models/checkpoints_phase7d/representation_hybrid_phase7d.joblib` | RF Head on fused latent embedding | Phase 7D | **$L = 5$** (25 min) | **63** ($32 + 31$) | Candidate A Train ($N=3,278$) | LSTM backbone + Phase 7C scaler | Macro-F1: `0.2880`<br>Accuracy: `60.88%` | **VERIFIED** |
| `models/checkpoints/best_lstm_checkpoint.pt` | `TemporalLSTMClassifier` (2 layers, hidden=64) | Phase 3B (Historical) | **$L = 15$** (75 min) | **31** | Phase 2 Train (`data/processed/X_train.npy`, shape `[4798, 15, 31]`) | `configs/fitted_scaler.json` | Macro-F1: `0.3409` (Phase 3 report) | **VERIFIED** |
| `models/checkpoints/exp3_lstm_2l_h32_balanced.pt` | `TemporalLSTMClassifier` (2 layers, hidden=32) | Phase 4 (Historical) | **$L = 15$** (75 min) | **31** | Phase 2 Train ($N=4,798$) | `configs/fitted_scaler.json` | Serves as backbone for Phase 7D hybrid | **VERIFIED** |

---

## 4. Practical Sliding-Window Test Execution Evidence

Per Section 4 of the audit mandate, we executed both the automated unit test suite and a standalone verification script using actual repository classes.

### A. Automated Unit Test Execution
- **Exact Command:** `.\.venv\Scripts\python.exe -m pytest tests/test_window_generation.py -v`
- **Exit Code:** `0`
- **Output:**
  ```
  ============================= test session starts =============================
  platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
  rootdir: C:\AntiGravity Files\cloud-failure-detection
  configfile: pytest.ini
  plugins: anyio-4.16.0
  collected 2 items

  tests\test_window_generation.py ..                                       [100%]

  ============================= 2 passed in 19.80s ==============================
  ```

---

### B. Minimal Synthetic Demonstration Execution
- **Script:** [`scripts/verify_sliding_window_execution.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/verify_sliding_window_execution.py)
- **Exact Command:** `.\.venv\Scripts\python.exe scripts/verify_sliding_window_execution.py`
- **Exit Code:** `0`
- **Recorded Results:**
  - **Window Length:** 5 steps ($25\text{ minutes}$)
  - **Stride:** 1 step ($5\text{ minutes}$ / $300\text{ seconds}$)
  - **Input Observations:** 8 contiguous 5-minute steps ($t = 300\text{s}$ to $2400\text{s}$); task fails at $t = 2400\text{s}$.
  - **Number of Generated Windows:** 3 valid pre-failure windows (step 8 at $t=2400\text{s}$ is strictly censored as UNKNOWN).
  - **Output 3D Tensor Shape:** `(3, 5, 2)`
  - **Entity Identifier:** `task_test_01`
  - **Target Label Vector:** `array([2, 2, 2])` (CRITICAL)

#### First Two Generated Windows:
```
--- Window 1 ---
Entity ID: task_test_01
Prediction Timestamp: 1500.0s (Anchor at index 4)
Assigned Label: 2 (CRITICAL)
Time to Failure: 900.0s (2400s - 1500s <= 900s)
Features matrix [cpu_utilization, memory_utilization]:
[[0.15 0.3 ]
 [0.22 0.32]
 [0.31 0.35]
 [0.45 0.4 ]
 [0.6  0.48]]

--- Window 2 ---
Entity ID: task_test_01
Prediction Timestamp: 1800.0s (Anchor at index 5)
Assigned Label: 2 (CRITICAL)
Time to Failure: 600.0s (2400s - 1800s <= 900s)
Features matrix [cpu_utilization, memory_utilization]:
[[0.22 0.32]
 [0.31 0.35]
 [0.45 0.4 ]
 [0.6  0.48]
 [0.78 0.55]]
```
- **Overlap Invariant:** `Window 1[1:, :]` is identical to `Window 2[:-1, :]`.
- **Temporal Stride:** $1800.0\text{s} - 1500.0\text{s} = 300.0\text{s}$ ($\equiv \Delta t$).

---

## 5. Training-versus-Inference Consistency

We confirmed that the live dashboard and prediction playground use the identical schema and sequence construction logic:

1. **Feature Ordering:** [`dashboard/utils/inference_pipeline.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/inference_pipeline.py) loads feature names directly from [`configs/phase7c/fitted_scaler_phase7c.json`](file:///C:/AntiGravity%20Files/cloud-failure-detection/configs/phase7c/fitted_scaler_phase7c.json), enforcing the exact 31-feature canonical order.
2. **Imputation & Scaling:** Missing values are imputed using strictly training-set medians stored in `fitted_scaler_phase7c.json`.
3. **Window Construction:**
   - Multi-row uploads ($N \ge 5$) are converted into 5-step sliding sequences with stride $s=1$.
   - Single-row uploads are replicated across 5 steps to satisfy model tensor rank.
4. **Execution Engine:** [`dashboard/utils/model_loader.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/model_loader.py) evaluates `best_lstm_checkpoint_phase7d.pt` and `best_transformer_checkpoint_phase7d.pt` directly on CPU, yielding identical probabilities to offline training evaluations.

---

## 6. Exact Discrepancies Remaining in the Research Paper

| Paper Section / Table | What the Paper Claims | What the Codebase / Data Actually Shows | Discrepancy Severity |
| :--- | :--- | :--- | :--- |
| **Section III, Problem Formulation & Table III Footnote** | Claims window length $W=15$ ($75\text{ minutes}$) | All final Phase 7/8 checkpoints, validation sets, and dashboard use $L=5$ ($25\text{ minutes}$). $L=15$ was abandoned in Phase 6.2 due to sequence mortality. | **HIGH (Methodology Mismatch)** |
| **Section IV, Data Preprocessing** | Claims feature dimension $d = 42$ | Deployed models and scalers use **31 features** (or 63 in the hybrid). 42 features was an exploratory ablation ($M_4$) in Phase 7K that was rejected. | **HIGH (Feature Schema Mismatch)** |
| **Section IV, Data Preprocessing** | Claims dataset is Google Cluster Trace 2019 (1,245,600 tasks, 45.2M steps) | All actual model training, validation, and holdout tests were conducted on **Google Borg 2011 trace**. Borg 2019 was labeled `DATA_ACCESS_BLOCKED` in Phase 7Y. | **CRITICAL (Dataset Provenance Mismatch)** |
| **Table II, Main Results** | Claims Stacked LSTM Macro-F1 = 0.87, Weighted F1 = 0.93 | Genuine Candidate A test Macro-F1 is **0.3155**; Phase 6 holdout Macro-F1 is **0.4425**. Random Forest decisively beat deep learning ($0.4023 - 0.6111$). | **CRITICAL (Result Inconsistency)** |
| **Table III, Lead Time** | Claims median lead time of 12.5 minutes (2.5 steps) on LSTM | Actual sequential models suffered severe majority collapse or high false alarm rates under real 98:1:1 imbalance, failing pre-registered operational gates. | **CRITICAL (Empirical Divergence)** |
| **Table IV, Cross-Dataset Transfer** | Claims cross-dataset transfer on Alibaba (Macro-F1 0.81) and Azure (0.83) | No Alibaba or Azure training or evaluation scripts exist in the repository; these are theoretical projections. | **CRITICAL (Unexecuted Experiments)** |

---

## 7. Recommended Corrections to the Research Paper

To align the research paper with the authentic, defensible implementation in your repository, the following revisions are recommended:

1. **Update Window Length to $L=5$ ($25\text{ minutes}$):**
   - Frame the transition from $L=15$ to $L=5$ as a **major empirical contribution**: cloud containers are short-lived, and demanding $L=15$ drops $>60\%$ of failing tasks. $L=5$ is the empirical Pareto frontier.
2. **Correct Feature Dimension from $d=42$ to $d=31$:**
   - Update Section IV to state $d=31$ canonical telemetry features across Groups A–I.
   - Mention the 42-feature ablation study (Phase 7K) as evidence that adding scheduler metadata and preemption counters did not resolve the operational trade-off.
3. **Accurately Report Dataset Provenance (Google Borg 2011):**
   - State that the empirical evaluation is conducted on the Google Borg 2011 trace across Candidate A and Candidate B partitions. Note that Google Borg 2019 was investigated in Phase 7Y and identified as `DATA_ACCESS_BLOCKED` due to BigQuery egress quotas and schema changes.
4. **Embrace the Empirical Negative Results:**
   - The greatest strength of this research is its **forensic integrity and negative results**: deep sequence models (LSTM, Transformer) do *not* magically solve cloud failure prediction under severe class imbalance ($98:1:1$), and classical tree models (Random Forest) outperform them on short windows. Documenting the operational "Dead Zone" (Gates 1 & 2) and Simpson's Paradox (Job 6251277720) makes this a landmark empirical paper that will easily withstand peer review.

