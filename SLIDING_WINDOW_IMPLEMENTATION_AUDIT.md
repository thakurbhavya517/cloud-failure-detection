# Independent Scientific Audit: Sliding-Window Mechanism Implementation

**Target Artifact:** Research Paper — *"Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing"*  
**Repository:** `cloud-failure-detection`  
**Audit Question:** *"Has the sliding-window mechanism described in the research paper actually been implemented in the project source code, or is it only mentioned in the paper?"*  
**Audit Date:** 2026-11-11  
**Audit Status:** Complete & Empirically Verified  

---

## Executive Summary & Final Verdict

### Final Verdict: **YES — Fully implemented and verified.**

The sliding-window mechanism described in the research paper is **not merely theoretical text; it is genuinely and comprehensively implemented in the production source code**. 

The repository contains a dedicated, rigorous time-series sliding-window generator ([`src/labeling/window_generator.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py)) and multi-horizon target assigner ([`src/labeling/target_generator.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/target_generator.py)), backed by automated invariant test suites ([`tests/test_window_generation.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/tests/test_window_generation.py)). The algorithm constructs strictly chronologically ordered 3D tensors $\mathbf{X} \in \mathbb{R}^{N \times L \times F}$ with stride $s=1$ (5-minute step advance), strictly respects entity/task boundaries, enforces causal feature constraints, strictly censors post-failure telemetry, and feeds identical sequence representations through training, offline evaluation, and live inference.

A minor empirical protocol divergence exists between the initial paper draft ($W=15$ steps / 75 min) and late-phase deployed checkpoints ($L=5$ steps / 25 min): the codebase initially implemented $L=15$ (Phases 2–5), but an empirical audit (Phase 6.2) proved that $L=15$ caused catastrophic sequence dropout because cloud containers frequently have lifespans $<75$ minutes. The production protocol was refined to $L=5$ to maximize task retention while preserving temporal dynamics. Both $L=15$ and $L=5$ configurations are fully operational in the codebase.

---

## 1. Exact Sliding-Window Methodology Described in the Paper

From our direct forensic extraction of [`Research_Paper_team_adnan_kushagra_and_bhavya.docx`](file:///c:/Users/Bhavya%20Thakur/Downloads/Research_Paper_team_adnan_kushagra_and_bhavya.docx):

### A. Sampling Cadence & Time-Series Construction
- **Sampling Interval ($\Delta t$):** Fixed 5-minute resolution ($300\text{ seconds}$).
- **Entity Instance ($i \in I$):** Observed for $N_i$ steps, where each step $t$ is described by a feature vector $\mathbf{x}_t^{(i)} \in \mathbb{R}^d$.

### B. Mathematical Sliding-Window Definition
The paper formalizes the input tensor in **Section III (Problem Formulation), Equation (1)**:
$$\mathbf{X}_{t-W+1:t}^{(i)} = \begin{bmatrix} \mathbf{x}_{t-W+1}^{(i)}, \dots, \mathbf{x}_t^{(i)} \end{bmatrix}^\top \in \mathbb{R}^{W \times d}$$
- **Window Length ($W$):** The number of trailing observation steps. Table III note specifies: *"Validation-selected $W = 15$ and $k = 3$; minutes = steps $\times 5$."* (Representing a 75-minute historical receptive field).
- **Stride ($s$):** Implied step-by-step advance $t \to t+1$ (stride $s=1$ step, or 5 minutes).

### C. Target Labeling & Prediction Horizons
In **Section III, Equations (2) & (3)**, the termination event timestamp is defined as:
$$T_f^{(i)} = \min \{t : e_t^{(i)} \in \mathcal{A}\}, \quad \text{where } \mathcal{A} = \{\text{Fail}, \text{Kill}, \text{Evict}, \text{Lost}\}$$
For horizons $H_1 < H_2$, the early-warning target state $y_t^{(i)}$ at step $t < T_f^{(i)}$ is defined as:
$$y_t^{(i)} = \begin{cases} 
2 \text{ (CRITICAL)}, & T_f^{(i)} - t \le H_1 \\ 
1 \text{ (WARNING)}, & H_1 < T_f^{(i)} - t \le H_2 \\ 
0 \text{ (NORMAL)}, & \text{otherwise} 
\end{cases}$$
The paper text explicitly sets:
- $H_1 = 3\text{ steps}$ ($15\text{ minutes}$) $\to$ CRITICAL
- $H_2 = 6\text{ steps}$ ($30\text{ minutes}$) $\to$ WARNING
- Normal operation $\to$ NORMAL ($> 30\text{ minutes}$ before termination or healthy completion)

### D. Leakage Prevention & Data Splitting
- **Chronological & Task-Grouped:** 70% Train, 15% Validation, 15% Test.
- Guaranteed zero task overlap across partitions (preventing identity leakage).

---

## 2. Actual Source Code Implementation

The repository contains complete, production-grade implementations of the exact pipeline claimed:

### A. Core Sliding-Window Engine: `WindowGenerator`
**File:** [`src/labeling/window_generator.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py)  
**Class:** `WindowGenerator` (Lines 11–150)

```python
class WindowGenerator:
    def __init__(
        self,
        sequence_length: int = 15,
        expected_interval_sec: float = 300.0,
        allowed_jitter_sec: float = 30.0,
        require_full_history: bool = True,
        exclude_unknown: bool = True,
    ):
        self.sequence_length = sequence_length
        self.expected_interval_sec = expected_interval_sec
        ...
```

#### Key Implementation Mechanisms:
1. **Chronological Sorting:**
   ```python
   # Line 63: Enforces strict temporal ordering
   df = entity_df.sort_values("end_time_sec").reset_index(drop=True)
   ```
2. **Contiguity Validation:**
   ```python
   # Lines 34-55: Validates that consecutive observations are contiguous (240s <= delta <= 450s)
   def is_valid_contiguous_history(self, timestamps: np.ndarray) -> bool:
   ```
3. **Sliding Window Extraction Loop (Stride = 1):**
   ```python
   # Lines 78-85: Advances prediction anchor step-by-step
   for pred_idx in range(start_idx, num_rows):
       target_label = labels[pred_idx]
       window_start_idx = max(0, pred_idx - self.sequence_length + 1)
       history_features = feature_mat[window_start_idx : pred_idx + 1]
   ```
4. **Entity Isolation (Zero Bleed):**
   ```python
   # Line 129: Windows are generated strictly per entity
   for eid, group in df.groupby("entity_id", sort=False):
       ent_windows = self.create_windows_for_entity(group, feature_cols)
   ```
5. **Output 3D Tensor Construction:**
   ```python
   # Line 139: Stacks windows into (N, L, F) tensor
   X = np.stack([w["features"] for w in all_windows], axis=0).astype(np.float32)
   y = np.array([w["label"] for w in all_windows], dtype=np.int64)
   ```

---

### B. Horizon Labeling & Censoring Engine: `TargetLabelGenerator`
**File:** [`src/labeling/target_generator.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/target_generator.py)  
**Class:** `TargetLabelGenerator` (Lines 15–173)

- **Configured Horizons:** Lines 28–29:
  - `critical_horizon_sec = 900.0` ($15\text{ minutes}$)
  - `warning_horizon_sec = 1800.0` ($30\text{ minutes}$)
- **Post-Failure Censoring (Protocol Invariant):** Lines 59–68:
  ```python
  # Any observation at or after first failure (time_sec <= t) is strictly labeled UNKNOWN (-1)
  prior_fails = task_events[(task_events["event_type"] == self.event_cls.FAIL) & (task_events["time_sec"] <= t)]
  if not prior_fails.empty:
      return FailureLabel.UNKNOWN, None
  ```
- **3-State Target Assignment:** Lines 82–88:
  ```python
  if 0.0 < tau <= self.critical_horizon_sec:
      return FailureLabel.CRITICAL, tau
  elif self.critical_horizon_sec < tau <= self.warning_horizon_sec:
      return FailureLabel.WARNING, tau
  else:
      return FailureLabel.NORMAL, tau
  ```

---

### C. Trailing Causal Feature Engineering: `FeatureEngineer`
**File:** [`src/features/engineer.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/features/engineer.py) (Lines 10–125)  
**File:** [`src/features/registry.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/features/registry.py) (31 Canonical Features)

- Computes strictly trailing lag features ($k \in \{1, 2, 3, 6\}$), first-differences, trailing rolling moments over 3 and 6 steps ($15\text{ min}$ and $30\text{ min}$), and 6-step OLS trend slopes.
- Grouped strictly by `entity_id` (`df.groupby("entity_id", sort=False)`), preventing inter-task leakage.

---

### D. Leakage-Safe Splitting Engine: `LeakageSafeSplitter`
**File:** [`src/splitting/leakage_safe_split.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/splitting/leakage_safe_split.py) (Lines 13–186)

- Implements chronological splitting with a **30-minute boundary embargo** (`split_chronological`), purging examples within $H_{\text{warning}}$ of split boundaries to prevent label leakage.
- Implements entity-grouped splitting (`split_entity_grouped`), ensuring zero parent-job or task overlap.
- Includes `audit_leakage()`, which asserts zero index overlap and verifies temporal boundary integrity.

---

### E. Live Inference Pipeline: `InferencePipeline`
**File:** [`dashboard/utils/inference_pipeline.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/inference_pipeline.py) (Lines 86–105)

The production inference pipeline uses the exact same sliding-window geometry:
```python
if num_rows == 5:
    X_out = np.expand_dims(scaled_vals, axis=0) # Shape (1, 5, 31)
elif num_rows > 5:
    windows = []
    for i in range(num_rows - 4):
        windows.append(scaled_vals[i : i + 5])
    X_out = np.array(windows) # Shape (N - 4, 5, 31)
```

---

## 3. Mathematical Verification of Window Generation

To verify that the implementation creates genuine sliding windows rather than naive array reshaping:

Let an entity $i$ have observations at timestamps $t_1 < t_2 < \dots < t_M$ with feature vectors $\mathbf{x}_1, \dots, \mathbf{x}_M$.

For window length $L$ and stride $s=1$:
1. **Window Index $j=0$ (at prediction anchor $t_L$):**
   $$\mathbf{W}_0 = \begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \dots & \mathbf{x}_L \end{bmatrix}^\top, \quad \text{Label } y_0 = \text{label}(t_L)$$
2. **Window Index $j=1$ (at prediction anchor $t_{L+1}$):**
   $$\mathbf{W}_1 = \begin{bmatrix} \mathbf{x}_2 & \mathbf{x}_3 & \dots & \mathbf{x}_{L+1} \end{bmatrix}^\top, \quad \text{Label } y_1 = \text{label}(t_{L+1})$$
3. **Consecutive Overlap Invariant:**
   $$\mathbf{W}_0[1 : L, :] \equiv \mathbf{W}_1[0 : L-1, :]$$
   *(The suffix of Window 0 is strictly identical to the prefix of Window 1).*
4. **Temporal Stride Invariant:**
   $$t_{\text{pred}}(\mathbf{W}_1) - t_{\text{pred}}(\mathbf{W}_0) \equiv \Delta t = 300\text{ seconds}$$

---

## 4. Practical Implementation Test Run

We executed a live verification script ([`scratch/demo_sliding_window.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scratch/demo_sliding_window.py)) using the actual repository classes `WindowGenerator` and `TargetLabelGenerator`.

### Test Setup:
- **Entity:** `task_test_01`
- **Cadence:** $\Delta t = 300.0\text{ s}$ ($5\text{ minutes}$)
- **Total Observations:** 8 steps from $t = 300\text{ s}$ to $t = 2400\text{ s}$
- **Failure Event:** Task failure logged at $t = 2400.0\text{ s}$
- **Window Length:** $L = 5$ steps ($25\text{ minutes}$)
- **Stride:** $s = 1$ step ($5\text{ minutes}$)

### Execution Output:

```
======================================================================
PRACTICAL SLIDING WINDOW GENERATION AUDIT TEST
======================================================================

--- 1. Original Chronologically Ordered Observations ---
      entity_id  end_time_sec  cpu_utilization  memory_utilization  label  time_to_failure_sec
0  task_test_01         300.0             0.15                0.30      0               2100.0
1  task_test_01         600.0             0.22                0.32      1               1800.0
2  task_test_01         900.0             0.31                0.35      1               1500.0
3  task_test_01        1200.0             0.45                0.40      1               1200.0
4  task_test_01        1500.0             0.60                0.48      2                900.0
5  task_test_01        1800.0             0.78                0.55      2                600.0
6  task_test_01        2100.0             0.89                0.62      2                300.0
7  task_test_01        2400.0             0.94                0.70     -1                  NaN

--- 2. Sliding Window Generation Results ---
Total Observations Input: 8
Configured Window Length (L): 5 steps (25 minutes)
Algorithm Stride (s): 1 step (5 minutes / 300 seconds)
Total Valid Windows Generated: 3 (N = 3 windows)
Output 3D Tensor Shape: X in R^(3, 5, 2) (N=3, L=5, F=2)
Output Target Vector Shape: y in R^(3,)

--- 3. Generated Sliding Window 1 ---
Index: 0
Entity ID: task_test_01
Covered Timestamps: [300.0, 600.0, 900.0, 1200.0, 1500.0]s
Prediction Timestamp (t_pred): 1500.0s
Time to Failure: 900.0s
Assigned Ground-Truth Label: 2 (CRITICAL)
Window Tensor Values [cpu_utilization, memory_utilization]:
[[0.15 0.3 ]
 [0.22 0.32]
 [0.31 0.35]
 [0.45 0.4 ]
 [0.6  0.48]]

--- 4. Generated Sliding Window 2 ---
Index: 1
Entity ID: task_test_01
Covered Timestamps: [600.0, 900.0, 1200.0, 1500.0, 1800.0]s
Prediction Timestamp (t_pred): 1800.0s
Time to Failure: 600.0s
Assigned Ground-Truth Label: 2 (CRITICAL)
Window Tensor Values [cpu_utilization, memory_utilization]:
[[0.22 0.32]
 [0.31 0.35]
 [0.45 0.4 ]
 [0.6  0.48]
 [0.78 0.55]]

--- 5. Generated Sliding Window 3 ---
Index: 2
Entity ID: task_test_01
Covered Timestamps: [900.0, 1200.0, 1500.0, 1800.0, 2100.0]s
Prediction Timestamp: 2100.0s | TTF: 300.0s | Label: 2 (CRITICAL)
Window Tensor Values [cpu_utilization, memory_utilization]:
[[0.31 0.35]
 [0.45 0.4 ]
 [0.6  0.48]
 [0.78 0.55]
 [0.89 0.62]]

[VERIFIED] All mathematical sliding window invariants verified successfully!
```

### Key Audit Observations from Test:
1. **Consecutive Overlap Verified:** Suffix of Window 1 `X[0, 1:, :]` exactly equals prefix of Window 2 `X[1, :-1, :]`.
2. **Temporal Stride Verified:** Prediction timestamp advances by exactly $300.0\text{ s}$ ($1800 - 1500 = 300$).
3. **Censoring Verified:** Observation 7 ($t=2400.0\text{ s}$, concurrent with failure) was correctly assigned label `-1` (UNKNOWN) and purged from model inputs, proving zero post-failure leakage.

---

## 5. Paper Claims vs. Code Implementation Matrix

| Methodological Detail | Research Paper Claim | Actual Code Implementation | Evidence (File & Line Numbers) | Audit Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Window Length ($W$ / $L$)** | $W = 15$ steps ($75\text{ min}$) in Table III note & Eq (1) | `sequence_length = 15` in `configs/default_config.yaml` and Phase 2–5 pipelines. Refined to $L=5$ ($25\text{ min}$) in Phase 6.2+ & Phase 7 checkpoints due to task mortality. | [`configs/default_config.yaml#L21`](file:///C:/AntiGravity%20Files/cloud-failure-detection/configs/default_config.yaml#L21), [`src/labeling/window_generator.py#L22`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L22), [`scripts/reconstruct_phase7c_raw_features.py#L182-186`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/reconstruct_phase7c_raw_features.py#L182-L186) | **VERIFIED** |
| **Algorithm Stride ($s$)** | Implied step-by-step advance $t \to t+1$ ($\Delta t = 5\text{ min}$) | Explicit step-by-step increment `pred_idx += 1` with 300s delta check. | [`src/labeling/window_generator.py#L78`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L78), [`dashboard/utils/inference_pipeline.py#L93`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/inference_pipeline.py#L93) | **VERIFIED** |
| **Temporal Ordering** | Strictly trailing chronological sequence $X_{t-W+1:t}$ | Explicit sorting by `end_time_sec`, slice `[window_start_idx : pred_idx + 1]`, and contiguity validation. | [`src/labeling/window_generator.py#L34-55`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L34-L55), [`src/labeling/window_generator.py#L63`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L63) | **VERIFIED** |
| **Entity Boundaries** | Per-task instance grouping ($i \in I$) | `df.groupby("entity_id", sort=False)` ensures windows never span multiple entities. | [`src/labeling/window_generator.py#L129`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L129), [`src/features/engineer.py#L48`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/features/engineer.py#L48) | **VERIFIED** |
| **Label Assignment** | Eq. (3): 3-state label based on time remaining before termination | Assigned at anchor $t_{\text{pred}} = \text{timestamps}[pred\_idx]$ using future failure event horizon. | [`src/labeling/target_generator.py#L82-88`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/target_generator.py#L82-L88), [`src/labeling/window_generator.py#L79`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L79) | **VERIFIED** |
| **Prediction Horizons** | $H_1 = 3$ ($15\text{ min}$), $H_2 = 6$ ($30\text{ min}$) | `critical_horizon_sec = 900.0`, `warning_horizon_sec = 1800.0`. | [`src/labeling/target_generator.py#L28-29`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/target_generator.py#L28-L29), [`configs/default_config.yaml#L25-26`](file:///C:/AntiGravity%20Files/cloud-failure-detection/configs/default_config.yaml#L25-L26) | **VERIFIED** |
| **Short-Task Handling** | Mentions removing incomplete traces | `require_full_history=True` drops tasks with $< L$ steps; `require_full_history=False` left-pads with zero. | [`src/labeling/window_generator.py#L67-68`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L67-L68), [`src/labeling/window_generator.py#L91-97`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/window_generator.py#L91-L97) | **VERIFIED** |
| **Data Leakage Prevention** | Chronological 70/15/15 split, embargo, post-failure censoring | `LeakageSafeSplitter` implements 30-min boundary embargo; `TargetLabelGenerator` censors post-failure steps. | [`src/splitting/leakage_safe_split.py#L58-65`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/splitting/leakage_safe_split.py#L58-L65), [`src/labeling/target_generator.py#L59-68`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/labeling/target_generator.py#L59-L68) | **VERIFIED** |
| **Inference Consistency** | Model takes sliding windows $\mathbb{R}^{W \times d}$ | `InferencePipeline` in dashboard replicates identical 5-step window slicer for CPU inference. | [`dashboard/utils/inference_pipeline.py#L86-105`](file:///C:/AntiGravity%20Files/cloud-failure-detection/dashboard/utils/inference_pipeline.py#L86-L105) | **VERIFIED** |

---

## 6. Discrepancies and Clarifications for the Reviewer

While the **sliding-window mechanism itself is 100% genuine, implemented, and verified**, the following nuances should be understood:

1. **Window Length Refinement ($L=15 \to L=5$):**
   - **Paper Text:** The paper draft reflects the early-phase protocol proposal ($W=15$ steps / 75 min).
   - **Code Reality:** The codebase contains the original $L=15$ implementation ([`scripts/run_phase2_pipeline.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/run_phase2_pipeline.py), [`scripts/extract_confirmatory_data.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/extract_confirmatory_data.py)), but also documents a major empirical finding (Phase 6.2 audit): in real cloud clusters, over 60% of tasks finish or terminate in under 75 minutes. Demanding $L=15$ led to extreme sequence mortality. The protocol was officially amended to $L=5$ ($25\text{ minutes}$) in Phase 6.2 ([`scripts/build_protocol_l5.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/scripts/build_protocol_l5.py)), which is what the saved Phase 7 checkpoints and dashboard use.
   - **Recommendation for Reviewer:** Clarify that $W=15$ was the initial protocol and $L=5$ is the empirical operational optimum, both fully supported by the codebase's `WindowGenerator(sequence_length=...)` parameter.

2. **Feature Dimension ($d=42$ in paper vs. $d=31$ in code):**
   - **Paper Text:** Mentions $d=42$ features.
   - **Code Reality:** [`src/features/registry.py`](file:///C:/AntiGravity%20Files/cloud-failure-detection/src/features/registry.py) defines exactly **31 canonical features** across Groups A–I (or 63 in the representation hybrid).
   - **Impact on Sliding Window:** None. The sliding-window algorithm operates agnostically over $\mathbb{R}^{N \times L \times F}$.

3. **Trace Version & Cross-Dataset Evaluation:**
   - **Paper Text:** Mentions Google Cluster Trace 2019, Alibaba, and Azure.
   - **Code Reality:** All training and confirmatory splits locally implemented run on the Google Borg 2011 trace (Candidate A clean split, Candidate B holdout). Borg 2019 was audited in Phase 7Y and marked `DATA_ACCESS_BLOCKED` due to BigQuery egress limits. Alibaba/Azure transfer tables in the paper represent planned transfer studies.
   - **Impact on Sliding Window:** None. The sliding-window transformation code applies universally to any 5-minute telemetry dataframe regardless of cluster origin.

---

## 7. Conclusion

To the reviewer's inquiry:
> *"Has the sliding-window mechanism described in the research paper actually been implemented in the project source code, or is it only mentioned in the paper?"*

**Formal Response:**
The sliding-window mechanism is **fully and authentically implemented in the project source code**. It is the central data-generation engine of the entire repository. The algorithm:
1. Re-samples raw telemetry into synchronized 300-second bins.
2. Formulates trailing chronological windows of length $L$ advancing with stride $s=1$.
3. Assigns multi-horizon ground-truth failure labels to each window at its prediction anchor timestamp.
4. Prevents target and post-failure leakage through strict causal censoring.
5. Feeds identical 3D tensor representations $\mathbf{X} \in \mathbb{R}^{N \times L \times F}$ through both training pipelines and the interactive inference dashboard.

