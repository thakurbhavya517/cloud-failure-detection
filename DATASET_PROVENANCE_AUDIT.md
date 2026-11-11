# Dataset Provenance Audit: Target Paper vs. Repository Reality

**Governing Standard:** Research Paper Integrity & Scientific Provenance Protocol  
**Target Paper Specification:** *"Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing"* (Adnan Iqbal, Bhavya Thakur, Kushagra Pandey, Dr. Kamaldeep)  
**Investigation Date:** November 11, 2026  
**Auditor:** Senior ML Research Engineer & Independent Experimental Auditor (Google DeepMind Team)  
**Target Paper Status:** **IMMUTABLE TARGET SPECIFICATION (STRICTLY FROZEN)**  
**Overall Dataset Verdict:** **`TARGET_DATASETS_UNAVAILABLE_LOCALLY` / `BORG_2011_SAMPLE_ONLY`**

---

## Executive Summary

This forensic provenance audit establishes the exact physical and empirical reality of all datasets claimed in the research paper compared to what genuinely exists inside the repository.

### Definitive Verdicts Across All Stated Datasets:

| Claimed Dataset | Paper Description | Local Existence Status | Sample Count Claimed vs. Actual | Usability for Paper Replication |
| :--- | :--- | :---: | :---: | :--- |
| **Google Cluster Trace 2019** | 1,245,600 task instances, 45.2M steps, 5-min intervals, CC-BY 4.0 | **DOES NOT EXIST LOCALLY** | Claimed: 1.25M tasks<br>Actual: **0 tasks** | **BLOCKED:** Phase 7Y dry run confirmed BigQuery scan requires **418.58 GB** per query; local access was halted (`DATA_ACCESS_BLOCKED`) due to lack of credentials and budget caps. |
| **Google Borg ClusterData 2011** | Not mentioned in paper (paper refers only to 2019) | **ACTUAL SOURCE PRESENT** | Claimed: N/A<br>Actual: **~197k raw tasks; 353 dev tasks** | **FUNCTIONAL DEV DATA:** Shards exist in `data/raw/` (~185.8 MB compressed). Generates ~3,935 dev windows ($L=5$) or ~4,798 train windows ($W=15$). |
| **Alibaba Cluster Trace** | Alibaba 2018 trace, zero-shot and 10% fine-tuned transfer | **DOES NOT EXIST** | Claimed: Cross-cloud transfer<br>Actual: **0 files, 0 bytes** | **UNAVAILABLE:** No raw data, ingestion scripts, or preprocessed arrays exist in the repository. Investigated as feasibility only in Phase 7Z. |
| **Microsoft Azure Public Trace** | Azure VM/task trace, zero-shot and 10% fine-tuned transfer | **DOES NOT EXIST** | Claimed: Cross-cloud transfer<br>Actual: **0 files, 0 bytes** | **UNAVAILABLE:** No raw data, ingestion scripts, or preprocessed arrays exist in the repository. Investigated as feasibility only in Phase 7Z. |
| **Synthetic Benchmark** | Synthetic cloud workload benchmark (`data/synthetic_benchmark/`) | **EMPTY DIRECTORY** | Claimed: N/A<br>Actual: **0 files** | Directory exists on disk but is completely empty (Phase 7V audit). |

---

## 1. Target Dataset Claims in the Research Paper

The research paper establishes the following explicit empirical claims regarding dataset provenance:

1. **Primary Dataset Identity (Section III-A, Paragraph 164):**
   > *"Using the Google Cluster Trace 2019, instance events are joined with five-minute usage records. After removing anomalous incomplete traces, the dataset comprises 1,245,600 task instances and 45.2 million task-steps."*
2. **Temporal Granularity & Features (Section III-A, Paragraph 164):**
   > *"Five-minute usage records... Missing values are handled via forward-filling for up to 3 steps, followed by zero-imputation. Causal rolling features (moving average, rate of change, slope) are computed over an $r = 6$ step (30 min) trailing period, yielding a feature dimension $d = 42$."*
3. **Partitioning Strategy (Section III-C, Paragraph 216):**
   > *"Data is partitioned chronologically (70/15/15) by timestamp, grouped strictly by task ID to guarantee no single task spans the train and test sets (preventing data leakage)."*
4. **Target Failure Definition (Section II, Paragraph 69):**
   > *"Let $e_t^{(i)}$ be the event logged for instance $i$ at step $t$, with anomalous set $\mathcal{A} = \{\text{Fail}, \text{Kill}, \text{Evict}, \text{Lost}\}$. The termination step is $T_f^{(i)} = \min \{ t : e_t^{(i)} \in \mathcal{A} \}$."*
5. **Cross-Dataset Generalisation (Section IV-C, Paragraph 237 & Table IV):**
   > *"Table IV validates the framework’s robustness. While zero-shot transfer to Alibaba and Azure clouds drops performance significantly due to differing infrastructural dynamics, incorporating a minimal 10% fine-tuning subset rapidly recovers Macro-F1 above 0.81 and restores a 10.0-minute median lead time."*
   > - *Alibaba (zero-shot): Macro-F1 0.61, Detection 45.2%, Lead 5.0 min*
   > - *Alibaba (fine-tuned): Macro-F1 0.81, Detection 76.4%, Lead 10.0 min*
   > - *Azure (zero-shot): Macro-F1 0.59, Detection 41.8%, Lead 0.0 min*
   > - *Azure (fine-tuned): Macro-F1 0.83, Detection 79.1%, Lead 10.0 min*

---

## 2. Physical Repository Inventory and Forensic Audit

### 2.1 Raw Data Directory (`data/raw/`) Forensic Inventory

A byte-level inspection of `data/raw/` reveals exactly 5 files totaling 185,797,809 bytes (~177.2 MB compressed):

| File Path | Byte Size | Compression | Format | Schema Identification |
| :--- | :---: | :---: | :---: | :--- |
| `data/raw/sample_task_events.csv.gz` | 4,139,742 | gzip | CSV | Google Borg 2011 (13 columns, $\mu\text{s}$ timestamps, `FAIL=3`, `FINISH=4`) |
| `data/raw/sample_task_events_part1.csv.gz` | 924,634 | gzip | CSV | Google Borg 2011 event shard part 1 |
| `data/raw/sample_task_events_part2.csv.gz` | 1,821,031 | gzip | CSV | Google Borg 2011 event shard part 2 |
| `data/raw/sample_task_usage.csv.gz` | 91,723,415 | gzip | CSV | Google Borg 2011 (20 columns, 300s windows, `cpu_rate`, `canonical_memory`) |
| `data/raw/sample_task_usage_part1.csv.gz` | 87,188,987 | gzip | CSV | Google Borg 2011 usage shard part 1 |

### 2.2 Schema Identification: Proof of Borg 2011 vs. Borg 2019

The table headers extracted directly from the raw files empirically confirm the Borg 2011 schema (`ClusterEventTypeV2`):

#### Raw Task Events Schema:
```
Columns: ['0', '2', '3418309', '0.1', '4155527081', '0.2', '70s3v5qRyCO...', '3', '9', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12']
Corresponding to Borg 2011:
1. timestamp (microseconds)
2. missing_info
3. job_id
4. task_index
5. machine_id
6. event_type (0=SUBMIT, 1=QUEUE, 2=ENABLE, 3=SCHEDULE, 4=EVICT, 5=FAIL in 2019, but 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL in 2011)
7. user_name
8. scheduling_class
9. priority
10. req_cpu
11. req_ram
12. req_disk
13. diff_machine_restriction
```

#### Raw Task Usage Schema:
```
Columns: ['600000000', '900000000', '3418309', '0', '4155527081', '0.001562', '0.06787', '0.07568', ...]
Corresponding to Borg 2011:
1. start_time (600,000,000 us = 600s = 10 min)
2. end_time (900,000,000 us = 900s = 15 min; difference = 300s = 5-minute sampling interval)
3. job_id
4. task_index
5. machine_id
6. cpu_rate
7. canonical_memory
8. assigned_memory
9. unmapped_page_cache
10. total_page_cache
11. max_memory
12. disk_io_time
13. local_disk_space
14. max_cpu_rate
15. max_disk_io_time
16. cpi
17. mai
18. sample_rate
19. aggregation_type
20. sampled_cpu_usage
```

**Definitive Schema Proof:**
- In Google Borg 2019 (v3 format, Wilkes et al. 2020), table schemas are nested BigQuery structs (`average_usage.cpus`, `maximum_usage.cpus`, `assigned_memory`, `collection_id`, `instance_index`), and event codes map `FAIL = 5` and `FINISH = 6`.
- In the local repository, the code explicitly relies on `ClusterEventTypeV2` where `FAIL = 3`, `FINISH = 4`, `KILL = 5`, and `EVICT = 2`.
- Therefore, **100% of the raw data in the local repository originates from the Google Borg 2011 trace**, NOT the Google Cluster Trace 2019.

---

## 3. Investigation History of Google Borg 2019 (Phase 7Y & 7Z Audits)

The repository contains extensive forensic evidence demonstrating why Google Cluster Trace 2019 was not downloaded locally:

1. **Phase 7Y Investigation (`results/phase7y_borg2019_pilot/phase7y_final_report.md`):**
   - **Date:** October 29, 2026.
   - **Target:** Google Borg ClusterData 2019 (`google.com:google-cluster-data`).
   - **Verdict:** **`DATA_ACCESS_BLOCKED`**.
   - **Root Cause:** Attempted BigQuery extraction failed with `DefaultCredentialsError: Your default credentials were not found.` No Google Cloud ADC credentials existed on the host machine.
   - **Governance Safeguard:** Experimental governance rules prohibited manufacturing synthetic pilot data; extraction was halted.
2. **Phase 7Y/7Z BigQuery Query Dry Run & Cost Audit:**
   - **Query File:** `results/phase7y_borg2019_pilot/query_plan.sql` (SHA-256 `58e5a2e4d864182f...`).
   - **Verified Service Dry Run:** A full telemetry join query across 8 unpartitioned cells (`clusterdata_2019_a` through `h`) requires scanning **418,578,205,568 bytes (~418.58 GB)**.
   - **Financial Cost / Quota Impact:** At standard BigQuery on-demand rates ($6.25/TiB), a single dry run consumes ~$2.38 USD and 38.1% of an entire monthly 1 TiB free tier.
   - **Budget Cap:** The project budget cap was strictly locked at $0.125 USD (25 GB), stopping any unauthorized cloud extraction.
3. **Phase 8 Research Integrity Audit (`reports/phase8_research_integrity/phase8_dataset_integrity_audit.md`):**
   - On November 3, 2026, Phase 8 formally suspended external cloud queries and BigQuery extraction to prioritize completing the research using local empirical evidence.

---

## 4. Alibaba and Azure Cross-Dataset Audit

The research paper reports cross-dataset generalization experiments on Alibaba and Azure clouds in Section IV-C (Table IV).

### Forensic Verification:
1. **File Search:** Scanning the entire repository for Alibaba and Azure data files returns **0 files**.
2. **Directory Search:** `data/` contains no subfolders for Alibaba or Azure.
3. **Script Search:** Scanning `src/ingestion/` reveals only `google_borg_loader.py` and `schema.py`. No ingestion parser, schema mapping, or loader exists for Alibaba or Azure trace formats.
4. **Prior Phase Audits:**
   - **Phase 7V Audit (`reports/phase7v_external_generalization/generalization_audit.md`):** Formally declared `NO_INDEPENDENT_COHORT_AVAILABLE (Path B)`.
   - **Phase 7W Audit (`reports/phase7w_external_data/label_semantics_audit.md`):** Explicitly observed that Alibaba 2018 (`batch_task.csv`, `batch_instance.csv`) and Azure traces do not provide native 3-class failure proximity labels and have irregular sampling cadences.
   - **Phase 7Z Feasibility Study (`results/phase7y_borg2019_pilot/phase7z_external_telemetry_access_feasibility.json`):** Evaluated Alibaba and Azure as theoretical alternatives, noting zero data extraction had occurred.

**Conclusion:** The Alibaba and Azure cross-dataset experiments reported in the paper were not executed against local data in this repository and cannot be reproduced from local assets.

---

## 5. Sample Count & Scale Discrepancy

| Metric | Paper Stated Claim | Actual Repository Data (Candidate A Dev) | Actual Repository Data (Full Borg 2011 Sample) | Discrepancy Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Task Instances ($I$)** | 1,245,600 | 353 entities (284 train, 69 val) | ~197,032 tasks in sample shard | **~3,528x** vs Dev<br>**~6.3x** vs Raw |
| **Telemetry Steps ($N$)** | 45.2 million | 3,935 dev windows | ~500,000 raw step intervals | **~11,486x** vs Dev<br>**~90x** vs Raw |
| **Trace Duration** | 30 days (typical 2019) | May 2011 sample shard | May 2011 sample shard | Differing eras (2019 vs 2011) |
| **Evaluation Sequences** | Full cluster-wide trace | 3,935 dev + 1,025 test | ~10,000 viable sliding windows | Fixed subsample |

### Cause of the Discrepancy:
The numbers reported in the research paper ($1,245,600$ tasks, $45.2\text{ million}$ steps) represent the full Borg 2019 8-cell production trace published by Google Research (Wilkes et al., 2020). The local repository operates on an exploratory research sample extracted from Google Borg 2011 shards.

---

## 6. Target Failure Definition Mismatch ($\mathcal{A}$ Set vs. Strict Failures)

A critical scientific discrepancy exists in how a "failure" is defined between the paper and the code:

### Paper Definition (Section II, Paragraph 69):
$$\mathcal{A} = \{\text{Fail}, \text{Kill}, \text{Evict}, \text{Lost}\}$$
- The paper treats catastrophic crashes (`FAIL`), user cancellations (`KILL`), infrastructure preemptions (`EVICT`), and lost worker signals (`LOST`) as an aggregated anomalous termination class $\mathcal{A}$.
- The paper defines termination time $T_f = \min \{ t : e_t \in \mathcal{A} \}$.

### Codebase Reality (`src/labeling/target_generator.py`, lines 50–125):
- The codebase restricts positive failure **strictly to `FAIL`** (`ClusterEventTypeV2.FAIL == 3`).
- Crucially, `KILL` and `EVICT` within 30 minutes are treated as **`UNKNOWN (-1)` (censored observations)**:
  ```python
  # Check if task was cancelled (KILL) within the 30-minute window
  if kill_time <= t + self.warning_horizon_sec:
      # Censored: User cancellation terminates the run; cannot observe potential failure
      return FailureLabel.UNKNOWN, None

  # Check if task was evicted (EVICT) within the 30-minute window
  if evict_time <= t + self.warning_horizon_sec:
      # Censored: Infrastructure preemption terminates the run
      return FailureLabel.UNKNOWN, None
  ```
- Because `WindowGenerator` excludes `UNKNOWN` labels (`exclude_unknown=True`), all tasks terminated by `KILL` or `EVICT` are completely dropped from the training and evaluation tensors.

**Impact:**
If the anomalous set $\mathcal{A}$ is expanded to include `KILL` and `EVICT` as the paper claims, the positive failure cohort increases significantly (since kills and evictions outnumber genuine fails by 5–10x in cluster traces), altering class balance and label dynamics.

---

## 7. Forensic Summary and Scientific Verdict

1. **Raw Google Cluster Trace 2019 Does Not Exist in Repository:** The local data is strictly Google Borg 2011.
2. **Alibaba and Azure Traces Do Not Exist:** Zero bytes of external multi-cloud telemetry exist locally.
3. **Scale Mismatch:** The repository contains a 353-entity development cohort and ~10,000 evaluable windows, not 1.25M tasks and 45.2M steps.
4. **Governing Rule Compliance:** Per user instructions, we do not substitute Borg 2011 data and falsely relabel it "Google Cluster Trace 2019".
5. **Reproducibility Verdict:** The paper's claimed dataset (Google Borg 2019 1.25M tasks) and cross-dataset transfer (Alibaba/Azure) **cannot be reproduced from local repository assets alone**. Any model alignment conducted locally must operate honestly on the available Google Borg trace telemetry while transparently documenting the provenance boundary.

