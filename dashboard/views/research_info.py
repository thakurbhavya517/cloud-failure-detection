"""Research Information & Provenance View for Cloud Failure Detection Dashboard."""

import streamlit as st


def render_research_info_view():
    st.markdown("## 📚 Research Information & Theoretical Methodology")
    st.markdown(
        "Scientific foundation, empirical breakthroughs, methodological principles, "
        "and formal citations for the predictive container anomaly detection framework."
    )

    # 1. Research Objectives & Problem Statement
    st.markdown("### 🎯 Core Research Objectives")
    st.markdown(
        r"""
        In multi-tenant hyperscale cloud environments (e.g., Google Borg, Kubernetes), containerized workloads 
        experience sudden terminations, evictions, out-of-memory (OOM) kills, and task failures. 
        Traditional anomaly detection relies on reactive thresholding, alerting operators only after service disruption.
        
        This research program investigates whether **multivariate time-series telemetry** can predict 
        runtime container failures in advance.
        
        - **Snapshot vs. Temporal Modeling:** Snapshot predictors evaluate a single instant feature vector, whereas 
          temporal predictors evaluate an ordered sequence of trailing observations to capture evolutionary degradation.
        - **Advance Lead-Time Quantification:** Rather than merely classifying binary status, models are evaluated 
          on the operational lead time ($\ell = T_{\text{fail}} - t_{\text{alert}}$) delivered before task termination.
        - **Reliability vs. False Alarms:** A persistent alert policy ($k = 3$ consecutive warning/critical flags) is 
          enforced to suppress transient jitter and ensure actionable alerts for cluster schedulers.
        """
    )

    st.markdown("---")

    # 2. Target Paper Architecture Specification
    st.markdown("### 🏗️ Target Paper Architecture Specification")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            r"""
            #### 1. Stacked Temporal LSTM
            - **Layers:** 2 Recurrent LSTM layers
            - **Hidden Dimension:** 128 hidden units per layer
            - **Input Dimension:** $d = 42$ causal rolling & usage features
            - **Sequence Length:** $W = 15$ observations ($75\text{ minutes}$ history at 5-min intervals)
            - **Regularization:** Dropout rate of $0.20$ between layers
            - **Loss Function:** Multi-Class Focal Loss ($\gamma = 2.0$) with class-frequency weighting
            """
        )

    with c2:
        st.markdown(
            r"""
            #### 2. Lightweight Compact Transformer
            - **Layers:** 2 Transformer Encoder layers
            - **Attention Heads:** 4 multi-head self-attention heads ($d_k = 16$)
            - **Model Dimension ($d_{\text{model}}$):** 64
            - **Feed-Forward Dimension ($d_{\text{ff}}$):** 128
            - **Sequence Length:** $W = 15$ observations ($75\text{ minutes}$)
            - **Pooling Mode:** Trailing-step extraction with LayerNorm
            - **Loss Function:** Multi-Class Focal Loss ($\gamma = 2.0$)
            """
        )

    st.markdown("---")

    # 3. Data Provenance & Empirical Boundaries
    st.markdown("### 🔬 Data Provenance & Empirical Boundaries")
    st.markdown(
        """
        In strict adherence to scientific integrity:
        
        - **Local Empirical Telemetry (Google Borg 2011 Pilot):** All models evaluated in this dashboard were trained and evaluated on 
          genuine production shards of the **Google Borg ClusterData 2011 trace** (`data/raw/`).
        - **Google Borg 2019 Trace Investigation (Phase 7Y/7Z):** Official BigQuery query plans were designed and 
          verified against Google's protobuf specifications. A service dry run confirmed that a single telemetry query 
          scans **418.58 GB of unpartitioned data**, which was stopped due to budget caps and local credential boundaries.
        - **Leakage Prevention Protocol:** A strict **30-minute temporal embargo** is enforced between training, validation, 
          and test partitions, completely eliminating cross-split contamination and horizon leakage.
        """
    )

    st.markdown("---")

    # 4. Google Cluster Trace 2019 Investigation & Feasibility Analysis (Phase 7Y/7Z)
    st.markdown("### 🛰️ Google Cluster Trace 2019 Access Investigation (Phase 7Y/7Z)")
    st.markdown(
        """
        A rigorous forensic and technical audit of the public **Google Borg ClusterData 2019** dataset 
        (`google.com:google-cluster-data.clusterdata_2019_a`) was conducted in Phase 7Y/7Z:

        #### 1. Exact Source Tables & Projected Columns
        - **`instance_events`** (1,717,317,922 rows, 399.09 GiB):
          - Projected fields: `collection_id`, `instance_index`, `time`, `type` (4 fields × 12.80 GiB/col = 51.2 GiB).
        - **`instance_usage`** (7,575,500,668 rows, 2,104.7 GiB / 2.26 TB):
          - Projected fields: `collection_id`, `instance_index`, `start_time`, `end_time`, `average_usage.cpus`, `average_usage.memory` (6 fields × 56.44 GiB/col = 338.6 GiB).
        - **Total Measured Dry-Run Volume (Candidate A):** **`418,578,205,568 bytes` (418.58 GB / 389.83 GiB)**.

        #### 2. Partitioning Diagnostics & Why `WHERE` Filters Do Not Reduce Scan Cost
        - Inspection of BigQuery table metadata (`client.get_table`) confirmed that **both tables have 0 partition keys and 0 clustering fields**.
        - In BigQuery's columnar architecture, reading an unpartitioned table scans every projected column in its entirety regardless of row filters.
        - **Candidate C** evaluated restricting queries to 5 jobs and 1-hour windows; the dry run remained exactly **418.58 GB**, proving that `WHERE` filters yield 0% scan reduction on the public trace.

        #### 3. Candidate D Alternative Strategy Evaluation
        - **Candidate D** queried `collection_events` (20.8M rows), scanning only **2.52 GB ($0.0143 USD)**.
        - **Scientific Verdict:** *Rejected for runtime failure prediction*. Candidate D represents static admission-time job failure classification and contains **zero runtime CPU/memory telemetry streams**, violating scientific equivalence with the research paper.

        #### 4. Financial Cost & Quota Analysis
        - **Standard On-Demand BigQuery Billing:** $6.25 per TiB → **$2.38 USD** nominal cost for 418.58 GB.
        - **Active GCP Project Reality (`canvas-advice-464215-n1`):** `billingEnabled: false` (BigQuery Sandbox).
        - **Actual Billed Charge:** **`$0.00 USD`** (incurs zero dollar charges, drawing 38.07% against the 1 TiB free monthly Sandbox query quota).
        - **Public GCS Alternative:** Google hosts gzip chunks in `gs://clusterdata_2019_a/` (>2.4 TB raw compressed volume).

        #### 5. Safe Extraction Roadmap
        Extraction remains gated under the formal **Pre-Extraction Authorization Package** (`READY_FOR_OWNER_AUTHORIZATION_REVIEW`). 
        Physical extraction will only proceed upon explicit user confirmation to allocate Sandbox quota.
        """
    )

    st.markdown("---")

    # 5. Academic References & Citations
    st.markdown("### 📖 Formal Academic References")
    st.markdown(
        """
        1. **Adnan Iqbal, Bhavya Thakur, Kushagra Pandey, Dr. Kamaldeep (2026):**  
           *Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing.*  
           Target Research Specification underpinning this alignment program.
        
        2. **S. Aldomi, H. Suleiman, A. Shatnawi, and L. Alawneh (2026):**  
           *A deep learning approach for early prediction of task failures in cloud computing environments.*  
           Systems and Soft Computing, vol. 8, Art. no. 200442.
        
        3. **C. Reiss, J. Wilkes, and J. L. Hellerstein (2011):**  
           *Google cluster-usage traces: format and schema.*  
           Google Research Technical Report.
        
        4. **J. Wilkes (2020):**  
           *More Google cluster data! Google Cloud Platform.*  
           Google Research (Google Borg 2019 dataset distribution).
        
        5. **T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár (2017):**  
           *Focal Loss for Dense Object Detection.*  
           IEEE International Conference on Computer Vision (ICCV), pp. 2980–2988.
        """
    )
