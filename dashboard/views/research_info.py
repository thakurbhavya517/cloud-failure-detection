"""Research Information & Provenance View for Cloud Failure Detection Dashboard."""

import streamlit as st


def render_research_info_view():
    st.markdown("## Research Information & Theoretical Methodology")
    st.markdown(
        "Scientific foundation, empirical breakthroughs, methodological principles, "
        "and formal citations for the predictive container anomaly detection framework."
    )

    # 1. Research Objectives & Problem Statement
    st.markdown("### Core Research Objectives")
    st.markdown(
        """
        In multi-tenant hyperscale cloud environments (e.g., Google Borg, Kubernetes), containerized workloads 
        experience sudden terminations, evictions, out-of-memory (OOM) kills, and task failures. 
        Traditional anomaly detection relies on reactive thresholding, alerting operators only after service disruption.
        
        This research program investigates whether **multivariate time-series telemetry** can predict 
        runtime container failures **ahead of occurrence** across heterogeneous cloud workloads without generating 
        an unacceptable volume of operational false alarms.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            """
            **Primary Scientific Inquiries:**
            - Can deep sequential architectures (LSTM, Transformers) outperform tree-based tabular baselines (Random Forest) on short cloud telemetry windows?
            - Does task-relative normalization ($S_2$) resolve Simpson's Paradox across mixed batch vs. serving workloads?
            - Can hybrid representation or fuzzy decision layers break the sensitivity-specificity dead zone?
            """
        )
    with col2:
        st.success(
            """
            **Pre-Registered Operational Targets:**
            - **Gate 1a (Episode Recall):** $\\ge 70.0\\%$ of failure episodes detected prior to event.
            - **Gate 1b (Critical Recall):** $\\ge 50.0\\%$ of terminal-stage windows flagged.
            - **Gate 2a (Entity False Alarm Rate):** $\\le 10.0\\%$ on pure-normal tasks.
            - **Gate 2b (Window False Positive Rate):** $\\le 5.0\\%$ across all healthy windows.
            """
        )

    st.markdown("---")

    # 2. Key Empirical Findings & Methodological Breakthroughs
    st.markdown("### Key Scientific Discoveries")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Sequence Length Feasibility",
        "2. Post-Failure Censoring Leakage",
        "3. Simpson's Paradox & Relative Telemetry",
        "4. Tabular vs. Deep Learning",
    ])

    with tab1:
        st.markdown("#### Sequence Length Feasibility: The Collapse of $L=15$")
        st.markdown(
            """
            Early cloud telemetry literature commonly posits long temporal contexts (e.g., 15 to 30 steps of 5-minute telemetry). 
            However, rigorous empirical audit of Google Borg traces revealed that **short-lived containers represent over 60% of tasks**.
            
            - **Finding:** Requiring a history of $L=15$ timesteps caused **catastrophic sequence dropout**, discarding the vast majority of failing tasks before prediction could occur.
            - **Resolution:** Established **$L=5$ (25 minutes)** as the operational Pareto frontier, maximizing task retention while providing sufficient temporal gradient context.
            """
        )

    with tab2:
        st.markdown("#### Post-Failure Censoring Leakage: Eliminating Data Artifacts")
        st.markdown(
            """
            A widespread vulnerability in cloud telemetry research is **post-failure leakage**: models learning 
            anomalies from telemetry logged *after* an eviction, kill signal, or task restart has already occurred.
            
            - **Audit Protocol:** All time-series windows were strictly censored at the first failure event timestamp ($t_0$).
            - **Impact:** Eliminates synthetic 99%+ accuracy illusions, ensuring the models predict impending degradation rather than post-mortem crash artifacts.
            """
        )

    with tab3:
        st.markdown("#### Simpson's Paradox: The Workload Confounding Resolution")
        st.markdown(
            """
            When evaluating absolute resource features ($S_0$, e.g., raw CPU $> 85\\%$), massive batch jobs 
            (such as Job `6251277720` with 940 healthy tasks) were continuously flagged as failing (**90.74% False Alarm Rate**).
            
            - **The Paradox:** At the cluster-aggregate level, high CPU correlates with failure; inside high-utilization batch workloads, high CPU is standard normal operating behavior.
            - **Relative Telemetry ($S_2$):** By computing robust Median/MAD deviations relative to task-lifetime baselines, Job `6251277720` false alarm rate dropped from **90.74% down to 6.10%**, proving that relative feature engineering resolves cross-workload confounding.
            """
        )

    with tab4:
        st.markdown("#### Tabular Ensembles vs. Deep Neural Networks")
        st.markdown(
            """
            Across both Phase 6 confirmatory holdout and Phase 7 evaluations, **Random Forest consistently outperformed or matched deep sequential networks** (Macro-F1 0.6111 vs LSTM 0.4425 and Transformer 0.4479 on Phase 6).
            
            - **Insight:** On short 5-step sequences with high noise-to-signal ratios, tree-based models with engineered domain features (lags, rolling stats, trend slopes) provide superior inductive bias compared to unconstrained self-attention and recurrent state transitions.
            """
        )

    st.markdown("---")

    # 3. System Architecture & Hardware Feasibility
    st.markdown("### System Architecture & Operational Profile")
    st.markdown(
        """
        The production inference pipeline and interactive dashboard are specifically engineered for 
        **100% CPU execution on commodity hardware** (e.g., Intel Core i5 / 8 GB RAM), requiring zero GPU acceleration:
        
        - **Preprocessing Engine:** Sub-millisecond NumPy vectorized sliding-window slicer with fitted median imputation.
        - **Inference Runtime:** PyTorch CPU (`torch.device("cpu")`) and scikit-learn threaded predictors.
        - **Single-Window Latency:** $< 1.8 \\text{ ms}$ on CPU for Random Forest; $< 8.5 \\text{ ms}$ for Compact Transformer.
        """
    )

    st.markdown("---")

    # 4. Citation & Academic Reference
    st.markdown("### Academic Citation")
    st.markdown("If you use this research repository, models, or empirical findings in your work, please cite:")

    bibtex_citation = """@software{cloud_failure_detection_2026,
  author = {Thakur, Bhavya and Research Contributors},
  title = {Multivariate Time-Series Telemetry for Container Anomaly and Failure Prediction in Hyperscale Cloud Clusters},
  year = {2026},
  url = {https://github.com/thakurbhavya517/cloud-failure-detection},
  note = {Research phases 4-8 covering Google Borg 2011/2019, Relative Telemetry, and Operational Gate Benchmarks}
}"""

    st.code(bibtex_citation, language="bibtex")
