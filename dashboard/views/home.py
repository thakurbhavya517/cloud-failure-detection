"""Home Dashboard View for Cloud Failure Detection Dashboard."""

import streamlit as st
from dashboard.utils.data_loader import load_phase8_audit_metrics, load_candidate_a_val_data


def render_home_view():
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #1e2638 0%, #111827 100%); border-radius: 10px; border: 1px solid #374151; margin-bottom: 2rem;">
        <h1 style="color: #f3f4f6; margin-bottom: 0.5rem; font-size: 2.2rem;">Cloud Workload Early Failure Detection</h1>
        <p style="color: #9ca3af; font-size: 1.1rem; margin-bottom: 0;">
            A Forensic Machine Learning Research Dashboard evaluating deep temporal sequence models (LSTM, Transformer, Hybrid) 
            versus classical tabular ensembles on production Google Borg cluster telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Research Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Evaluated Phases", value="8 Phases", delta="Phases 1 → 8", delta_color="normal")
    with col2:
        st.metric(label="Total Windows Studied", value="11,000+", delta="335+ Entities", delta_color="normal")
    with col3:
        st.metric(label="Best Offline Macro-F1", value="0.6111", delta="Random Forest Baseline", delta_color="normal")
    with col4:
        st.metric(label="Operational Verdict", value="NO WINNER", delta="Strict Reliability Gates", delta_color="inverse")

    st.markdown("---")

    # Core Scientific Takeaways
    st.markdown("### Executive Research Takeaways")
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown(r"""
        > **Scientific Headline: A Masterclass in Empirical Rigor & Negative Results**
        > 
        > This project investigated whether deep sequence models (LSTM, Transformer) outperform classical tree models for proactive cloud failure prediction.
        > 
        > - **Classical Baseline Dominance:** Random Forest outperformed deep sequence models in offline multi-class Macro-F1 ($0.611$ vs $0.443 - 0.448$).
        > - **Universal Operational Failure:** Under real-world class imbalance ($98:1:1$), all models failed pre-registered operational ceilings ($\le 5\%$ Window FPR and $\le 10\%$ Entity FAR), generating false alarms on $17\% - 64\%$ of healthy workloads.
        > - **Simpson's Paradox Discovery:** Absolute telemetry models memorized parent-job footprints rather than failure dynamics, triggering false alarms on $>90\%$ of healthy high-resource batch jobs.
        > - **Relative Telemetry Breakthrough:** Task-relative Median/MAD normalization ($S_2$) slashed false alarms on healthy batch jobs from $90.7\%$ to $6.1\%$ and improved OOF ROC-AUC by $+0.2014$.
        """)

    with c_right:
        st.markdown("#### Dataset Provenance & Partition Status")
        st.info("""
        **Candidate A (Governing Clean Split):**
        - **Train:** 3,278 windows (51 parent jobs) — *Active Local*
        - **Validation:** 657 windows (12 parent jobs) — *Active Local*
        - **Test:** 1,025 windows (18 parent jobs) — *Quarantined*
        
        **Protected Partitions:**
        - **Phase 6 Holdout:** 5,511 windows — *Spent & Quarantined*
        - **Google Borg 2019:** *Paused (`DATA_ACCESS_BLOCKED`)*
        """)

    st.markdown("---")
    st.markdown("### Repository Navigation Guide")

    nav_c1, nav_c2, nav_c3 = st.columns(3)
    with nav_c1:
        st.markdown("""
        **Model Comparison**  
        Compare evaluated architectures side-by-side across offline classification metrics, operational gates, and ROC/PR trade-offs.
        """)
    with nav_c2:
        st.markdown("""
        **Telemetry Explorer**  
        Visually inspect multivariate time-series telemetry (CPU, Memory, CPI, Disk I/O) across genuine Borg cluster tasks.
        """)
    with nav_c3:
        st.markdown("""
        **Prediction Playground**  
        Upload custom telemetry CSVs or select validation tasks to run live CPU inference through genuine trained models.
        """)
