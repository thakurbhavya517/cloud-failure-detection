"""Main entry point for Cloud Failure Detection Research Dashboard.

Streamlit multi-page application integrating aligned research models, real telemetry data,
and audited experimental artifacts.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.views.home import render_home_view
from dashboard.views.model_comparison import render_model_comparison_view
from dashboard.views.telemetry_explorer import render_telemetry_explorer_view
from dashboard.views.prediction_playground import render_prediction_playground_view
from dashboard.views.evaluation_testing import render_evaluation_testing_view
from dashboard.views.research_info import render_research_info_view

# Configure page layout and metadata
st.set_page_config(
    page_title="Cloud Failure Detection | ML Research Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Card-like appearance for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Header subtitle styling */
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    
    /* Sidebar header */
    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .sidebar-status-box {
        background-color: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 10px;
        margin-top: 15px;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-brand">☁️ Cloud Failure ML</div>', unsafe_allow_html=True)
    st.caption("Target Paper Aligned Dashboard • v3.1")
    st.markdown("---")

    # Explicit Dataset Mode Selector (Task 3)
    st.markdown("### 🗄️ Dataset Selector")
    current_mode = st.session_state.get("dataset_mode", "Local Borg 2011 — Available")
    default_idx = 0 if "2011" in current_mode else 1
    dataset_mode = st.radio(
        "Active Dataset Context",
        [
            "Local Borg 2011 — Available",
            "Google Cluster Trace 2019 — Access pending",
        ],
        index=default_idx,
        key="dataset_mode_selector",
        help="Select active dataset context. Local Borg 2011 is available on disk; Google 2019 access is pending BigQuery extraction approval.",
    )
    st.session_state["dataset_mode"] = dataset_mode

    st.markdown("---")

    nav_selection = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "📊 Model Comparison",
            "📈 Telemetry Explorer",
            "⚡ Prediction Playground",
            "🧪 Evaluation & Testing",
            "📚 Research Information",
        ],
        index=0,
    )

    st.markdown("---")

    # Operational Environment Information
    st.markdown("### Operational Environment")
    if dataset_mode == "Local Borg 2011 — Available":
        st.markdown(
            """
            <div class="sidebar-status-box">
                <b>Compute Engine:</b> CPU Execution<br>
                <b>Target Architecture:</b> W=15, d=42<br>
                <b>Active Models:</b> 128h LSTM & 64d Trans<br>
                <b>Validation Partition:</b> 724 Windows (30m Embargo)<br>
                <b>Dataset:</b> Google Borg 2011 (Available)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-status-box" style="border-color: #f59e0b; background-color: rgba(245, 158, 11, 0.1);">
                <b>Compute Engine:</b> CPU Execution<br>
                <b>Dataset:</b> Google Cluster Trace 2019<br>
                <b>Status:</b> ⏳ ACCESS PENDING<br>
                <b>Source:</b> BigQuery clusterdata_2019_a<br>
                <b>Models:</b> Ingestion Authorization Required
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "🔗 **GitHub Repo:** [thakurbhavya517/cloud-failure-detection](https://github.com/thakurbhavya517/cloud-failure-detection)"
    )

# Page Routing
if nav_selection == "🏠 Home Dashboard":
    render_home_view()
elif nav_selection == "📊 Model Comparison":
    render_model_comparison_view()
elif nav_selection == "📈 Telemetry Explorer":
    render_telemetry_explorer_view()
elif nav_selection == "⚡ Prediction Playground":
    render_prediction_playground_view()
elif nav_selection == "🧪 Evaluation & Testing":
    render_evaluation_testing_view()
elif nav_selection == "📚 Research Information":
    render_research_info_view()
