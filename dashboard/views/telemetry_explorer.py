"""Telemetry Explorer View for Cloud Failure Detection Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.data_loader import (
    load_aligned_pilot_data,
    load_candidate_a_val_data,
    FEATURE_NAMES_42,
    FEATURE_NAMES_31,
)


def render_telemetry_explorer_view():
    dataset_mode = st.session_state.get("dataset_mode", "Local Borg 2011 — Available")

    st.markdown("## 📈 Production Telemetry Explorer")

    if dataset_mode == "Google Cluster Trace 2019 — Access pending":
        st.warning("⚠️ **Google Cluster Trace 2019 Telemetry — Ingestion Pending**")
        st.info(
            "**Scientific Safeguard Notice:** Raw 2019 telemetry resides in Google BigQuery "
            "(`google.com:google-cluster-data.clusterdata_2019_a`) and has not been downloaded to local disk. "
            "No synthetic or simulated 2019 telemetry is displayed here.\n\n"
            "To explore real multivariate telemetry distributions (42 features, 2,787 windows) from local cluster shards, "
            "switch the Dataset Selector in the sidebar to **Local Borg 2011 — Available**."
        )
        return

    st.markdown(
        "Interactively inspect multivariate time-series telemetry from genuine **Google Borg 2011 Pilot Traces**. "
        "Explore 15-step sequence dynamics, resource headroom, and actual failure proximity labels."
    )

    dataset_choice = st.radio(
        "Select Dataset Partition:",
        ["Aligned Chronological Cohort (W=15, d=42, 75-min Windows)", "Legacy Candidate A Validation (L=5, d=31, 25-min Windows)"],
        index=0,
        horizontal=True,
    )

    if "Aligned" in dataset_choice:
        metadata, x_val, y_val = load_aligned_pilot_data("val")
        feature_names = FEATURE_NAMES_42
        seq_len = 15
    else:
        metadata, x_val, y_val = load_candidate_a_val_data()
        feature_names = FEATURE_NAMES_31
        seq_len = 5

    if metadata is None or x_val is None or y_val is None:
        st.error("Telemetry artifacts could not be loaded. Ensure preprocessing has been completed.")
        return

    # Filter by task entity
    entity_list = sorted(list(metadata["entity_id"].unique()))

    col_filter1, col_filter2, col_filter3 = st.columns([1.2, 1, 1.2])

    with col_filter1:
        chosen_entity = st.selectbox("Select Task Entity ID", entity_list, index=0)

    ent_mask = metadata["entity_id"] == chosen_entity
    ent_df = metadata[ent_mask].reset_index(drop=True)
    ent_indices = metadata[ent_mask].index.tolist()

    with col_filter2:
        class_filter = st.selectbox(
            "Filter Windows by Label",
            ["All Labels", "Class 0: NORMAL", "Class 1: WARNING", "Class 2: CRITICAL"],
            index=0,
        )

    # Apply class filter
    if class_filter == "Class 0: NORMAL":
        ent_df = ent_df[ent_df["label"] == 0]
    elif class_filter == "Class 1: WARNING":
        ent_df = ent_df[ent_df["label"] == 1]
    elif class_filter == "Class 2: CRITICAL":
        ent_df = ent_df[ent_df["label"] == 2]

    with col_filter3:
        feature_choices = st.multiselect(
            "Select Telemetry Features to Plot",
            feature_names,
            default=[feature_names[0], feature_names[1], feature_names[32 if len(feature_names) >= 42 else 2]],
        )

    st.markdown("---")

    # Display Entity Overview Metrics
    total_ent_windows = len(metadata[ent_mask])
    is_failing = (metadata.loc[ent_indices, "label"] > 0).any() if len(ent_indices) > 0 else False
    raw_max_label = metadata.loc[ent_indices, "label"].max() if len(ent_indices) > 0 else 0
    max_label = int(raw_max_label) if pd.notna(raw_max_label) else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Entity ID", chosen_entity)
    with kpi2:
        st.metric("Total Windows Observed", total_ent_windows)
    with kpi3:
        st.metric(
            "Execution Fate",
            "TERMINATED / FAILURE" if is_failing else "COMPLETED / NORMAL",
            delta="Anomaly Detected" if is_failing else "Healthy",
            delta_color="inverse" if is_failing else "normal",
        )
    with kpi4:
        label_names = ["NORMAL", "WARNING", "CRITICAL"]
        safe_idx = min(max(max_label, 0), len(label_names) - 1)
        st.metric("Highest Severity", label_names[safe_idx])

    st.markdown("---")

    # Time-Series Trajectory Chart
    if not ent_indices:
        st.warning("No windows found matching the current entity and label filter.")
        return

    # Use first window of selected entity for 15-step trajectory visualization
    sample_win_idx = ent_indices[0]
    sample_tensor = x_val[sample_win_idx]  # Shape (seq_len, n_features)
    step_indices = list(range(1, seq_len + 1))

    fig_trace = go.Figure()
    for feat in feature_choices:
        f_idx = feature_names.index(feat)
        vals = sample_tensor[:, f_idx]
        fig_trace.add_trace(go.Scatter(
            x=step_indices,
            y=vals,
            mode="lines+markers",
            name=feat,
        ))

    fig_trace.update_layout(
        title=f"Multivariate Observation Window (Window Index {sample_win_idx}, Sequence Length = {seq_len} steps)",
        xaxis_title="Trailing Observation Step (1 = oldest, 15 = prediction time t)",
        yaxis_title="Normalized Telemetry Value",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trace, use_container_width=True)

    # Feature Inspection Table
    with st.expander("🔍 Inspect Full 42-Feature Matrix for Selected Window", expanded=False):
        inspect_df = pd.DataFrame(sample_tensor, columns=feature_names)
        inspect_df.insert(0, "Observation Step", step_indices)
        st.dataframe(inspect_df, use_container_width=True, hide_index=True)
