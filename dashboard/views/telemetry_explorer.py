"""Telemetry Explorer View for Cloud Failure Detection Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.data_loader import load_candidate_a_val_data, load_feature_scaler_spec


def render_telemetry_explorer_view():
    st.markdown("## Production Telemetry Explorer")
    st.markdown(
        "Interactively inspect multivariate time-series telemetry from genuine **Google Borg Cluster Traces** "
        "(Candidate A Validation split). Visualizes resource utilization, sequence windows, and actual failure proximity labels."
    )

    metadata, x_val, y_val = load_candidate_a_val_data()
    scaler_spec = load_feature_scaler_spec()
    feature_cols = scaler_spec.get("feature_cols", [])

    if metadata is None or x_val is None or y_val is None:
        st.error("Validation telemetry artifacts could not be loaded from data/processed/phase7c/.")
        return

    # Sidebar / Top selection
    entity_list = sorted(list(metadata["entity_id"].unique()))
    
    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        selected_entity = st.selectbox("Select Cluster Task (Entity)", entity_list, index=0)

    # Filter metadata for this entity
    entity_mask = metadata["entity_id"] == selected_entity
    entity_meta = metadata[entity_mask].sort_values("prediction_timestamp").reset_index(drop=True)
    entity_indices = metadata[entity_mask].index.values

    with col_stat:
        total_wins = len(entity_meta)
        parent_job = entity_meta["parent_job_id"].iloc[0]
        has_fail = (entity_meta["label"] > 0).any()
        status_label = "Failing Task" if has_fail else "Pure-Normal Task"
        status_color = "red" if has_fail else "green"

        st.markdown(f"""
        **Entity ID:** `{selected_entity}` | **Parent Job ID:** `{parent_job}`  
        **Observation Windows:** `{total_wins}` | **Lifecycle Status:** :{status_color}[**{status_label}**]
        """)

    st.markdown("---")

    # Time series of predictions/labels
    st.markdown("### Telemetry Trajectory Across Observation Windows")

    # Build DataFrame of all windows for this entity
    records = []
    for row_idx, val_idx in enumerate(entity_indices):
        timestamp = entity_meta.loc[row_idx, "prediction_timestamp"]
        label = entity_meta.loc[row_idx, "label"]
        time_to_fail = entity_meta.loc[row_idx, "time_to_failure"]
        
        # Last step of 5-step window contains instantaneous features
        window_features = x_val[val_idx, -1, :]
        rec = {
            "window_index": row_idx,
            "prediction_timestamp": timestamp,
            "label": "NORMAL" if label == 0 else ("WARNING" if label == 1 else "CRITICAL"),
            "time_to_failure_sec": time_to_fail if pd.notna(time_to_fail) else None,
        }
        for f_idx, col_name in enumerate(feature_cols):
            rec[col_name] = window_features[f_idx]
        records.append(rec)

    df_entity = pd.DataFrame(records)

    # Plot metrics
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "CPU & Memory Telemetry",
        "Cycles Per Instruction & Disk I/O",
        "Raw Feature Table",
    ])

    with chart_tab1:
        fig_cpu_mem = go.Figure()
        
        # CPU
        if "cpu_utilization" in df_entity.columns:
            fig_cpu_mem.add_trace(go.Scatter(
                x=df_entity["prediction_timestamp"],
                y=df_entity["cpu_utilization"],
                mode="lines+markers",
                name="CPU Utilization",
                line=dict(color="#3b82f6", width=2.5),
            ))
        
        # Memory
        if "memory_utilization" in df_entity.columns:
            fig_cpu_mem.add_trace(go.Scatter(
                x=df_entity["prediction_timestamp"],
                y=df_entity["memory_utilization"],
                mode="lines+markers",
                name="Memory Utilization",
                line=dict(color="#10b981", width=2.5),
            ))

        # Add shaded background markers for Warning and Critical windows
        for _, row in df_entity.iterrows():
            if row["label"] == "CRITICAL":
                fig_cpu_mem.add_vrect(
                    x0=row["prediction_timestamp"] - 150,
                    x1=row["prediction_timestamp"] + 150,
                    fillcolor="rgba(239, 68, 68, 0.25)",
                    line_width=0,
                    annotation_text="Critical Window",
                    annotation_position="top left",
                )
            elif row["label"] == "WARNING":
                fig_cpu_mem.add_vrect(
                    x0=row["prediction_timestamp"] - 150,
                    x1=row["prediction_timestamp"] + 150,
                    fillcolor="rgba(249, 115, 22, 0.20)",
                    line_width=0,
                    annotation_text="Warning Window",
                    annotation_position="top left",
                )

        fig_cpu_mem.update_layout(
            title=f"CPU and Memory Dynamics: Task {selected_entity}",
            xaxis_title="Cluster Timestamp (seconds)",
            yaxis_title="Standardized Resource Level",
            height=450,
        )
        st.plotly_chart(fig_cpu_mem, use_container_width=True)

    with chart_tab2:
        fig_cpi_io = go.Figure()

        if "cycles_per_instruction" in df_entity.columns:
            fig_cpi_io.add_trace(go.Scatter(
                x=df_entity["prediction_timestamp"],
                y=df_entity["cycles_per_instruction"],
                mode="lines+markers",
                name="Cycles Per Instruction (CPI)",
                line=dict(color="#8b5cf6", width=2.5),
            ))

        if "disk_io_time" in df_entity.columns:
            fig_cpi_io.add_trace(go.Scatter(
                x=df_entity["prediction_timestamp"],
                y=df_entity["disk_io_time"],
                mode="lines+markers",
                name="Disk I/O Time",
                line=dict(color="#ec4899", width=2.5),
            ))

        fig_cpi_io.update_layout(
            title=f"Hardware Counters & I/O: Task {selected_entity}",
            xaxis_title="Cluster Timestamp (seconds)",
            yaxis_title="Standardized Counter Level",
            height=450,
        )
        st.plotly_chart(fig_cpi_io, use_container_width=True)

    with chart_tab3:
        st.dataframe(df_entity, use_container_width=True)
