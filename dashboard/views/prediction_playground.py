"""Prediction Playground View for Cloud Failure Detection Dashboard."""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.model_loader import ModelManager
from dashboard.utils.inference_pipeline import InferencePipeline
from dashboard.utils.data_loader import (
    load_aligned_pilot_data,
    load_candidate_a_val_data,
    get_sample_telemetry_csv,
    FEATURE_NAMES_42,
    FEATURE_NAMES_31,
)


def render_prediction_playground_view():
    dataset_mode = st.session_state.get("dataset_mode", "Local Borg 2011 — Available")

    st.markdown("## ⚡ Interactive Prediction Playground")

    if dataset_mode == "Google Cluster Trace 2019 — Access pending":
        st.warning("⚠️ **Google Cluster Trace 2019 Inference — Checkpoint Pending**")
        st.info(
            "**Scientific Safeguard Notice:** No model checkpoints trained on Google Cluster Trace 2019 telemetry exist yet. "
            "Real-time inference cannot be executed for this dataset context without verified weights.\n\n"
            "To test live CPU inference across verified models (Aligned Stacked LSTM, Compact Transformer, Baselines), "
            "switch the Dataset Selector in the sidebar to **Local Borg 2011 — Available**."
        )
        return

    st.markdown(
        "Execute live, genuine CPU inference across trained model checkpoints. "
        "Test 15-step chronological sequences ($W=15, d=42$) from Google Borg 2011 traces, "
        "upload custom telemetry tables, or evaluate the operational $k=3$ alert policy in real time."
    )

    mgr = ModelManager()

    col_model, col_mode = st.columns([1.2, 1])

    with col_model:
        selected_model = st.selectbox(
            "Select Machine Learning Architecture",
            mgr.get_available_models(),
            index=0,
            help="Models execute locally on CPU using actual saved checkpoint weights.",
        )
        model_info = mgr.get_model_info(selected_model)
        exp_w = model_info.get("expected_seq_len", 15)
        exp_d = model_info.get("expected_features", 42)
        st.caption(f"**Origin:** {model_info.get('phase')} | **Input Spec:** W={exp_w} steps, d={exp_d} features")

    with col_mode:
        input_mode = st.radio(
            "Select Telemetry Input Source",
            [
                f"Built-In Aligned Task (W=15, d=42)",
                "Built-In Legacy Task (L=5, d=31)",
                "Upload Custom Telemetry CSV",
                "Default 15-Step Sample Telemetry",
            ],
            index=0 if exp_d == 42 else 1,
        )

    st.markdown("---")

    X_to_predict = None
    task_label_truth = None
    ttf_truth = None
    task_entity_id = "N/A"
    pipeline_mode = "aligned" if exp_d == 42 else "legacy"
    pipeline = InferencePipeline(mode=pipeline_mode)

    # 1. Built-In Aligned Task (W=15, d=42)
    if "Built-In Aligned Task" in input_mode:
        metadata, x_val, y_val = load_aligned_pilot_data("val")
        if metadata is not None and x_val is not None:
            entity_list = sorted(list(metadata["entity_id"].unique()))
            col_ent, col_win = st.columns(2)
            with col_ent:
                chosen_entity = st.selectbox("Select Task Entity ID", entity_list, index=0)
            
            ent_indices = metadata[metadata["entity_id"] == chosen_entity].index.tolist()
            with col_win:
                chosen_window_idx = st.slider(
                    f"Observation Window Index (Total: {len(ent_indices)})",
                    min_value=0,
                    max_value=max(0, len(ent_indices) - 1),
                    value=0,
                )
            
            selected_idx = ent_indices[chosen_window_idx]
            X_to_predict = x_val[selected_idx : selected_idx + 1]  # Shape (1, 15, 42)
            task_label_truth = int(y_val[selected_idx])
            ttf_val = metadata.loc[selected_idx, "time_to_failure"]
            ttf_truth = float(ttf_val) if not pd.isna(ttf_val) else None
            task_entity_id = chosen_entity

            st.info(
                f"📌 Selected Entity: `{chosen_entity}` | Ground Truth Label: "
                f"`{mgr.CLASS_NAMES[task_label_truth]}` "
                f"({f'Time to failure: {ttf_truth/60.0:.1f} min' if ttf_truth else 'Normal / Completed Task'})"
            )

    # 2. Built-In Legacy Task (L=5, d=31)
    elif "Built-In Legacy Task" in input_mode:
        metadata, x_val, y_val = load_candidate_a_val_data()
        if metadata is not None and x_val is not None:
            entity_list = sorted(list(metadata["entity_id"].unique()))
            col_ent, col_win = st.columns(2)
            with col_ent:
                chosen_entity = st.selectbox("Select Legacy Entity ID", entity_list, index=0)
            ent_indices = metadata[metadata["entity_id"] == chosen_entity].index.tolist()
            with col_win:
                chosen_window_idx = st.slider("Window Index", 0, max(0, len(ent_indices) - 1), 0)
            selected_idx = ent_indices[chosen_window_idx]
            X_to_predict = x_val[selected_idx : selected_idx + 1]  # Shape (1, 5, 31)
            task_label_truth = int(y_val[selected_idx])
            task_entity_id = chosen_entity
            st.info(f"📌 Legacy Entity: `{chosen_entity}` | Ground Truth: `{mgr.CLASS_NAMES[task_label_truth]}`")

    # 3. Upload Custom Telemetry CSV
    elif input_mode == "Upload Custom Telemetry CSV":
        uploaded_file = st.file_uploader(
            f"Upload Multivariate Telemetry CSV ({'42 features' if exp_d == 42 else '31 features'})",
            type=["csv"],
        )
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                valid, err, x_proc, rep = pipeline.validate_and_preprocess_dataframe(raw_df, auto_scale=True)
                if valid and x_proc is not None:
                    st.success(f"CSV Validated: {rep['total_input_rows']} rows formatted into {rep['evaluated_windows']} windows.")
                    X_to_predict = x_proc
                else:
                    st.error(f"Validation Error: {err}")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        else:
            st.info("Upload a CSV file or download the template below.")
            sample_csv_str = get_sample_telemetry_csv(version="aligned_w15_d42" if exp_d == 42 else "legacy_l5_d31")
            st.download_button(
                "📥 Download Sample Telemetry CSV Template",
                data=sample_csv_str,
                file_name=f"sample_telemetry_w{exp_w}_d{exp_d}.csv",
                mime="text/csv",
            )

    # 4. Default Sample
    elif "Default" in input_mode:
        sample_csv_str = get_sample_telemetry_csv(version="aligned_w15_d42" if exp_d == 42 else "legacy_l5_d31")
        if sample_csv_str:
            sample_df = pd.read_csv(io.StringIO(sample_csv_str))
            valid, err, x_proc, rep = pipeline.validate_and_preprocess_dataframe(sample_df, auto_scale=False)
            if valid:
                X_to_predict = x_proc
                st.success(f"Loaded default sample sequence of {rep['evaluated_windows']} window(s).")

    # Run Prediction If Data Ready
    if X_to_predict is not None:
        try:
            preds, probs = mgr.predict(selected_model, X_to_predict)
            top_pred = int(preds[0])
            top_probs = probs[0]
            confidence = float(top_probs[top_pred]) * 100.0

            st.markdown("### 🎯 Inference Results")
            c_badge, c_bars, c_kpolicy = st.columns([1, 1.2, 1])

            with c_badge:
                class_colors = {"NORMAL": "#10b981", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}
                pred_label = mgr.CLASS_NAMES[top_pred]
                badge_color = class_colors.get(pred_label, "#3b82f6")

                st.markdown(
                    f"""
                    <div style="background-color: {badge_color}22; border: 2px solid {badge_color}; border-radius: 10px; padding: 20px; text-align: center;">
                        <h4 style="color: #9ca3af; margin: 0; font-size: 0.9rem;">MODEL PREDICTION</h4>
                        <h2 style="color: {badge_color}; margin: 8px 0; font-size: 2.2rem; font-weight: 800;">{pred_label}</h2>
                        <p style="color: #d1d5db; margin: 0; font-size: 1.1rem; font-weight: 600;">{confidence:.1f}% Confidence</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_bars:
                st.markdown("#### Probability Distribution")
                for c_idx, c_name in enumerate(mgr.CLASS_NAMES):
                    p_val = float(top_probs[c_idx])
                    bar_col = "#10b981" if c_idx == 0 else ("#f59e0b" if c_idx == 1 else "#ef4444")
                    st.write(f"**{c_name}:** {p_val*100:.1f}%")
                    st.progress(min(1.0, max(0.0, p_val)))

            with c_kpolicy:
                st.markdown("#### Operational Alert Policy ($k=3$)")
                # Evaluate consecutive alert policy on window predictions
                k_res = mgr.evaluate_k_consecutive_alert(preds, k=3)
                if k_res["alert_triggered"]:
                    st.markdown(
                        """
                        <div style="background-color: #ef444422; border: 1px solid #ef4444; border-radius: 8px; padding: 12px;">
                            <b style="color: #ef4444;">🚨 OPERATIONAL ALERT TRIGGERED</b><br>
                            <span style="font-size: 0.85rem; color: #fca5a5;">
                                Policy requirement met: Model flagged consecutive WARNING/CRITICAL states. Schedulers should initiate pre-failure checkpointing.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="background-color: #10b98122; border: 1px solid #10b981; border-radius: 8px; padding: 12px;">
                            <b style="color: #10b981;">✅ NO PERSISTENT ALERT</b><br>
                            <span style="font-size: 0.85rem; color: #a7f3d0;">
                                System operates normally. Consecutive failure proximity criteria not met.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # Telemetry Sequence Line Chart
            st.markdown("### 📈 Sequence Dynamics Across Sliding Window")
            # Extract sequence features for plotting (CPU, Memory, Burstiness, Headroom)
            seq_matrix = X_to_predict[0]  # Shape (seq_len, n_features)
            n_steps = seq_matrix.shape[0]
            step_nums = list(range(1, n_steps + 1))

            fig_seq = go.Figure()
            # Feature 0: cpu_utilization
            fig_seq.add_trace(go.Scatter(x=step_nums, y=seq_matrix[:, 0], mode="lines+markers", name="CPU Rate", line=dict(color="#3b82f6", width=2)))
            # Feature 1: memory_utilization
            fig_seq.add_trace(go.Scatter(x=step_nums, y=seq_matrix[:, 1], mode="lines+markers", name="Canonical Memory", line=dict(color="#10b981", width=2)))

            # If 42 features, plot burstiness (idx 32) and headroom (idx 34)
            if seq_matrix.shape[1] >= 42:
                fig_seq.add_trace(go.Scatter(x=step_nums, y=seq_matrix[:, 32], mode="lines+markers", name="CPU Burstiness", line=dict(color="#f59e0b", dash="dash")))
                fig_seq.add_trace(go.Scatter(x=step_nums, y=seq_matrix[:, 34], mode="lines+markers", name="Memory Headroom", line=dict(color="#ec4899", dash="dot")))

            fig_seq.update_layout(
                title=f"Sequence Telemetry Trajectory (Window Length = {n_steps} observations)",
                xaxis_title="Observation Step within Window (t - 14 to t)",
                yaxis_title="Normalized Telemetry Magnitude",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_seq, use_container_width=True)

        except Exception as e:
            st.error(f"Inference execution failed: {e}")
            import traceback
            st.code(traceback.format_exc())
