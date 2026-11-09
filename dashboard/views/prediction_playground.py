"""Prediction Playground View for Cloud Failure Detection Dashboard."""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.model_loader import ModelManager
from dashboard.utils.inference_pipeline import InferencePipeline
from dashboard.utils.data_loader import load_candidate_a_val_data, get_sample_telemetry_csv


def render_prediction_playground_view():
    st.markdown("## Interactive Prediction Playground")
    st.markdown(
        "Execute genuine live CPU inference across trained model checkpoints. "
        "Upload custom telemetry CSV files or evaluate real-world tasks from Candidate A Validation."
    )

    mgr = ModelManager()
    pipeline = InferencePipeline()

    col_model, col_mode = st.columns([1.2, 1])

    with col_model:
        selected_model = st.selectbox(
            "Select Machine Learning Architecture",
            mgr.get_available_models(),
            index=0,
            help="All models execute locally on CPU using actual saved checkpoint weights.",
        )
        model_info = mgr.get_model_info(selected_model)
        st.caption(f"**Phase Origin:** {model_info.get('phase')} | {model_info.get('description')}")

    with col_mode:
        input_mode = st.radio(
            "Select Telemetry Input Source",
            ["Built-In Validation Task", "Upload Custom Telemetry CSV", "Use Default 5-Step Sample"],
            index=0,
        )

    st.markdown("---")

    X_to_predict = None
    input_df = None
    task_label_truth = None

    if input_mode == "Built-In Validation Task":
        metadata, x_val, y_val = load_candidate_a_val_data()
        if metadata is not None and x_val is not None:
            # Let user select entity
            entity_list = sorted(list(metadata["entity_id"].unique()))
            col_ent, col_win = st.columns(2)
            with col_ent:
                chosen_entity = st.selectbox("Select Task Entity", entity_list, index=0)

            ent_mask = metadata["entity_id"] == chosen_entity
            ent_indices = metadata[ent_mask].index.values
            ent_meta = metadata[ent_mask].reset_index(drop=True)

            with col_win:
                win_options = [f"Window #{i} (t={ent_meta.loc[i, 'prediction_timestamp']}s)" for i in range(len(ent_meta))]
                chosen_win_idx = st.selectbox("Select Observation Window", range(len(win_options)), format_func=lambda i: win_options[i])

            actual_val_idx = ent_indices[chosen_win_idx]
            X_to_predict = x_val[actual_val_idx : actual_val_idx + 1]  # shape (1, 5, 31)
            raw_label = y_val[actual_val_idx]
            task_label_truth = "NORMAL" if raw_label == 0 else ("WARNING" if raw_label == 1 else "CRITICAL")

            st.info(f"Loaded 5-step sequence for `{chosen_entity}` at Window #{chosen_win_idx}. Ground-Truth Target Label: **{task_label_truth}**")
        else:
            st.error("Validation dataset not found.")

    elif input_mode == "Use Default 5-Step Sample":
        sample_csv = get_sample_telemetry_csv()
        if sample_csv:
            input_df = pd.read_csv(io.StringIO(sample_csv))
            st.markdown("##### Preview of Default 5-Step Sequence:")
            st.dataframe(input_df.head(), use_container_width=True)

            st.download_button(
                label="📥 Download Template CSV",
                data=sample_csv,
                file_name="telemetry_sample_template.csv",
                mime="text/csv",
            )

            is_valid, err_msg, X_processed, rep = pipeline.validate_and_preprocess_dataframe(input_df, auto_scale=False)
            if is_valid:
                X_to_predict = X_processed
                st.success("Template CSV successfully validated against 31-feature schema.")
            else:
                st.error(err_msg)

    elif input_mode == "Upload Custom Telemetry CSV":
        st.markdown(
            "Upload a CSV file containing multivariate telemetry. "
            "The file must contain the 31 canonical telemetry features (e.g. `cpu_utilization`, `memory_utilization`, lags, and differences)."
        )

        sample_csv = get_sample_telemetry_csv()
        st.download_button(
            label="📥 Download Reference Schema CSV",
            data=sample_csv,
            file_name="telemetry_reference_schema.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])
        auto_scale = st.checkbox("Apply Standard Scaling (Check if data contains unscaled raw percentages)", value=False)

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.markdown("##### Uploaded Telemetry Preview:")
                st.dataframe(input_df.head(), use_container_width=True)

                is_valid, err_msg, X_processed, rep = pipeline.validate_and_preprocess_dataframe(input_df, auto_scale=auto_scale)
                if not is_valid:
                    st.error(f"Schema Validation Error: {err_msg}")
                else:
                    X_to_predict = X_processed
                    st.success(
                        f"Validation Passed: Formatted {rep['evaluated_windows']} sequence windows "
                        f"(Sequence Length: {rep['sequence_length']}, Features: {rep['feature_count']}, Imputed NaNs: {rep['imputed_nans']})."
                    )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

    st.markdown("---")

    # Run Prediction Button
    if X_to_predict is not None:
        if st.button("🚀 Run Live CPU Inference", type="primary", use_container_width=True):
            with st.spinner(f"Executing inference with {selected_model}..."):
                try:
                    preds, probs = mgr.predict(selected_model, X_to_predict)

                    st.markdown("### Inference Results")

                    if len(preds) == 1:
                        # Single window result view
                        pred_class_idx = preds[0]
                        class_names = ["NORMAL", "WARNING", "CRITICAL"]
                        pred_class = class_names[pred_class_idx]
                        p_norm, p_warn, p_crit = probs[0][0], probs[0][1], probs[0][2]

                        res_c1, res_c2, res_c3 = st.columns(3)
                        with res_c1:
                            if pred_class == "CRITICAL":
                                st.error("### 🔴 CRITICAL ALERT\nImpending failure detected (0–5 min horizon)")
                            elif pred_class == "WARNING":
                                st.warning("### 🟠 WARNING ALERT\nAbnormal degradation detected (10–25 min horizon)")
                            else:
                                st.success("### 🟢 NORMAL OPERATION\nHealthy task execution (>25 min horizon)")

                        with res_c2:
                            st.metric("Predicted Class", pred_class)
                            st.metric("P(Critical)", f"{p_crit:.4f}")

                        with res_c3:
                            if task_label_truth is not None:
                                is_match = pred_class == task_label_truth
                                match_color = "green" if is_match else "red"
                                st.metric("Ground-Truth Label", task_label_truth)
                                st.markdown(f"Verification: :{match_color}[{'MATCH' if is_match else 'MISMATCH'}]")

                        # Probability Bar Chart
                        fig_probs = go.Figure(data=[
                            go.Bar(
                                x=["NORMAL (Healthy)", "WARNING (Degraded)", "CRITICAL (Imminent Failure)"],
                                y=[p_norm, p_warn, p_crit],
                                marker_color=["#22c55e", "#f97316", "#ef4444"],
                                text=[f"{p_norm:.2%}", f"{p_warn:.2%}", f"{p_crit:.2%}"],
                                textposition="auto",
                            )
                        ])
                        fig_probs.update_layout(
                            title=f"Softmax Class Posterior Probabilities ({selected_model})",
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1.05]),
                            height=350,
                        )
                        st.plotly_chart(fig_probs, use_container_width=True)

                    else:
                        # Multi-window table view
                        results_list = []
                        class_names = ["NORMAL", "WARNING", "CRITICAL"]
                        for i in range(len(preds)):
                            results_list.append({
                                "window_idx": i,
                                "predicted_label": class_names[preds[i]],
                                "P(NORMAL)": round(float(probs[i][0]), 4),
                                "P(WARNING)": round(float(probs[i][1]), 4),
                                "P(CRITICAL)": round(float(probs[i][2]), 4),
                            })
                        df_res = pd.DataFrame(results_list)

                        st.dataframe(df_res, use_container_width=True)

                        # Download Predictions CSV
                        csv_data = df_res.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_data,
                            file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Inference execution failed: {str(e)}")
