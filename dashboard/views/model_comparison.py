"""Model Comparison View for Cloud Failure Detection Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.data_loader import load_phase8_audit_metrics, OPERATIONAL_GATES


def render_model_comparison_view():
    st.markdown("## Model Performance & Operational Benchmarking")
    st.markdown(
        "Evaluate the performance of all 8 candidate architectures on **Candidate A Validation** "
        "(657 windows, 12 parent jobs, zero job overlap with training) against pre-registered operational gates."
    )

    # Benchmark Data Table
    benchmark_data = [
        {
            "Architecture": "Random Forest (M*)",
            "Phase": "Phase 7D",
            "Macro-F1": 0.3653,
            "Accuracy": 0.7306,
            "Episode Recall": 42.86,
            "Critical Recall": 23.64,
            "Entity FAR": 65.79,
            "Window FPR": 11.18,
            "ROC-AUC": 0.6748,
            "Operational Verdict": "FAIL (FAR too high)",
        },
        {
            "Architecture": "Temporal LSTM (Unweighted)",
            "Phase": "Phase 7D",
            "Macro-F1": 0.3068,
            "Accuracy": 0.8524,
            "Episode Recall": 0.00,
            "Critical Recall": 0.00,
            "Entity FAR": 0.00,
            "Window FPR": 0.00,
            "ROC-AUC": 0.6120,
            "Operational Verdict": "FAIL (Majority Collapse)",
        },
        {
            "Architecture": "Temporal LSTM (Weighted)",
            "Phase": "Phase 7E",
            "Macro-F1": 0.3409,
            "Accuracy": 0.7823,
            "Episode Recall": 10.00,
            "Critical Recall": 8.16,
            "Entity FAR": 16.33,
            "Window FPR": 6.79,
            "ROC-AUC": 0.6986,
            "Operational Verdict": "FAIL (Low Sensitivity)",
        },
        {
            "Architecture": "Compact Transformer",
            "Phase": "Phase 7D/7E",
            "Macro-F1": 0.3463,
            "Accuracy": 0.8128,
            "Episode Recall": 30.00,
            "Critical Recall": 10.20,
            "Entity FAR": 20.41,
            "Window FPR": 8.57,
            "ROC-AUC": 0.7223,
            "Operational Verdict": "FAIL (Low Sensitivity)",
        },
        {
            "Architecture": "Representation Hybrid",
            "Phase": "Phase 7D",
            "Macro-F1": 0.2880,
            "Accuracy": 0.6088,
            "Episode Recall": 80.95,
            "Critical Recall": 58.18,
            "Entity FAR": 57.89,
            "Window FPR": 18.57,
            "ROC-AUC": 0.7612,
            "Operational Verdict": "FAIL (Extreme Alarms)",
        },
        {
            "Architecture": "Relative Telemetry (S2)",
            "Phase": "Phase 7S",
            "Macro-F1": 0.3685,
            "Accuracy": 0.8356,
            "Episode Recall": 43.33,
            "Critical Recall": 22.45,
            "Entity FAR": 37.41,
            "Window FPR": 10.30,
            "ROC-AUC": 0.7845,
            "Operational Verdict": "FAIL (FAR too high)",
        },
        {
            "Architecture": "Linear Hybrid Blend (0.8 S2 + 0.2 S0)",
            "Phase": "Phase 7T",
            "Macro-F1": 0.3702,
            "Accuracy": 0.8412,
            "Episode Recall": 36.67,
            "Critical Recall": 17.01,
            "Entity FAR": 24.49,
            "Window FPR": 4.40,
            "ROC-AUC": 0.7890,
            "Operational Verdict": "FAIL (Low Sensitivity)",
        },
        {
            "Architecture": "Type-1 Mamdani Fuzzy FIS",
            "Phase": "Phase 7U",
            "Macro-F1": 0.3650,
            "Accuracy": 0.7915,
            "Episode Recall": 66.67,
            "Critical Recall": 27.21,
            "Entity FAR": 48.30,
            "Window FPR": 17.44,
            "ROC-AUC": 0.7410,
            "Operational Verdict": "FAIL (FAR too high)",
        },
    ]

    df_bench = pd.DataFrame(benchmark_data)

    st.dataframe(
        df_bench.style.format({
            "Macro-F1": "{:.4f}",
            "Accuracy": "{:.2%}",
            "Episode Recall": "{:.1f}%",
            "Critical Recall": "{:.1f}%",
            "Entity FAR": "{:.1f}%",
            "Window FPR": "{:.1f}%",
            "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # Interactive Plots Tabs
    tab1, tab2, tab3 = st.tabs([
        "Operational Gates Trade-Off",
        "Simpson's Paradox (Relative vs Absolute)",
        "Phase 6 Confirmatory Holdout",
    ])

    with tab1:
        st.markdown("### The Sensitivity-Specificity Dead Zone")
        st.markdown(
            "Plotting **Episode Recall (Gate 1)** against **Pure-Normal Entity False Alarm Rate (Gate 2)**. "
            "To qualify as an operational winner, a model must land in the green highlighted region."
        )

        fig_tradeoff = go.Figure()

        # Add Success Zone Box (Recall >= 70%, FAR <= 10%)
        fig_tradeoff.add_shape(
            type="rect",
            x0=0,
            y0=70,
            x1=10,
            y1=100,
            fillcolor="rgba(34, 197, 94, 0.2)",
            line=dict(color="rgba(34, 197, 94, 0.8)", width=2, dash="dot"),
        )
        fig_tradeoff.add_annotation(
            x=5,
            y=85,
            text="Operational Success Zone<br>(Gates 1 & 2)",
            showarrow=False,
            font=dict(color="green", size=12),
        )

        # Plot models
        for item in benchmark_data:
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=[item["Entity FAR"]],
                    y=[item["Episode Recall"]],
                    mode="markers+text",
                    name=item["Architecture"],
                    text=[item["Architecture"]],
                    textposition="top center",
                    marker=dict(size=14, line=dict(width=1.5, color="black")),
                )
            )

        # Add Threshold Lines
        fig_tradeoff.add_hline(y=70, line_dash="dash", line_color="#3b82f6", annotation_text="Gate 1: Recall ≥ 70%")
        fig_tradeoff.add_vline(x=10, line_dash="dash", line_color="#ef4444", annotation_text="Gate 2: FAR ≤ 10%")

        fig_tradeoff.update_layout(
            title="Recall vs. False Alarm Rate (Candidate A Validation)",
            xaxis_title="Pure-Normal Entity False Alarm Rate (%) [Lower is Better]",
            yaxis_title="Episode Recall (%) [Higher is Better]",
            xaxis=dict(range=[-2, 75]),
            yaxis=dict(range=[-5, 105]),
            height=550,
            showlegend=False,
        )

        st.plotly_chart(fig_tradeoff, use_container_width=True)

    with tab2:
        st.markdown("### Simpson's Paradox: Workload Confounding Resolution")
        st.markdown(
            "In multi-tenant clusters, heavy batch workloads (e.g., job `6251277720` with 940 healthy windows) "
            "run at sustained high utilization. Absolute models ($S_0$) flagged this job as failing on **90.74% of windows**. "
            "Task-relative Median/MAD normalization ($S_2$) collapsed this false alarm rate to **6.10%**."
        )

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            fig_job = go.Figure(
                data=[
                    go.Bar(
                        x=["Absolute Telemetry (S0)", "Combined Telemetry (S1)", "Relative Telemetry (S2)"],
                        y=[90.74, 90.11, 6.10],
                        marker_color=["#ef4444", "#f97316", "#22c55e"],
                        text=["90.74%", "90.11%", "6.10%"],
                        textposition="auto",
                    )
                ]
            )
            fig_job.add_hline(y=10.0, line_dash="dot", line_color="gray", annotation_text="Operational Ceiling (10%)")
            fig_job.update_layout(
                title="False Alarm Rate on Healthy Batch Job 6251277720",
                yaxis_title="False Alarm Rate (%)",
                height=400,
            )
            st.plotly_chart(fig_job, use_container_width=True)

        with col_s2:
            fig_auc = go.Figure(
                data=[
                    go.Bar(
                        name="Train Out-of-Fold",
                        x=["S0 (Absolute)", "S1 (Combined)", "S2 (Relative)"],
                        y=[0.3066, 0.3139, 0.5080],
                        marker_color="#3b82f6",
                    ),
                    go.Bar(
                        name="Validation",
                        x=["S0 (Absolute)", "S1 (Combined)", "S2 (Relative)"],
                        y=[0.4244, 0.4435, 0.5864],
                        marker_color="#8b5cf6",
                    ),
                ]
            )
            fig_auc.add_hline(y=0.50, line_dash="dash", line_color="red", annotation_text="Random Guess (0.50)")
            fig_auc.update_layout(
                title="Cross-Job Failure Ranking (Binary ROC-AUC)",
                yaxis_title="ROC-AUC",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig_auc, use_container_width=True)

    with tab3:
        st.markdown("### Historical Phase 6 Confirmatory Holdout Results")
        st.markdown(
            "Blinded evaluation on 5,511 quarantined windows (335 entities, 141 failure episodes). "
            "Random Forest decisively outperformed deep sequential networks in Macro-F1, but all models failed operational ceilings."
        )

        p6_data = [
            {"Model": "Random Forest", "Macro-F1": 0.6111, "Accuracy": 78.06, "Window FPR": 21.39, "Entity FAR": 35.35},
            {"Model": "Temporal LSTM", "Macro-F1": 0.4425, "Accuracy": 70.95, "Window FPR": 26.88, "Entity FAR": 41.92},
            {"Model": "Compact Transformer", "Macro-F1": 0.4479, "Accuracy": 76.21, "Window FPR": 20.27, "Entity FAR": 31.82},
            {"Model": "Representation Hybrid", "Macro-F1": 0.3924, "Accuracy": 64.09, "Window FPR": 34.65, "Entity FAR": 64.14},
        ]
        df_p6 = pd.DataFrame(p6_data)
        st.dataframe(df_p6, use_container_width=True)
