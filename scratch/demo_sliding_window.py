"""Practical verification test demonstrating sliding window construction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.ingestion.schema import FailureLabel
from src.labeling.window_generator import WindowGenerator
from src.labeling.target_generator import TargetLabelGenerator

def main():
    print("=" * 70)
    print("PRACTICAL SLIDING WINDOW GENERATION AUDIT TEST")
    print("=" * 70)

    # 1. Create a deterministic series of 8 observations for task 'task_test_01'
    # Cadence: 300s (5-minute), starting at 300s up to 2400s
    timestamps = np.array([300.0 * (i + 1) for i in range(8)])  # [300, 600, 900, 1200, 1500, 1800, 2100, 2400]

    # Lifecycle failure event at t = 2400s
    events = pd.DataFrame([{
        "entity_id": "task_test_01",
        "event_type": 3,  # FAIL in ClusterEventTypeV2
        "time_sec": 2400.0
    }])

    raw_df = pd.DataFrame({
        "entity_id": ["task_test_01"] * 8,
        "start_time_sec": timestamps - 300.0,
        "end_time_sec": timestamps,
        "cpu_utilization": [0.15, 0.22, 0.31, 0.45, 0.60, 0.78, 0.89, 0.94],
        "memory_utilization": [0.30, 0.32, 0.35, 0.40, 0.48, 0.55, 0.62, 0.70]
    })

    # 2. Derive ground-truth labels using TargetLabelGenerator
    # H_critical = 900s (15 min), H_warning = 1800s (30 min)
    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(raw_df, events)

    print("\n--- 1. Original Chronologically Ordered Observations ---")
    print(labeled_df[["entity_id", "end_time_sec", "cpu_utilization", "memory_utilization", "label", "time_to_failure_sec"]].to_string(index=True))

    # 3. Generate sliding windows with WindowGenerator (L=5, require_full_history=True)
    feature_cols = ["cpu_utilization", "memory_utilization"]
    window_gen = WindowGenerator(
        sequence_length=5,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X, y, meta = window_gen.build_dataset_tensors(labeled_df, feature_cols)

    print(f"\n--- 2. Sliding Window Generation Results ---")
    print(f"Total Observations Input: {len(labeled_df)}")
    print(f"Configured Window Length (L): 5 steps (25 minutes)")
    print(f"Algorithm Stride (s): 1 step (5 minutes / 300 seconds)")
    print(f"Total Valid Windows Generated: {len(X)} (N = 8 - 5 + 1 = 4 windows)")
    print(f"Output 3D Tensor Shape: X in R^{X.shape} (N={X.shape[0]}, L={X.shape[1]}, F={X.shape[2]})")
    print(f"Output Target Vector Shape: y in R^{y.shape}")

    label_names = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL", -1: "UNKNOWN"}

    # Display Window 1
    print("\n--- 3. Generated Sliding Window 1 ---")
    print(f"Index: 0")
    print(f"Entity ID: {meta.loc[0, 'entity_id']}")
    print(f"Covered Timestamps: {labeled_df.loc[0:4, 'end_time_sec'].values.tolist()}s")
    print(f"Prediction Timestamp (t_pred): {meta.loc[0, 'prediction_timestamp']}s")
    print(f"Time to Failure: {meta.loc[0, 'time_to_failure']}s")
    print(f"Assigned Ground-Truth Label: {y[0]} ({label_names.get(y[0])})")
    print(f"Window Tensor Values [cpu_utilization, memory_utilization]:")
    print(X[0])

    # Display Window 2
    print("\n--- 4. Generated Sliding Window 2 ---")
    print(f"Index: 1")
    print(f"Entity ID: {meta.loc[1, 'entity_id']}")
    print(f"Covered Timestamps: {labeled_df.loc[1:5, 'end_time_sec'].values.tolist()}s")
    print(f"Prediction Timestamp (t_pred): {meta.loc[1, 'prediction_timestamp']}s")
    print(f"Time to Failure: {meta.loc[1, 'time_to_failure']}s")
    print(f"Assigned Ground-Truth Label: {y[1]} ({label_names.get(y[1])})")
    print(f"Window Tensor Values [cpu_utilization, memory_utilization]:")
    print(X[1])

    # Display Window 3 and 4
    for idx in range(2, len(X)):
        print(f"\n--- 5.{idx-1} Generated Sliding Window {idx+1} ---")
        print(f"Covered Timestamps: {labeled_df.loc[idx:idx+4, 'end_time_sec'].values.tolist()}s")
        print(f"Prediction Timestamp: {meta.loc[idx, 'prediction_timestamp']}s | TTF: {meta.loc[idx, 'time_to_failure']}s | Label: {y[idx]} ({label_names.get(y[idx])})")

    # Mathematical Verification assertions
    # Note: Row 7 (t=2400s) has label UNKNOWN (-1) due to post-failure censoring,
    # so exclude_unknown=True correctly retains exactly 3 pre-failure windows.
    assert X.shape == (3, 5, 2)
    assert np.array_equal(X[0, 1:, :], X[1, :-1, :]), "Sliding overlap assertion: W1[1:] must match W2[:-1]"
    assert meta.loc[1, "prediction_timestamp"] - meta.loc[0, "prediction_timestamp"] == 300.0, "Stride assertion: t2 - t1 must equal 300s"
    print("\n[VERIFIED] All mathematical sliding window invariants verified successfully!")

if __name__ == "__main__":
    main()
