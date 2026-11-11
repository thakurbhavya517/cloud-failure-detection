"""Demonstration script verifying actual sliding window execution."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.ingestion.schema import FailureLabel
from src.labeling.window_generator import WindowGenerator
from src.labeling.target_generator import TargetLabelGenerator


def run_demonstration():
    print("=== SLIDING WINDOW EXECUTION VERIFICATION ===")

    # 1. Deterministic synthetic sequence of 8 observations for task 'task_test_01'
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

    # 2. Target labeling
    label_gen = TargetLabelGenerator(
        critical_horizon_sec=900.0,
        warning_horizon_sec=1800.0,
        trace_version="2011",
    )
    labeled_df = label_gen.label_telemetry_windows(raw_df, events)

    # 3. Window generation (L=5, stride=1)
    feature_cols = ["cpu_utilization", "memory_utilization"]
    win_gen = WindowGenerator(
        sequence_length=5,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )
    X, y, meta = win_gen.build_dataset_tensors(labeled_df, feature_cols)

    # Report results
    print(f"Configured Window Length: 5 steps (25 minutes)")
    print(f"Algorithm Stride: 1 step (5 minutes / 300 seconds)")
    print(f"Number of Generated Windows: {len(X)}")
    print(f"Output 3D Tensor Shape: {X.shape}")
    print(f"Target Label Vector Shape: {y.shape}")

    label_names = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL", -1: "UNKNOWN"}

    print("\n--- Window 1 ---")
    print(f"Entity ID: {meta.loc[0, 'entity_id']}")
    print(f"Prediction Timestamp: {meta.loc[0, 'prediction_timestamp']}s")
    print(f"Assigned Label: {y[0]} ({label_names.get(y[0])})")
    print(f"Time to Failure: {meta.loc[0, 'time_to_failure']}s")
    print("Features matrix:")
    print(X[0])

    print("\n--- Window 2 ---")
    print(f"Entity ID: {meta.loc[1, 'entity_id']}")
    print(f"Prediction Timestamp: {meta.loc[1, 'prediction_timestamp']}s")
    print(f"Assigned Label: {y[1]} ({label_names.get(y[1])})")
    print(f"Time to Failure: {meta.loc[1, 'time_to_failure']}s")
    print("Features matrix:")
    print(X[1])


if __name__ == "__main__":
    run_demonstration()

