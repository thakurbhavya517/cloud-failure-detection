"""Data and artifact loading utilities for the Cloud Failure Detection Dashboard.

Strictly protects quarantined holdouts while loading authorized Candidate A
validation partitions and audit evidence.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data" / "processed" / "phase7c"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

# Pre-registered Operational Gates
OPERATIONAL_GATES = {
    "Gate 1a (Episode Recall)": {"threshold": 70.0, "unit": "%", "direction": ">="},
    "Gate 1b (Critical Recall)": {"threshold": 50.0, "unit": "%", "direction": ">="},
    "Gate 2a (Pure-Normal Entity FAR)": {"threshold": 10.0, "unit": "%", "direction": "<="},
    "Gate 2b (Normal Window FPR)": {"threshold": 5.0, "unit": "%", "direction": "<="},
}


def load_phase8_audit_metrics() -> dict:
    """Load genuine audited metrics from Phase 8 research integrity audit."""
    p8_path = RESULTS_DIR / "phase8_research_integrity" / "phase8_dataset_integrity_audit.json"
    if p8_path.exists():
        with open(p8_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_candidate_a_val_data():
    """Load Candidate A validation arrays and metadata safely."""
    val_meta_path = DATA_DIR / "val_metadata.csv"
    x_val_path = DATA_DIR / "X_val.npy"
    y_val_path = DATA_DIR / "y_val.npy"

    if val_meta_path.exists() and x_val_path.exists() and y_val_path.exists():
        metadata = pd.read_csv(val_meta_path)
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        return metadata, x_val, y_val
    return None, None, None


def load_feature_scaler_spec():
    """Load fitted scaler configuration from Candidate A Train."""
    scaler_path = CONFIGS_DIR / "phase7c" / "fitted_scaler_phase7c.json"
    if scaler_path.exists():
        with open(scaler_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_sample_telemetry_csv() -> str:
    """Generate a valid sample CSV containing 5 time-steps of telemetry for 31 features."""
    _, x_val, _ = load_candidate_a_val_data()
    scaler_spec = load_feature_scaler_spec()
    feature_cols = scaler_spec.get("feature_cols", [])

    if x_val is not None and feature_cols:
        # Take the first window (5 time-steps, 31 features)
        sample_window = x_val[0]  # shape (5, 31)
        df = pd.DataFrame(sample_window, columns=feature_cols)
        df.insert(0, "step", [1, 2, 3, 4, 5])
        return df.to_csv(index=False)

    return ""
