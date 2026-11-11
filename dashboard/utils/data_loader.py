"""Data and artifact loading utilities for the Cloud Failure Detection Dashboard.

Supports both:
1. Aligned Research Models (W=15, d=42) from data/processed/pilot_w15_d42/
2. Historical Phase 7D Models (L=5, d=31) from data/processed/phase7c/
"""

from pathlib import Path
import json
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR_PHASE7C = REPO_ROOT / "data" / "processed" / "phase7c"
DATA_DIR_ALIGNED = REPO_ROOT / "data" / "processed" / "pilot_w15_d42"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

# Pre-registered Operational Gates
OPERATIONAL_GATES = {
    "Gate 1a (Episode Recall)": {"threshold": 70.0, "unit": "%", "direction": ">="},
    "Gate 1b (Critical Recall)": {"threshold": 50.0, "unit": "%", "direction": ">="},
    "Gate 2a (Pure-Normal Entity FAR)": {"threshold": 10.0, "unit": "%", "direction": "<="},
    "Gate 2b (Normal Window FPR)": {"threshold": 5.0, "unit": "%", "direction": "<="},
}

FEATURE_NAMES_31 = [
    "cpu_utilization", "memory_utilization", "assigned_memory", "disk_io_time",
    "cycles_per_instruction", "memory_accesses_per_instruction", "cpu_utilization_lag_1",
    "memory_utilization_lag_1", "cpu_utilization_lag_2", "memory_utilization_lag_2",
    "cpu_utilization_lag_3", "memory_utilization_lag_3", "cpu_utilization_lag_6",
    "memory_utilization_lag_6", "cpu_diff_1", "cpu_rel_change_1", "memory_diff_1",
    "cpu_rolling_mean_3", "cpu_rolling_std_3", "cpu_rolling_range_3", "memory_rolling_mean_3",
    "cpu_rolling_mean_6", "cpu_rolling_std_6", "cpu_rolling_range_6", "memory_rolling_mean_6",
    "cpu_trend_slope_6", "cpu_exceed_85", "cpu_exceed_fraction_6", "memory_to_assigned_ratio",
    "task_age_minutes", "cpi_missing_fraction_6"
]

FEATURE_NAMES_42 = FEATURE_NAMES_31 + [
    "max_cpu_rate", "cpu_burstiness", "max_memory", "memory_headroom", "total_page_cache",
    "priority", "scheduling_class", "req_cpu", "req_ram",
    "eviction_count_prior", "time_since_last_schedule"
]


def load_aligned_pilot_metrics() -> Dict[str, Any]:
    """Load genuine empirical metrics from the aligned pilot experiment."""
    metrics_path = RESULTS_DIR / "pilot_alignment_w15_d42" / "pilot_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_phase8_audit_metrics() -> dict:
    """Load genuine audited metrics from Phase 8 research integrity audit."""
    p8_path = RESULTS_DIR / "phase8_research_integrity" / "phase8_dataset_integrity_audit.json"
    if p8_path.exists():
        with open(p8_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_aligned_pilot_data(split: str = "val") -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load Aligned (W=15, d=42) data partition and metadata safely.
    
    Args:
        split: 'val', 'test', or 'train'
    """
    valid_splits = {"val": "val", "test": "test", "train": "train"}
    sp = valid_splits.get(split, "val")

    meta_path = DATA_DIR_ALIGNED / f"{sp}_metadata.csv"
    x_path = DATA_DIR_ALIGNED / f"X_{sp}.npy"
    y_path = DATA_DIR_ALIGNED / f"y_{sp}.npy"

    if meta_path.exists() and x_path.exists() and y_path.exists():
        metadata = pd.read_csv(meta_path)
        x_arr = np.load(x_path)
        y_arr = np.load(y_path)
        return metadata, x_arr, y_arr
    return None, None, None


def load_candidate_a_val_data() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load historical Candidate A validation arrays and metadata safely."""
    val_meta_path = DATA_DIR_PHASE7C / "val_metadata.csv"
    x_val_path = DATA_DIR_PHASE7C / "X_val.npy"
    y_val_path = DATA_DIR_PHASE7C / "y_val.npy"

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


def get_sample_telemetry_csv(version: str = "legacy_l5_d31") -> str:
    """Generate a valid sample CSV containing telemetry observations.
    
    Args:
        version: 'aligned_w15_d42' (15 steps, 42 features) or 'legacy_l5_d31' (5 steps, 31 features)
    """
    if version == "aligned_w15_d42":
        _, x_val, _ = load_aligned_pilot_data("val")
        if x_val is not None:
            sample_window = x_val[0]  # shape (15, 42)
            df = pd.DataFrame(sample_window, columns=FEATURE_NAMES_42)
            df.insert(0, "step", list(range(1, 16)))
            return df.to_csv(index=False)
    
    # Fallback to legacy
    _, x_val, _ = load_candidate_a_val_data()
    if x_val is not None:
        sample_window = x_val[0]  # shape (5, 31)
        df = pd.DataFrame(sample_window, columns=FEATURE_NAMES_31)
        df.insert(0, "step", [1, 2, 3, 4, 5])
        return df.to_csv(index=False)

    return ""
