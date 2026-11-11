"""Causal Preprocessing and Schema Validation Pipeline for Dashboard Inference.

Supports both:
1. Aligned Research Specification: 42 causal features (W=15 sequence length)
2. Legacy Phase 7D Specification: 31 causal features (L=5 sequence length)
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dashboard.utils.data_loader import (
    FEATURE_NAMES_31,
    FEATURE_NAMES_42,
    load_feature_scaler_spec,
)


class InferencePipeline:
    """Preprocesses input telemetry tables for model inference."""

    def __init__(self, mode: str = "legacy"):
        """Args:
            mode: 'aligned' (42 features, W=15) or 'legacy' (31 features, L=5)
        """
        self.mode = mode
        if mode == "aligned":
            self.feature_cols: List[str] = FEATURE_NAMES_42
            self.expected_seq_len: int = 15
        else:
            self.feature_cols: List[str] = FEATURE_NAMES_31
            self.expected_seq_len: int = 5

        self.scaler_spec = load_feature_scaler_spec()
        self.medians_31: Dict[str, float] = self.scaler_spec.get("medians", {})
        self.means_31: Dict[str, float] = self.scaler_spec.get("means", {})
        self.stds_31: Dict[str, float] = self.scaler_spec.get("stds", {})

    def validate_and_preprocess_dataframe(
        self, df: pd.DataFrame, auto_scale: bool = False
    ) -> Tuple[bool, str, Optional[np.ndarray], Dict[str, Any]]:
        """Validates CSV input and prepares (N, seq_len, n_features) array for prediction.
        
        Args:
            df: Input pandas DataFrame.
            auto_scale: If True, applies standard scaling.
            
        Returns:
            Tuple of (is_valid: bool, error_message: str, array: np.ndarray, report: dict)
        """
        if df.empty:
            return False, "Uploaded telemetry file is completely empty.", None, {}

        # Strip whitespace from column names
        df_cols = [c.strip() for c in df.columns]
        df.columns = df_cols

        # Check for matching features
        missing_cols = [col for col in self.feature_cols if col not in df_cols]
        if missing_cols:
            if len(missing_cols) > 5:
                err_msg = f"Missing {len(missing_cols)} required columns for '{self.mode}' mode. (e.g., {missing_cols[:5]})"
            else:
                err_msg = f"Missing required telemetry columns: {', '.join(missing_cols)}"
            return False, err_msg, None, {"missing_columns": missing_cols}

        # Extract features in strict canonical order
        feature_df = df[self.feature_cols].copy()

        # Coerce non-numeric columns
        non_numeric = []
        for col in self.feature_cols:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                non_numeric.append(col)
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

        # Impute missing values with training medians or 0.0
        nan_counts = feature_df.isna().sum().to_dict()
        total_nans = sum(nan_counts.values())

        if total_nans > 0:
            for col in self.feature_cols:
                med = self.medians_31.get(col, 0.0)
                feature_df[col] = feature_df[col].fillna(med)

        raw_vals = feature_df.values.astype(np.float32)

        # Scale if requested
        if auto_scale:
            means_vec = np.array([self.means_31.get(c, 0.0) for c in self.feature_cols], dtype=np.float32)
            stds_vec = np.array([self.stds_31.get(c, 1.0) for c in self.feature_cols], dtype=np.float32)
            stds_vec = np.where(stds_vec < 1e-8, 1.0, stds_vec)
            scaled_vals = (raw_vals - means_vec) / stds_vec
        else:
            scaled_vals = raw_vals

        num_rows = scaled_vals.shape[0]
        L = self.expected_seq_len
        F = len(self.feature_cols)

        # Format into sliding windows of length L
        if num_rows == L:
            X_out = np.expand_dims(scaled_vals, axis=0)  # Shape (1, L, F)
        elif num_rows > L:
            windows = [scaled_vals[i : i + L] for i in range(num_rows - L + 1)]
            X_out = np.array(windows)  # Shape (num_rows - L + 1, L, F)
        elif num_rows == 1:
            X_out = np.repeat(np.expand_dims(scaled_vals, axis=1), L, axis=1)  # Shape (1, L, F)
        else:
            pad_needed = L - num_rows
            first_row_repeated = np.repeat(scaled_vals[:1], pad_needed, axis=0)
            padded = np.concatenate([first_row_repeated, scaled_vals], axis=0)
            X_out = np.expand_dims(padded, axis=0)  # Shape (1, L, F)

        report = {
            "mode": self.mode,
            "total_input_rows": num_rows,
            "evaluated_windows": X_out.shape[0],
            "sequence_length": X_out.shape[1],
            "feature_count": X_out.shape[2],
            "imputed_nans": total_nans,
            "non_numeric_columns_coerced": non_numeric,
            "auto_scaling_applied": auto_scale,
        }

        return True, "", X_out, report
