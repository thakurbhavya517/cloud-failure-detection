"""Causal Preprocessing and Schema Validation Pipeline for Dashboard Inference.

Enforces strictly training-derived median imputation and standard scaling
derived from configs/phase7c/fitted_scaler_phase7c.json.
"""

from typing import Tuple, Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
from dashboard.utils.data_loader import load_feature_scaler_spec


class InferencePipeline:
    """Preprocesses input telemetry tables for model inference."""

    def __init__(self):
        self.scaler_spec = load_feature_scaler_spec()
        self.feature_cols: List[str] = self.scaler_spec.get("feature_cols", [])
        self.medians: Dict[str, float] = self.scaler_spec.get("medians", {})
        self.means: Dict[str, float] = self.scaler_spec.get("means", {})
        self.stds: Dict[str, float] = self.scaler_spec.get("stds", {})
        self.epsilon: float = self.scaler_spec.get("epsilon", 1e-8)

    def validate_and_preprocess_dataframe(
        self, df: pd.DataFrame, auto_scale: bool = False
    ) -> Tuple[bool, str, Optional[np.ndarray], Dict[str, Any]]:
        """Validates CSV input and prepares (N, 5, 31) array for prediction.
        
        Args:
            df: Input pandas DataFrame.
            auto_scale: If True, applies training set StandardScaler parameters.
            
        Returns:
            Tuple of (is_valid: bool, error_message: str, array: np.ndarray, report: dict)
        """
        if df.empty:
            return False, "Uploaded file is completely empty.", None, {}

        # 1. Check feature columns
        df_cols = [c.strip() for c in df.columns]
        df.columns = df_cols

        missing_cols = [col for col in self.feature_cols if col not in df_cols]
        if missing_cols:
            if len(missing_cols) > 5:
                err_msg = f"Missing {len(missing_cols)} required telemetry columns. (e.g. {missing_cols[:5]})"
            else:
                err_msg = f"Missing required telemetry columns: {', '.join(missing_cols)}"
            return False, err_msg, None, {"missing_columns": missing_cols}

        # 2. Extract feature subset in strict canonical order
        feature_df = df[self.feature_cols].copy()

        # 3. Check for non-numeric types
        non_numeric = []
        for col in self.feature_cols:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                non_numeric.append(col)
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
                except Exception:
                    pass

        # 4. Impute NaNs using strict training medians
        nan_counts = feature_df.isna().sum().to_dict()
        total_nans = sum(nan_counts.values())

        if total_nans > 0:
            for col in self.feature_cols:
                med = self.medians.get(col, 0.0)
                feature_df[col] = feature_df[col].fillna(med)

        # 5. Optional Standard Scaling
        raw_vals = feature_df.values.astype(np.float32)
        if auto_scale:
            means_vec = np.array([self.means.get(c, 0.0) for c in self.feature_cols], dtype=np.float32)
            stds_vec = np.array([self.stds.get(c, 1.0) for c in self.feature_cols], dtype=np.float32)
            stds_vec = np.where(stds_vec < self.epsilon, 1.0, stds_vec)
            scaled_vals = (raw_vals - means_vec) / stds_vec
        else:
            scaled_vals = raw_vals

        num_rows = scaled_vals.shape[0]

        # 6. Format into 5-step sequence windows
        if num_rows == 5:
            # Exactly 1 sequence window
            X_out = np.expand_dims(scaled_vals, axis=0)  # Shape (1, 5, 31)
        elif num_rows > 5:
            # Generate sliding windows of length 5
            windows = []
            for i in range(num_rows - 4):
                windows.append(scaled_vals[i : i + 5])
            X_out = np.array(windows)  # Shape (N - 4, 5, 31)
        elif num_rows == 1:
            # Repeat single observation across 5 steps for point evaluation
            X_out = np.repeat(np.expand_dims(scaled_vals, axis=1), 5, axis=1)  # Shape (1, 5, 31)
        else:
            # 2, 3, or 4 rows: pad with oldest observation
            pad_needed = 5 - num_rows
            first_row_repeated = np.repeat(scaled_vals[:1], pad_needed, axis=0)
            padded = np.concatenate([first_row_repeated, scaled_vals], axis=0)
            X_out = np.expand_dims(padded, axis=0)  # Shape (1, 5, 31)

        report = {
            "total_input_rows": num_rows,
            "evaluated_windows": X_out.shape[0],
            "feature_count": X_out.shape[2],
            "sequence_length": X_out.shape[1],
            "imputed_nans": total_nans,
            "non_numeric_columns_coerced": non_numeric,
            "auto_scaling_applied": auto_scale,
        }

        return True, "", X_out, report
