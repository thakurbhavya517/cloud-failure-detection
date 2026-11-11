"""Verification test for dashboard backend components."""

import io
import numpy as np
import pandas as pd
from dashboard.utils.data_loader import (
    load_candidate_a_val_data,
    load_phase8_audit_metrics,
    load_feature_scaler_spec,
    get_sample_telemetry_csv,
)
from dashboard.utils.model_loader import ModelManager
from dashboard.utils.inference_pipeline import InferencePipeline


def test_data_loader():
    meta, x_val, y_val = load_candidate_a_val_data()
    assert meta is not None, "Metadata should not be None"
    assert x_val.shape == (657, 5, 31), f"Unexpected x_val shape {x_val.shape}"
    assert y_val.shape == (657,), f"Unexpected y_val shape {y_val.shape}"

    scaler = load_feature_scaler_spec()
    assert len(scaler.get("feature_cols", [])) == 31, "Feature count should be 31"


def test_model_manager():
    mm = ModelManager()
    models = mm.get_available_models()
    assert len(models) >= 5, f"Expected at least 5 models, got {len(models)}"

    _, x_val, _ = load_candidate_a_val_data()
    test_slice = x_val[:5]

    for model_name in models:
        preds, probs = mm.predict(model_name, test_slice)
        probs_only = mm.predict_proba(model_name, test_slice)
        assert len(preds) == 5, f"Expected 5 predictions for {model_name}"
        assert probs.shape == (5, 3), f"Expected shape (5, 3) for {model_name}"
        assert probs_only.shape == (5, 3), f"Expected shape (5, 3) for {model_name}"


def test_inference_pipeline():
    ip = InferencePipeline()
    sample_csv = get_sample_telemetry_csv()
    df = pd.read_csv(io.StringIO(sample_csv))
    valid, err, win, report = ip.validate_and_preprocess_dataframe(df)
    assert valid, f"Validation failed: {err}"
    assert win is not None
    assert win.shape == (1, 5, 31), f"Unexpected window shape {win.shape}"
    assert report["feature_count"] == 31


def test_aligned_models_and_pipeline():
    """Verify aligned model loading and W=15, d=42 inference."""
    mm = ModelManager()
    lstm_name = "Aligned Stacked LSTM (W=15, d=42)"
    trans_name = "Aligned Compact Transformer (W=15, d=42)"

    assert lstm_name in mm.get_available_models()
    assert trans_name in mm.get_available_models()

    # Test with 15-step, 42-feature input
    x_42 = np.random.randn(2, 15, 42).astype(np.float32)

    preds_l, probs_l = mm.predict(lstm_name, x_42)
    assert len(preds_l) == 2
    assert probs_l.shape == (2, 3)

    preds_t, probs_t = mm.predict(trans_name, x_42)
    assert len(preds_t) == 2
    assert probs_t.shape == (2, 3)

    # Test aligned inference pipeline
    ip_aligned = InferencePipeline(mode="aligned")
    sample_csv_42 = get_sample_telemetry_csv(version="aligned_w15_d42")
    df_42 = pd.read_csv(io.StringIO(sample_csv_42))
    valid, err, win, report = ip_aligned.validate_and_preprocess_dataframe(df_42)
    assert valid, f"Aligned validation failed: {err}"
    assert win.shape == (1, 15, 42)
    assert report["feature_count"] == 42
    assert report["sequence_length"] == 15
