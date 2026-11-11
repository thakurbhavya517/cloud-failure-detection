"""Model loading and CPU inference manager for Cloud Failure Detection Dashboard.

Supports both:
1. Aligned Research Models (W=15, d=42) trained in the pilot alignment phase
2. Legacy Phase 7D Models (L=5, d=31)
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import torch

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


class ModelManager:
    """Manages model loading, dimension validation, and CPU inference."""

    MODEL_REGISTRY = {
        "Aligned Stacked LSTM (W=15, d=42)": {
            "type": "pytorch_lstm_aligned",
            "path": RESULTS_DIR / "pilot_alignment_w15_d42" / "checkpoints" / "pilot_lstm_w15_d42.pt",
            "phase": "Aligned Pilot (Paper Target Spec)",
            "expected_seq_len": 15,
            "expected_features": 42,
            "description": "Paper Target: 2-layer Stacked LSTM (128 hidden units, Focal Loss γ=2.0) on 15-step (75 min), 42-feature telemetry.",
        },
        "Aligned Compact Transformer (W=15, d=42)": {
            "type": "pytorch_transformer_aligned",
            "path": RESULTS_DIR / "pilot_alignment_w15_d42" / "checkpoints" / "pilot_transformer_w15_d42.pt",
            "phase": "Aligned Pilot (Paper Target Spec)",
            "expected_seq_len": 15,
            "expected_features": 42,
            "description": "Paper Target: 2-layer Lightweight Transformer (64 d_model, 4 heads, Focal Loss γ=2.0) on 15-step (75 min), 42-feature telemetry.",
        },
        "Random Forest (M*) Baseline": {
            "type": "sklearn",
            "path": MODELS_DIR / "checkpoints_phase7d" / "random_forest_phase7d.joblib",
            "phase": "Phase 7D",
            "expected_seq_len": 5,
            "expected_features": 31,
            "description": "100 balanced decision trees operating on 31 causal telemetry features.",
        },
        "Legacy Temporal LSTM (Phase 7D)": {
            "type": "pytorch_lstm_legacy",
            "path": MODELS_DIR / "checkpoints_phase7d" / "best_lstm_checkpoint_phase7d.pt",
            "phase": "Phase 7D",
            "expected_seq_len": 5,
            "expected_features": 31,
            "description": "Legacy 2-layer causal LSTM (hidden_dim=64, dropout=0.20) evaluating 5-step, 31-feature sequences.",
        },
        "Legacy Compact Transformer (Phase 7D)": {
            "type": "pytorch_transformer_legacy",
            "path": MODELS_DIR / "checkpoints_phase7d" / "best_transformer_checkpoint_phase7d.pt",
            "phase": "Phase 7D",
            "expected_seq_len": 5,
            "expected_features": 31,
            "description": "Legacy 2-layer TransformerEncoder (d_model=32, nhead=4) on 5-step, 31-feature telemetry.",
        },
        "Representation Hybrid (Phase 7D)": {
            "type": "hybrid",
            "path": MODELS_DIR / "checkpoints_phase7d" / "representation_hybrid_phase7d.joblib",
            "backbone_path": MODELS_DIR / "checkpoints" / "exp3_lstm_2l_h32_balanced.pt",
            "phase": "Phase 7D",
            "expected_seq_len": 5,
            "expected_features": 31,
            "description": "Random Forest trained on fused 32-dim LSTM latent representations + 31 instantaneous features.",
        },
    }

    MODEL_ALIASES = {
        "Aligned Stacked LSTM": "Aligned Stacked LSTM (W=15, d=42)",
        "Aligned LSTM": "Aligned Stacked LSTM (W=15, d=42)",
        "Aligned Compact Transformer": "Aligned Compact Transformer (W=15, d=42)",
        "Aligned Transformer": "Aligned Compact Transformer (W=15, d=42)",
        "Random Forest (M*)": "Random Forest (M*) Baseline",
        "Random Forest (Phase 7D)": "Random Forest (M*) Baseline",
        "Temporal LSTM": "Legacy Temporal LSTM (Phase 7D)",
        "Temporal LSTM (Phase 7D)": "Legacy Temporal LSTM (Phase 7D)",
        "Compact Temporal Transformer": "Legacy Compact Transformer (Phase 7D)",
        "Compact Temporal Transformer (Phase 7D)": "Legacy Compact Transformer (Phase 7D)",
        "Representation Hybrid": "Representation Hybrid (Phase 7D)",
    }

    CLASS_NAMES = ["NORMAL", "WARNING", "CRITICAL"]

    def __init__(self):
        self._loaded_models = {}

    def _resolve_name(self, model_name: str) -> str:
        return self.MODEL_ALIASES.get(model_name, model_name)

    def get_available_models(self) -> List[str]:
        return list(self.MODEL_REGISTRY.keys())

    def get_model_info(self, model_name: str) -> dict:
        canonical = self._resolve_name(model_name)
        return self.MODEL_REGISTRY.get(canonical, {})

    def load_model(self, model_name: str):
        canonical = self._resolve_name(model_name)
        if canonical in self._loaded_models:
            return self._loaded_models[canonical]

        info = self.MODEL_REGISTRY.get(canonical)
        if not info:
            raise ValueError(f"Unknown model architecture requested: {model_name} (canonical: {canonical})")

        mtype = info["type"]
        path = info["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint file missing for '{canonical}'. Expected at: {path}. "
                "Ensure Stage 4 pilot training has been executed."
            )

        if mtype == "sklearn":
            model = joblib.load(path)
            self._loaded_models[canonical] = model
            return model

        elif mtype == "pytorch_lstm_aligned":
            model = TemporalLSTMClassifier(input_dim=42, hidden_dim=128, num_layers=2, dropout=0.20, num_classes=3)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[canonical] = model
            return model

        elif mtype == "pytorch_transformer_aligned":
            model = CompactTemporalTransformer(
                input_dim=42, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.20, num_classes=3, max_seq_len=15
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[canonical] = model
            return model

        elif mtype == "pytorch_lstm_legacy":
            model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, dropout=0.20, num_classes=3)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[canonical] = model
            return model

        elif mtype == "pytorch_transformer_legacy":
            model = CompactTemporalTransformer(
                input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.10, num_classes=3, max_seq_len=5
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[canonical] = model
            return model

        elif mtype == "hybrid":
            rf = joblib.load(path)
            b_path = info.get("backbone_path")
            lstm_backbone = None
            if b_path and b_path.exists():
                lstm_backbone = TemporalLSTMClassifier(input_dim=31, hidden_dim=32, num_layers=2, dropout=0.20, num_classes=3)
                ckpt = torch.load(b_path, map_location="cpu", weights_only=False)
                lstm_backbone.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                lstm_backbone.eval()
            bundle = {"rf": rf, "backbone": lstm_backbone}
            self._loaded_models[canonical] = bundle
            return bundle

        raise ValueError(f"Unsupported model type configuration: {mtype}")

    def validate_input_dimensions(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Validates and reshapes input array X to strictly match model expectations."""
        canonical = self._resolve_name(model_name)
        info = self.MODEL_REGISTRY.get(canonical, {})
        exp_seq_len = info.get("expected_seq_len", 15)
        exp_features = info.get("expected_features", 42)

        # Handle 1D input (single observation step)
        if X.ndim == 1:
            if len(X) == exp_features:
                # Shape (1, 1, F) -> repeat along sequence length
                X = np.tile(X, (1, exp_seq_len, 1))
            else:
                raise ValueError(
                    f"Feature dimension mismatch for '{canonical}': Received {len(X)} features, "
                    f"but model expects {exp_features} features."
                )

        # Handle 2D input
        elif X.ndim == 2:
            n_rows, n_cols = X.shape
            if n_cols == 31 and exp_features == 42:
                pad_cols = np.zeros((n_rows, 11), dtype=np.float32)
                X = np.concatenate([X, pad_cols], axis=1)
                n_cols = 42
            elif n_cols == 42 and exp_features == 31:
                X = X[:, :31]
                n_cols = 31

            if n_cols == exp_features:
                if n_rows == exp_seq_len:
                    X = np.expand_dims(X, axis=0)
                elif n_rows == 1:
                    X = np.repeat(np.expand_dims(X, axis=1), exp_seq_len, axis=1)
                elif n_rows > exp_seq_len:
                    windows = [X[i : i + exp_seq_len] for i in range(n_rows - exp_seq_len + 1)]
                    X = np.array(windows)
                else:
                    pad_len = exp_seq_len - n_rows
                    pad_rows = np.repeat(X[:1], pad_len, axis=0)
                    X = np.expand_dims(np.concatenate([pad_rows, X], axis=0), axis=0)
            elif n_rows * n_cols == exp_seq_len * exp_features:
                X = X.reshape(-1, exp_seq_len, exp_features)
            else:
                raise ValueError(
                    f"Feature dimension mismatch for '{canonical}': Received input with {n_cols} columns, "
                    f"but model strictly requires {exp_features} features (expected sequence length: {exp_seq_len})."
                )

        # Handle 3D input
        elif X.ndim == 3:
            _, s_len, n_feats = X.shape
            if n_feats != exp_features:
                if n_feats == 31 and exp_features == 42:
                    pad_feats = np.zeros((X.shape[0], s_len, 11), dtype=np.float32)
                    X = np.concatenate([X, pad_feats], axis=2)
                elif n_feats == 42 and exp_features == 31:
                    X = X[:, :, :31]
                else:
                    raise ValueError(
                        f"Feature dimension mismatch for '{canonical}': Input tensor has {n_feats} features "
                        f"(shape {X.shape}), but model expects exactly {exp_features} features."
                    )
            if s_len != exp_seq_len:
                if s_len > exp_seq_len:
                    X = X[:, -exp_seq_len:, :]
                else:
                    pad_len = exp_seq_len - s_len
                    pad_tensor = np.repeat(X[:, :1, :], pad_len, axis=1)
                    X = np.concatenate([pad_tensor, X], axis=1)

        return X.astype(np.float32)

    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform CPU inference.
        
        Args:
            model_name: Target model name.
            X: Input telemetry array.
            
        Returns:
            Tuple of (predicted_classes: np.ndarray, class_probabilities: np.ndarray)
        """
        canonical = self._resolve_name(model_name)
        model = self.load_model(canonical)
        info = self.MODEL_REGISTRY[canonical]
        mtype = info["type"]

        # Validate and format input
        X_adj = self.validate_input_dimensions(canonical, X)

        if mtype == "sklearn":
            # RF operates on the last time-step instantaneous features
            X_last = X_adj[:, -1, :]
            probs = model.predict_proba(X_last)
            preds = np.argmax(probs, axis=-1)
            return preds, probs

        elif mtype in ("pytorch_lstm_aligned", "pytorch_transformer_aligned", "pytorch_lstm_legacy", "pytorch_transformer_legacy"):
            tensor_x = torch.tensor(X_adj, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            return preds, probs

        elif mtype == "hybrid":
            rf = model["rf"]
            backbone = model["backbone"]
            X_tabular = X_adj[:, -1, :]

            if backbone is not None:
                tensor_x = torch.tensor(X_adj, dtype=torch.float32)
                with torch.no_grad():
                    lstm_out, _ = backbone.lstm(tensor_x)
                    H_temp = lstm_out[:, -1, :].numpy()
                Z = np.concatenate([H_temp, X_tabular], axis=1)
            else:
                Z = np.pad(X_tabular, ((0, 0), (32, 0)), mode="constant")

            probs = rf.predict_proba(Z)
            preds = np.argmax(probs, axis=-1)
            return preds, probs

        raise ValueError(f"Unsupported model type: {mtype}")

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (N, 3) for the given input."""
        _, probs = self.predict(model_name, X)
        return probs

    @staticmethod
    def evaluate_k_consecutive_alert(predictions: np.ndarray, k: int = 3) -> Dict[str, Any]:
        """Evaluates whether the k consecutive alert policy triggers an advance warning.
        
        Args:
            predictions: 1D array of class predictions [0, 1, 2] over consecutive steps.
            k: Threshold of consecutive warning/critical flags required (default 3).
            
        Returns:
            Dictionary with alert status, triggering step index, and flag history.
        """
        flags = np.array(predictions >= 1, dtype=bool)
        alert_triggered = False
        trigger_step = None
        consecutive_count = 0

        for idx, flag in enumerate(flags):
            if flag:
                consecutive_count += 1
                if consecutive_count >= k:
                    alert_triggered = True
                    trigger_step = idx
                    break
            else:
                consecutive_count = 0

        # Also trigger if the very last step is critical (fail-safe)
        if not alert_triggered and len(predictions) > 0 and predictions[-1] == 2:
            alert_triggered = True
            trigger_step = len(predictions) - 1

        return {
            "alert_triggered": alert_triggered,
            "trigger_step": trigger_step,
            "consecutive_count": consecutive_count,
            "k_threshold": k,
            "flags": flags.tolist(),
        }
