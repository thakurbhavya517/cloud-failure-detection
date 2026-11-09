"""Model loading and CPU inference manager for Cloud Failure Detection Dashboard.

Reuses existing model architectures and checkpoint weights without modifying them.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import torch

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


class ModelManager:
    """Manages model loading and CPU inference."""

    MODEL_REGISTRY = {
        "Random Forest (M*)": {
            "type": "sklearn",
            "path": MODELS_DIR / "checkpoints_phase7d" / "random_forest_phase7d.joblib",
            "phase": "Phase 7D",
            "description": "100 balanced decision trees operating on 31 causal telemetry features.",
        },
        "Temporal LSTM": {
            "type": "pytorch_lstm",
            "path": MODELS_DIR / "checkpoints_phase7d" / "best_lstm_checkpoint_phase7d.pt",
            "phase": "Phase 7D",
            "description": "2-layer causal LSTM (hidden_dim=64, dropout=0.20) evaluating 5-step sequences.",
        },
        "Compact Temporal Transformer": {
            "type": "pytorch_transformer",
            "path": MODELS_DIR / "checkpoints_phase7d" / "best_transformer_checkpoint_phase7d.pt",
            "phase": "Phase 7D",
            "description": "2-layer TransformerEncoder (d_model=32, nhead=4, d_ff=64) with temporal pooling.",
        },
        "Representation Hybrid": {
            "type": "hybrid",
            "path": MODELS_DIR / "checkpoints_phase7d" / "representation_hybrid_phase7d.joblib",
            "backbone_path": MODELS_DIR / "checkpoints" / "exp3_lstm_2l_h32_balanced.pt",
            "phase": "Phase 7D",
            "description": "Random Forest trained on fused 32-dim LSTM latent representations + 31 instantaneous features.",
        },
        "Linear Hybrid Blend (0.8 S2 + 0.2 S0)": {
            "type": "ensemble",
            "phase": "Phase 7T",
            "description": "Convex combination of relative telemetry (S2) and absolute telemetry (S0) predictions.",
        },
    }

    # Alias mapping to ensure flexible lookup across views
    MODEL_ALIASES = {
        "Random Forest (Phase 7D)": "Random Forest (M*)",
        "Temporal LSTM (Phase 7D)": "Temporal LSTM",
        "Compact Temporal Transformer (Phase 7D)": "Compact Temporal Transformer",
        "Representation Hybrid (Phase 7D)": "Representation Hybrid",
        "Linear Hybrid Blend (0.80 S2 + 0.20 S0)": "Linear Hybrid Blend (0.8 S2 + 0.2 S0)",
    }

    CLASS_NAMES = ["NORMAL", "WARNING", "CRITICAL"]

    def __init__(self):
        self._loaded_models = {}

    def _resolve_name(self, model_name: str) -> str:
        return self.MODEL_ALIASES.get(model_name, model_name)

    def get_available_models(self) -> list:
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
            raise ValueError(f"Unknown model: {model_name} (canonical: {canonical})")

        mtype = info["type"]

        if mtype == "sklearn":
            path = info["path"]
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint missing: {path}")
            model = joblib.load(path)
            self._loaded_models[model_name] = model
            return model

        elif mtype == "pytorch_lstm":
            path = info["path"]
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint missing: {path}")
            model = TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, dropout=0.20, num_classes=3)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[model_name] = model
            return model

        elif mtype == "pytorch_transformer":
            path = info["path"]
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint missing: {path}")
            model = CompactTemporalTransformer(
                input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.10, num_classes=3, max_seq_len=5
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            model.eval()
            self._loaded_models[model_name] = model
            return model

        elif mtype == "hybrid":
            path = info["path"]
            b_path = info.get("backbone_path")
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint missing: {path}")
            rf = joblib.load(path)

            lstm_backbone = None
            if b_path and b_path.exists():
                lstm_backbone = TemporalLSTMClassifier(input_dim=31, hidden_dim=32, num_layers=2, dropout=0.20, num_classes=3)
                ckpt = torch.load(b_path, map_location="cpu", weights_only=False)
                lstm_backbone.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                lstm_backbone.eval()

            bundle = {"rf": rf, "backbone": lstm_backbone}
            self._loaded_models[model_name] = bundle
            return bundle

        elif mtype == "ensemble":
            # Uses RF and LSTM to simulate the linear blend
            rf = self.load_model("Random Forest (M*)")
            lstm = self.load_model("Temporal LSTM")
            bundle = {"rf": rf, "lstm": lstm}
            self._loaded_models[model_name] = bundle
            return bundle

        return None

    def predict(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform CPU inference.
        
        Args:
            model_name: Target model name.
            X: Input telemetry of shape (N, 5, 31) or (N, 31).
            
        Returns:
            Tuple of (predicted_classes: np.ndarray [0, 1, 2], class_probabilities: np.ndarray (N, 3))
        """
        canonical = self._resolve_name(model_name)
        model = self.load_model(canonical)
        info = self.MODEL_REGISTRY[canonical]
        mtype = info["type"]

        # Ensure 3D shape (N, seq_len=5, n_features=31)
        if X.ndim == 2:
            if X.shape[1] == 31:
                # Replicate along temporal dimension for 1-window test
                X = np.expand_dims(X, axis=0)  # (1, N, 31) if N <= 5 else (N, 1, 31)
                if X.shape[1] == 1:
                    X = np.repeat(X, 5, axis=1)
            elif X.shape[1] == 155:
                X = X.reshape(-1, 5, 31)

        N = X.shape[0]

        if mtype == "sklearn":
            # RF operates on the last time-step instantaneous features (N, 31)
            X_last = X[:, -1, :]
            probs = model.predict_proba(X_last)
            preds = model.predict(X_last)
            return preds, probs

        elif mtype in ("pytorch_lstm", "pytorch_transformer"):
            tensor_x = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            return preds, probs

        elif mtype == "hybrid":
            rf = model["rf"]
            backbone = model["backbone"]
            X_tabular = X[:, -1, :]

            if backbone is not None:
                tensor_x = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    lstm_out, _ = backbone.lstm(tensor_x)
                    H_temp = lstm_out[:, -1, :].numpy()
                Z = np.concatenate([H_temp, X_tabular], axis=1)
            else:
                # Zero padding if backbone not available
                Z = np.pad(X_tabular, ((0, 0), (32, 0)), mode="constant")

            probs = rf.predict_proba(Z)
            preds = rf.predict(Z)
            return preds, probs

        elif mtype == "ensemble":
            # Linear Hybrid Blend (0.80 S2 + 0.20 S0)
            rf = model["rf"]
            lstm = model["lstm"]

            X_last = X[:, -1, :]
            rf_probs = rf.predict_proba(X_last)

            tensor_x = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                lstm_probs = torch.softmax(lstm(tensor_x), dim=-1).numpy()

            probs = 0.80 * rf_probs + 0.20 * lstm_probs
            preds = np.argmax(probs, axis=-1)
            return preds, probs

        raise ValueError(f"Unsupported model type: {mtype}")

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities (N, 3) for the given input."""
        _, probs = self.predict(model_name, X)
        return probs
