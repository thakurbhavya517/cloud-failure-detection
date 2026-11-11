"""Stage 4 Pilot Experiment: Train and Evaluate Aligned Models (W=15, d=42) on Borg 2011 Chronological Data.

Guiding Standards:
- Non-destructive: saves strictly to results/pilot_alignment_w15_d42/
- Zero access to sealed holdouts: uses only data/processed/pilot_w15_d42/
- Architecture: 128-dim Stacked LSTM, 64-dim 4-head Compact Transformer, RF baseline
- Loss: Multi-Class Focal Loss (gamma=2.0)
- Operational Lead-Time: k=3 consecutive alerts
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
# Track Hardware Usage
import os
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import TemporalLSTMClassifier
from src.models.transformer import CompactTemporalTransformer

# Setup Experiment Directory
exp_dir = PROJECT_ROOT / "results" / "pilot_alignment_w15_d42"
ckpt_dir = exp_dir / "checkpoints"
exp_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Set Seeds for Deterministic Reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Multi-Class Focal Loss (gamma = 2.0)
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_p = torch.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        focal_weight = torch.pow(1.0 - target_p, self.gamma)
        loss = -focal_weight * target_log_p
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=-1, index=targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

# Load Data
data_dir = PROJECT_ROOT / "data" / "processed" / "pilot_w15_d42"
X_train = np.load(data_dir / "X_train.npy") # (1603, 15, 42)
y_train = np.load(data_dir / "y_train.npy")
train_meta = pd.read_csv(data_dir / "train_metadata.csv")

X_val = np.load(data_dir / "X_val.npy")     # (724, 15, 42)
y_val = np.load(data_dir / "y_val.npy")
val_meta = pd.read_csv(data_dir / "val_metadata.csv")

X_test = np.load(data_dir / "X_test.npy")   # (460, 15, 42)
y_test = np.load(data_dir / "y_test.npy")
test_meta = pd.read_csv(data_dir / "test_metadata.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Pilot Experiment on device: {device}")

# Operational Lead-Time Evaluation with k=3 Consecutive Alert Policy
def evaluate_operational_metrics(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> Dict[str, Any]:
    df = meta_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    
    # Calculate Macro-F1, Weighted-F1, and per-class stats
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
    p_per, r_per, f_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0.0)
    
    # Group by entity to evaluate failure episodes and lead times
    episodes_detected = 0
    total_episodes = 0
    lead_times_min = []
    
    # False alarm tracking on pure normal entities
    total_normal_steps = 0
    false_alarm_steps = 0
    
    for eid, grp in df.groupby("entity_id", sort=False):
        grp = grp.sort_values("prediction_timestamp").reset_index(drop=True)
        is_failing_entity = (grp["y_true"] > 0).any()
        
        if is_failing_entity:
            # Locate earliest true failure proximity step
            fail_step_idx = grp[grp["y_true"] > 0].index[0]
            fail_pred_time = grp.loc[fail_step_idx, "prediction_timestamp"]
            ttf = grp.loc[fail_step_idx, "time_to_failure"]
            actual_fail_time = fail_pred_time + ttf if not pd.isna(ttf) else fail_pred_time
            total_episodes += 1
            
            # Check for k consecutive alert flags (y_pred >= 1) prior to or at failure
            alert_time = None
            pred_flags = (grp["y_pred"] >= 1).values
            for idx in range(len(pred_flags)):
                # Alert triggers if k consecutive flags or last step flag
                if idx >= k - 1 and all(pred_flags[idx - (k - 1) : idx + 1]):
                    alert_time = grp.loc[idx, "prediction_timestamp"]
                    break
                elif idx == len(pred_flags) - 1 and pred_flags[idx]:
                    alert_time = grp.loc[idx, "prediction_timestamp"]
                    break
                    
            if alert_time is not None and alert_time <= actual_fail_time:
                episodes_detected += 1
                lead_min = max(0.0, (actual_fail_time - alert_time) / 60.0)
                lead_times_min.append(lead_min)
        else:
            # Pure normal entity: evaluate false alarms
            normal_preds = grp["y_pred"].values
            total_normal_steps += len(normal_preds)
            # A false alarm is triggered if k consecutive alerts occur in pure normal execution
            for idx in range(len(normal_preds)):
                if idx >= k - 1 and all(normal_preds[idx - (k - 1) : idx + 1] >= 1):
                    false_alarm_steps += 1
                    
    ep_detection_rate = float(episodes_detected / total_episodes * 100.0) if total_episodes > 0 else 0.0
    mean_lead = float(np.mean(lead_times_min)) if lead_times_min else 0.0
    median_lead = float(np.median(lead_times_min)) if lead_times_min else 0.0
    fa_per_1000 = float((false_alarm_steps / total_normal_steps) * 1000.0) if total_normal_steps > 0 else 0.0
    
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "class_0_f1": float(f_per[0]),
        "class_1_f1": float(f_per[1]),
        "class_2_f1": float(f_per[2]),
        "class_0_precision": float(p_per[0]),
        "class_1_precision": float(p_per[1]),
        "class_2_precision": float(p_per[2]),
        "class_0_recall": float(r_per[0]),
        "class_1_recall": float(r_per[1]),
        "class_2_recall": float(r_per[2]),
        "total_episodes": total_episodes,
        "episodes_detected": episodes_detected,
        "detection_rate_pct": ep_detection_rate,
        "mean_lead_time_min": mean_lead,
        "median_lead_time_min": median_lead,
        "false_alarms_per_1000_steps": fa_per_1000,
    }

# Track Hardware Usage
start_time = time.time()

# -----------------------------------------------------------------------------
# 1. Baseline Model: Random Forest (Last Step)
# -----------------------------------------------------------------------------
print("\n--- Training Random Forest (Last-Step Baseline) ---")
rf_t0 = time.time()
X_tr_last = X_train[:, -1, :] # (1603, 42)
X_v_last = X_val[:, -1, :]   # (724, 42)
X_te_last = X_test[:, -1, :] # (460, 42)

rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_tr_last, y_train)
rf_time = time.time() - rf_t0

rf_val_preds = rf.predict(X_v_last)
rf_test_preds = rf.predict(X_te_last)

rf_val_metrics = evaluate_operational_metrics(val_meta, y_val, rf_val_preds)
rf_test_metrics = evaluate_operational_metrics(test_meta, y_test, rf_test_preds)

print(f"Random Forest (Test) — Macro-F1: {rf_test_metrics['macro_f1']:.4f}, Det: {rf_test_metrics['detection_rate_pct']:.1f}%, Lead: {rf_test_metrics['median_lead_time_min']:.1f}m, FA: {rf_test_metrics['false_alarms_per_1000_steps']:.1f}/1000")

# -----------------------------------------------------------------------------
# 2. Stacked LSTM (128 hidden, 2 layers, dropout 0.2, Focal Loss gamma=2.0)
# -----------------------------------------------------------------------------
print("\n--- Training Stacked LSTM (128 units, 2 layers) ---")
lstm_t0 = time.time()
lstm_model = TemporalLSTMClassifier(input_dim=42, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2).to(device)

# Alpha weights: inversely proportional to class frequency
alpha_weights = torch.tensor([0.1, 0.45, 0.45], device=device)
criterion = MultiClassFocalLoss(alpha=alpha_weights, gamma=2.0)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)), batch_size=64, shuffle=True)
val_x = torch.from_numpy(X_val).to(device)
test_x = torch.from_numpy(X_test).to(device)

best_val_f1 = -1.0
best_lstm_weights = None

for epoch in range(1, 21):
    lstm_model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = lstm_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    lstm_model.eval()
    with torch.no_grad():
        v_out = lstm_model(val_x)
        v_preds = torch.argmax(v_out, dim=-1).cpu().numpy()
        val_macro = f1_score(y_val, v_preds, average="macro", zero_division=0.0)
        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_lstm_weights = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}

lstm_time = time.time() - lstm_t0
lstm_model.load_state_dict(best_lstm_weights)
lstm_model.eval()
with torch.no_grad():
    lstm_val_preds = torch.argmax(lstm_model(val_x), dim=-1).cpu().numpy()
    lstm_test_preds = torch.argmax(lstm_model(test_x), dim=-1).cpu().numpy()

lstm_val_metrics = evaluate_operational_metrics(val_meta, y_val, lstm_val_preds)
lstm_test_metrics = evaluate_operational_metrics(test_meta, y_test, lstm_test_preds)

torch.save({"model_state_dict": best_lstm_weights, "metrics": lstm_test_metrics}, ckpt_dir / "pilot_lstm_w15_d42.pt")
print(f"Stacked LSTM (Test) — Macro-F1: {lstm_test_metrics['macro_f1']:.4f}, Det: {lstm_test_metrics['detection_rate_pct']:.1f}%, Lead: {lstm_test_metrics['median_lead_time_min']:.1f}m, FA: {lstm_test_metrics['false_alarms_per_1000_steps']:.1f}/1000")

# -----------------------------------------------------------------------------
# 3. Compact Transformer (64 d_model, 4 heads, 2 layers, Focal Loss gamma=2.0)
# -----------------------------------------------------------------------------
print("\n--- Training Compact Transformer (64 d_model, 4 heads, 2 layers) ---")
trans_t0 = time.time()
trans_model = CompactTemporalTransformer(input_dim=42, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=3, dropout=0.2, max_seq_len=15).to(device)
optimizer_t = torch.optim.Adam(trans_model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_f1_t = -1.0
best_trans_weights = None

for epoch in range(1, 21):
    trans_model.train()
    for bx, by in train_loader:
        optimizer_t.zero_grad()
        out = trans_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer_t.step()
        
    trans_model.eval()
    with torch.no_grad():
        v_out = trans_model(val_x)
        v_preds = torch.argmax(v_out, dim=-1).cpu().numpy()
        val_macro = f1_score(y_val, v_preds, average="macro", zero_division=0.0)
        if val_macro > best_val_f1_t:
            best_val_f1_t = val_macro
            best_trans_weights = {k: v.cpu().clone() for k, v in trans_model.state_dict().items()}

trans_time = time.time() - trans_t0
trans_model.load_state_dict(best_trans_weights)
trans_model.eval()
with torch.no_grad():
    trans_val_preds = torch.argmax(trans_model(val_x), dim=-1).cpu().numpy()
    trans_test_preds = torch.argmax(trans_model(test_x), dim=-1).cpu().numpy()

trans_val_metrics = evaluate_operational_metrics(val_meta, y_val, trans_val_preds)
trans_test_metrics = evaluate_operational_metrics(test_meta, y_test, trans_test_preds)

torch.save({"model_state_dict": best_trans_weights, "metrics": trans_test_metrics}, ckpt_dir / "pilot_transformer_w15_d42.pt")
print(f"Transformer (Test) — Macro-F1: {trans_test_metrics['macro_f1']:.4f}, Det: {trans_test_metrics['detection_rate_pct']:.1f}%, Lead: {trans_test_metrics['median_lead_time_min']:.1f}m, FA: {trans_test_metrics['false_alarms_per_1000_steps']:.1f}/1000")

# Record Resources
gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
total_runtime_sec = time.time() - start_time

results = {
    "experiment_metadata": {
        "dataset": "Google Borg ClusterData 2011 Chronological Cohort (PILOT ONLY)",
        "window_length_W": 15,
        "feature_count_d": 42,
        "sampling_interval_sec": 300,
        "prediction_horizons": {"H1_critical_min": 15, "H2_warning_min": 30},
        "alert_policy": "k = 3 consecutive flags",
        "random_seed": 42,
        "train_samples": int(len(train_meta)),
        "val_samples": int(len(val_meta)),
        "test_samples": int(len(test_meta)),
        "device": str(device),
        "total_runtime_seconds": float(total_runtime_sec),
        "gpu_max_allocated_mb": float(gpu_mem_mb)
    },
    "models": {
        "random_forest_baseline": {
            "runtime_seconds": float(rf_time),
            "val_metrics": rf_val_metrics,
            "test_metrics": rf_test_metrics
        },
        "stacked_lstm_aligned": {
            "config": {"input_dim": 42, "hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "loss": "FocalLoss(gamma=2.0)"},
            "runtime_seconds": float(lstm_time),
            "val_metrics": lstm_val_metrics,
            "test_metrics": lstm_test_metrics
        },
        "transformer_aligned": {
            "config": {"input_dim": 42, "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.2, "loss": "FocalLoss(gamma=2.0)"},
            "runtime_seconds": float(trans_time),
            "val_metrics": trans_val_metrics,
            "test_metrics": trans_test_metrics
        }
    }
}

with open(exp_dir / "pilot_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nPilot Experiment Successfully Completed! Metrics saved to:", exp_dir / "pilot_metrics.json")
