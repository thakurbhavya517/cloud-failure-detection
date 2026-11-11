"""Extract 11 additional features (Groups J1, J2, J3) for Chronological W=15 dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(".").resolve()
out_dir = PROJECT_ROOT / "data" / "processed" / "pilot_w15_d42"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Chronological metadata
train_meta = pd.read_csv("data/processed/chronological/train_metadata.csv")
val_meta = pd.read_csv("data/processed/chronological/val_metadata.csv")
test_meta = pd.read_csv("data/processed/chronological/test_metadata.csv")

all_meta = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)
all_eids = set(all_meta["entity_id"].unique())
print(f"Total unique entities across Train, Val, Test: {len(all_eids)}")

# 2. Invert baseline 31 features to raw scale
with open("configs/fitted_scaler.json") as f:
    scaler_data = json.load(f)
feat_cols = scaler_data["feature_cols"]
means = np.array([scaler_data["means"][k] for k in feat_cols], dtype=np.float32)
stds = np.array([scaler_data["stds"][k] for k in feat_cols], dtype=np.float32)

X_tr_31_raw = np.load("data/processed/chronological/X_train.npy") * stds + means
X_v_31_raw = np.load("data/processed/chronological/X_val.npy") * stds + means
X_te_31_raw = np.load("data/processed/chronological/X_test.npy") * stds + means

# 3. Ingest Events for J2 (Scheduler Metadata) and J3 (Preemption Events)
EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]
event_shards = [
    PROJECT_ROOT / "data" / "raw" / "sample_task_events.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_events_part1.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_events_part2.csv.gz",
]
events_list = []
for s_path in event_shards:
    if s_path.exists():
        df = pd.read_csv(s_path, header=None, names=EVENT_COLS)
        df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
        matched = df[df["entity_id"].isin(all_eids)].copy()
        events_list.append(matched)

all_events = pd.concat(events_list, ignore_index=True)
all_events["time_sec"] = all_events["timestamp"] / 1e6
events_by_eid = all_events.groupby("entity_id")

# Static attributes dict (J2)
static_dict = {}
for eid, grp in events_by_eid:
    prio = float(grp["priority"].dropna().iloc[0]) / 10.0 if not grp["priority"].dropna().empty else 0.0
    sched = float(grp["scheduling_class"].dropna().iloc[0]) / 3.0 if not grp["scheduling_class"].dropna().empty else 0.0
    rcpu = float(grp["req_cpu"].dropna().iloc[0]) if not grp["req_cpu"].dropna().empty else 0.0
    rram = float(grp["req_ram"].dropna().iloc[0]) if not grp["req_ram"].dropna().empty else 0.0
    static_dict[eid] = np.array([prio, sched, rcpu, rram], dtype=np.float32)

# Default for any entity missing static events
default_static = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# 4. Ingest Usage for J1 (Peak Telemetry & Headroom)
USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]
usage_shards = [
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage.csv.gz",
    PROJECT_ROOT / "data" / "raw" / "sample_task_usage_part1.csv.gz",
]
usage_chunks = []
for s_path in usage_shards:
    if s_path.exists():
        for chunk in pd.read_csv(s_path, header=None, names=USAGE_COLS, chunksize=500000):
            chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
            matched = chunk[chunk["entity_id"].isin(all_eids)].copy()
            if not matched.empty:
                usage_chunks.append(matched)

usage_df = pd.concat(usage_chunks, ignore_index=True)
usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

# Resample to 300s bins
agg_cols = {
    "cpu_rate": "mean",
    "canonical_memory": "mean",
    "assigned_memory": "mean",
    "max_cpu_rate": "mean",
    "max_memory": "mean",
    "total_page_cache": "mean",
}
resampled = usage_df.groupby(["entity_id", "grid_bin"], as_index=False).agg(agg_cols)
resampled["cpu_burstiness"] = resampled["max_cpu_rate"] - resampled["cpu_rate"]
resampled["memory_headroom"] = resampled["assigned_memory"] - resampled["max_memory"]

j1_cols = ["max_cpu_rate", "cpu_burstiness", "max_memory", "memory_headroom", "total_page_cache"]
resampled["key"] = resampled["entity_id"] + "@" + resampled["grid_bin"].astype(str)
j1_lookup = dict(zip(resampled["key"], resampled[j1_cols].values))

# 5. Tensor Builder for W=15
def build_w15_additional_tensors(meta_df: pd.DataFrame):
    n = len(meta_df)
    j1_tensor = np.zeros((n, 15, 5), dtype=np.float32)
    j2_tensor = np.zeros((n, 15, 4), dtype=np.float32)
    j3_tensor = np.zeros((n, 15, 2), dtype=np.float32)

    for i, row in meta_df.iterrows():
        eid = row["entity_id"]
        t = row["prediction_timestamp"]
        
        # 15 trailing steps: [t - 14*300, ..., t]
        step_times = [t - 14.0 * 300.0 + step * 300.0 for step in range(15)]
        
        # J2 Static metadata
        static_feats = static_dict.get(eid, default_static)
        for step in range(15):
            j2_tensor[i, step, :] = static_feats
            
        # J1 Usage & J3 Events at each step
        evs = events_by_eid.get_group(eid) if eid in events_by_eid.groups else pd.DataFrame()
        for step, step_t in enumerate(step_times):
            k_key = f"{eid}@{step_t}"
            if k_key in j1_lookup:
                j1_tensor[i, step, :] = j1_lookup[k_key]
            else:
                j1_tensor[i, step, :] = 0.0
                
            # J3 events lookup strictly <= step_t
            if not evs.empty:
                causal_evs = evs[evs["time_sec"] <= step_t]
                num_evicts = float((causal_evs["event_type"] == 2).sum())
                sched_evs = causal_evs[causal_evs["event_type"] == 1]
                last_sched = sched_evs["time_sec"].max() if not sched_evs.empty else step_t
                time_since_sched_min = float((step_t - last_sched) / 60.0)
            else:
                num_evicts = 0.0
                time_since_sched_min = 0.0
                
            j3_tensor[i, step, 0] = num_evicts
            j3_tensor[i, step, 1] = time_since_sched_min

    return j1_tensor, j2_tensor, j3_tensor

print("Building W=15 additional features for Train...")
tr_j1, tr_j2, tr_j3 = build_w15_additional_tensors(train_meta)
print(f"  Train: J1 {tr_j1.shape}, J2 {tr_j2.shape}, J3 {tr_j3.shape}")

print("Building W=15 additional features for Val...")
val_j1, val_j2, val_j3 = build_w15_additional_tensors(val_meta)
print(f"  Val: J1 {val_j1.shape}, J2 {val_j2.shape}, J3 {val_j3.shape}")

print("Building W=15 additional features for Test...")
te_j1, te_j2, te_j3 = build_w15_additional_tensors(test_meta)
print(f"  Test: J1 {te_j1.shape}, J2 {te_j2.shape}, J3 {te_j3.shape}")

# Combine into 42-feature raw tensors
X_tr_42_raw = np.concatenate([X_tr_31_raw, tr_j1, tr_j2, tr_j3], axis=2)
X_v_42_raw = np.concatenate([X_v_31_raw, val_j1, val_j2, val_j3], axis=2)
X_te_42_raw = np.concatenate([X_te_31_raw, te_j1, te_j2, te_j3], axis=2)

print(f"Combined raw 42-feature tensors: Train {X_tr_42_raw.shape}, Val {X_v_42_raw.shape}, Test {X_te_42_raw.shape}")

# Impute any remaining NaNs using Train median strictly
for f_idx in range(42):
    train_vals = X_tr_42_raw[:, :, f_idx]
    med = np.nanmedian(train_vals)
    if np.isnan(med):
        med = 0.0
    X_tr_42_raw[:, :, f_idx] = np.nan_to_num(X_tr_42_raw[:, :, f_idx], nan=med)
    X_v_42_raw[:, :, f_idx] = np.nan_to_num(X_v_42_raw[:, :, f_idx], nan=med)
    X_te_42_raw[:, :, f_idx] = np.nan_to_num(X_te_42_raw[:, :, f_idx], nan=med)

# Fit RobustScaler strictly on Train
n_tr, seq_len, n_feat = X_tr_42_raw.shape
scaler = RobustScaler()
X_tr_flat = X_tr_42_raw.reshape(-1, n_feat)
scaler.fit(X_tr_flat)

X_tr_42_scaled = scaler.transform(X_tr_flat).reshape(n_tr, seq_len, n_feat).astype(np.float32)
X_v_42_scaled = scaler.transform(X_v_42_raw.reshape(-1, n_feat)).reshape(len(val_meta), seq_len, n_feat).astype(np.float32)
X_te_42_scaled = scaler.transform(X_te_42_raw.reshape(-1, n_feat)).reshape(len(test_meta), seq_len, n_feat).astype(np.float32)

# Load labels
y_tr = np.load("data/processed/chronological/y_train.npy")
y_v = np.load("data/processed/chronological/y_val.npy")
y_te = np.load("data/processed/chronological/y_test.npy")

# Save artifacts
np.save(out_dir / "X_train.npy", X_tr_42_scaled)
np.save(out_dir / "X_val.npy", X_v_42_scaled)
np.save(out_dir / "X_test.npy", X_te_42_scaled)
np.save(out_dir / "y_train.npy", y_tr)
np.save(out_dir / "y_val.npy", y_v)
np.save(out_dir / "y_test.npy", y_te)
train_meta.to_csv(out_dir / "train_metadata.csv", index=False)
val_meta.to_csv(out_dir / "val_metadata.csv", index=False)
test_meta.to_csv(out_dir / "test_metadata.csv", index=False)

print("Saved all aligned W=15, d=42 tensors to:", out_dir)
print(f"Final shapes: Train={X_tr_42_scaled.shape}, Val={X_v_42_scaled.shape}, Test={X_te_42_scaled.shape}")
print(f"NaN checks: Train={np.isnan(X_tr_42_scaled).sum()}, Val={np.isnan(X_v_42_scaled).sum()}, Test={np.isnan(X_te_42_scaled).sum()}")

