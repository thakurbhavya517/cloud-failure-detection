# Cloud Failure Detection — Research Web Dashboard

An interactive, production-grade web dashboard built for visual exploration, multi-architecture benchmarking, telemetry inspection, and live CPU inference for multivariate cloud container failure prediction.

The application connects directly to genuine trained model checkpoints, audited evaluation artifacts, and real-world Google Borg trace partitions.

---

## 1. System Architecture

```
dashboard/
├── app.py                      # Main entry point & multi-page router
├── README.md                   # Dashboard technical documentation
├── utils/
│   ├── data_loader.py          # Data ingestion & schema loader (holdout protected)
│   ├── model_loader.py         # ModelManager: CPU inference for RF, LSTM, Transformer, Hybrids
│   └── inference_pipeline.py   # Preprocessing, median imputation & sliding window slicer
└── views/
    ├── home.py                 # Executive overview, research headline & dataset status
    ├── model_comparison.py     # Benchmark tables, sensitivity-specificity trade-off, Simpson's paradox
    ├── telemetry_explorer.py   # Interactive Plotly time-series for 56 validation tasks
    ├── prediction_playground.py# Upload CSV / select task, live CPU inference & alert badges
    ├── evaluation_testing.py   # Test suite status, live 657-sequence validation benchmark
    └── research_info.py        # Methodology, sequence length audit, BibTeX citation
```

---

## 2. Hardware & Runtime Feasibility

The dashboard is engineered for high efficiency on commodity developer laptops:
- **Target OS:** Windows 11 (compatible with Linux/macOS)
- **Minimum Requirements:** Intel Core i5 / AMD Ryzen 5, 8 GB RAM.
- **Compute Engine:** **100% CPU execution** (`torch.device("cpu")`). No GPU or CUDA installation is required.
- **Framework:** Streamlit + Plotly + scikit-learn + PyTorch.

---

## 3. Installation & Environment Setup

All UI dependencies are pinned in the repository's root [`requirements.txt`](../requirements.txt).

```powershell
# 1. Activate your virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 2. Verify installed packages
python -m pip install -r requirements.txt
```

Core libraries used:
- `streamlit>=1.30.0`
- `plotly>=5.18.0`
- `torch>=2.0.0` (CPU build)
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `joblib>=1.3.0`

---

## 4. How to Launch the Dashboard

From the repository root (`cloud-failure-detection`):

```powershell
streamlit run dashboard/app.py
```

The application will start and open automatically in your default browser at:
`http://localhost:8501`

---

## 5. Dashboard Pages & Capabilities

### 🏠 A. Home Dashboard
- **Executive Headline:** Overview of the empirical findings and negative results.
- **Audited Metrics Cards:** Best Macro-F1 (0.6111 RF), Minimum Window FPR (4.40% Linear Blend), Maximum Episode Recall (80.95% Rep Hybrid).
- **Dataset Provenance Table:** Breakdown of Candidate A (Clean Train/Val/Test), Candidate B (Holdout), and Quarantined Confirmatory partitions.

### 📊 B. Model Comparison
- **Benchmark Table:** Comprehensive evaluation of 8 candidate architectures across 7 metrics.
- **Interactive Trade-Off Plot:** Scatter visualization of Episode Recall (Gate 1) vs. Pure-Normal False Alarm Rate (Gate 2) highlighting the operational "Dead Zone".
- **Simpson's Paradox Resolution:** Deep dive into heavy batch Job `6251277720`, illustrating how relative telemetry ($S_2$) collapsed false alarms from $90.74\%$ to $6.10\%$.
- **Phase 6 Holdout Table:** Blinded evaluation results on 5,511 quarantined windows.

### 📈 C. Telemetry Explorer
- **Interactive Task Selector:** Browse 56 real-world container tasks from Candidate A Validation.
- **Multi-Metric Time-Series:** Synchronized interactive Plotly plots of CPU Utilization, Memory Utilization, Cycles Per Instruction (CPI), and Disk I/O.
- **Label State Annotations:** Visual indicators distinguishing NORMAL, WARNING, and CRITICAL runtime stages.
- **Summary Statistics:** Window counts, mean utilization, peak spikes, and task durations.

### ⚡ D. Prediction Playground
- **Triple-Mode Input:**
  1. *Built-In Validation Task:* Select an actual Google Borg container sequence to evaluate.
  2. *Upload Custom Telemetry CSV:* Validate and predict on user-provided telemetry files.
  3. *Use Default 5-Step Sample:* Quick test with pre-packaged canonical telemetry.
- **Model Selector:** Choose between Random Forest, Temporal LSTM, Compact Transformer, Representation Hybrid, or Linear Hybrid Blend.
- **Live CPU Inference:** Computes predicted alert status (`NORMAL`, `WARNING`, `CRITICAL`) and exact multi-class probabilities.
- **Downloadable Predictions:** Export formatted predictions and probability distributions as CSV.

### 🧪 E. Evaluation & Testing
- **Test Suite Verification:** Summary of the 213-test automated pytest suite (196 passing, 17 legacy audit path errors, 0 logic errors).
- **Live Non-Destructive Benchmark:** Execute live inference on all 657 Candidate A validation sequences to generate a real-time confusion matrix and classification report on CPU.
- **Operational Gates Status:** Gate 1a, Gate 1b, Gate 2a, Gate 2b pre-registered criteria verification.
- **Quarantine Safeguards:** Cryptographic isolation status of `data/processed/confirmatory/`.

### 📚 F. Research Information
- **Theoretical Foundations:** Problem statement, research questions, and experimental protocols.
- **Empirical Breakthroughs:** Sequence length collapse ($L=15$ vs $L=5$), post-failure censoring leakage elimination, and tabular vs. deep sequential dynamics.
- **BibTeX Citation:** Ready-to-copy academic citation snippet.

---

## 6. Supported Input CSV Schema

For custom predictions in the **Prediction Playground**, uploaded CSVs must contain the **31 canonical telemetry features** derived during Candidate A feature engineering:

| Feature Name | Description |
| :--- | :--- |
| `cpu_utilization` | Average CPU core utilization |
| `memory_utilization` | Ratio of resident memory to machine capacity |
| `assigned_memory` | Memory allocation limit |
| `disk_io_time` | Cumulative disk I/O time |
| `cycles_per_instruction` | CPU CPI metric |
| `memory_accesses_per_instruction` | Memory bus intensity |
| `cpu_utilization_lag_1..3,6` | Historical CPU observations |
| `memory_utilization_lag_1..3,6` | Historical memory observations |
| `cpu_utilization_diff_1..3` | Discrete first-difference derivatives |
| `cpu_roll_mean_3,6`, `cpu_roll_std_3,6`, `cpu_roll_range_3,6` | Rolling temporal moments |
| `mem_roll_mean_3,6`, `mem_roll_std_3,6`, `mem_roll_range_3,6` | Rolling memory moments |
| `cpu_trend_slope_6` | 6-step linear regression slope |
| `cpu_exceed_85`, `cpu_exceed_fraction_6` | Extreme threshold indicators |
| `memory_to_assigned_ratio` | Dynamic memory headroom |
| `task_age_minutes` | Execution lifetime |
| `cpi_missing_fraction_6` | Missing metric indicator |

### Preprocessing & Validation Behavior:
- **Missing Telemetry / NaNs:** Automatically imputed using strictly training-derived medians from `configs/phase7c/fitted_scaler_phase7c.json`.
- **Temporal Slicing:** Automatically formats inputs into 5-step sliding sequences (`N - 4, 5, 31`). Single rows are expanded across the temporal axis for instantaneous assessment.
- **Scaling:** Raw unscaled data can be automatically standardized using training-set StandardScaler parameters by enabling the checkbox.

---

## 7. Research Integrity & Safeguards

1. **No Data Leakage:** The dashboard strictly accesses only `data/processed/phase7c/val_metadata.csv`, `X_val.npy`, and `y_val.npy`. Quarantined holdouts (`data/processed/confirmatory/`) and final test arrays (`X_test.npy`) are protected and inaccessible.
2. **No Model Retraining:** Models are loaded strictly in evaluation mode (`model.eval()`) with frozen weights.
3. **No Fabricated Outputs:** All displayed metrics, probabilities, and confusion matrices are either calculated directly from model weights or loaded from audited Phase 8 JSON reports.

