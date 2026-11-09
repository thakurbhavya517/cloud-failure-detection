"""Generate publication-quality figures for the cloud-failure-detection research project.

All plots are generated strictly from genuine empirical data documented in
results/phase8_research_integrity/phase8_dataset_integrity_audit.json,
results/phase6/confirmatory_metrics_l5.json, and Phase 7 experimental reports.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Candidate A Validation Benchmark Comparison
# ---------------------------------------------------------------------------
def plot_model_comparison():
    models = [
        "Random Forest\n(M*)",
        "Temporal LSTM\n(Unweighted)",
        "Temporal LSTM\n(Weighted)",
        "Compact\nTransformer",
        "Representation\nHybrid",
        "Relative S2\n(MAD Norm)",
        "Linear Blend\n(0.8 S2 + 0.2 S0)",
        "Mamdani Fuzzy\nFIS",
    ]
    f1_scores = [0.3653, 0.3068, 0.3409, 0.3463, 0.2880, 0.3685, 0.3702, 0.3650]
    ep_recalls = [42.86, 0.0, 10.00, 30.00, 80.95, 43.33, 36.67, 66.67]
    crit_recalls = [23.64, 0.0, 8.16, 10.20, 58.18, 22.45, 17.01, 27.21]
    window_fprs = [11.18, 0.0, 6.79, 8.57, 18.57, 10.30, 4.40, 17.44]
    entity_fars = [65.79, 0.0, 16.33, 20.41, 57.89, 37.41, 24.49, 48.30]

    x = np.arange(len(models))
    width = 0.18

    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    # Recall and False Alarm rates (Percentages)
    r1 = ax1.bar(x - 1.5 * width, ep_recalls, width, label="Episode Recall (%)", color="#1f77b4", alpha=0.9)
    r2 = ax1.bar(x - 0.5 * width, crit_recalls, width, label="Critical Recall (%)", color="#2ca02c", alpha=0.9)
    r3 = ax1.bar(x + 0.5 * width, entity_fars, width, label="Entity False Alarm Rate (%)", color="#d62728", alpha=0.9)
    r4 = ax1.bar(x + 1.5 * width, window_fprs, width, label="Window False Positive Rate (%)", color="#ff7f0e", alpha=0.9)

    # Operational Gates
    ax1.axhline(70.0, color="#1f77b4", linestyle="--", linewidth=1.5, label="Gate 1a: Episode Recall ≥ 70%")
    ax1.axhline(50.0, color="#2ca02c", linestyle="--", linewidth=1.5, label="Gate 1b: Critical Recall ≥ 50%")
    ax1.axhline(10.0, color="#d62728", linestyle=":", linewidth=1.8, label="Gate 2a: Entity FAR ≤ 10%")
    ax1.axhline(5.0, color="#ff7f0e", linestyle=":", linewidth=1.8, label="Gate 2b: Window FPR ≤ 5%")

    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.set_title("Operational Gate Metrics Across Evaluated Architectures (Candidate A Validation)", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=0, ha="center")
    ax1.set_ylim(0, 100)

    # Secondary axis for Macro-F1
    ax2 = ax1.twinx()
    ax2.plot(x, f1_scores, color="#9467bd", marker="D", linewidth=2.5, markersize=8, label="Macro-F1 (right axis)")
    ax2.set_ylabel("Macro-F1", color="#9467bd", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_ylim(0.20, 0.50)
    ax2.grid(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=2, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance_comparison.png")
    plt.close()
    print("Saved model_performance_comparison.png")

# ---------------------------------------------------------------------------
# Figure 2: The Sensitivity-Specificity Dead Zone (Operational Gate Trade-off)
# ---------------------------------------------------------------------------
def plot_operational_tradeoff():
    models = {
        "Random Forest": (42.86, 65.79, "#8c564b"),
        "Temporal LSTM (Unweighted)": (0.0, 0.0, "#7f7f7f"),
        "Temporal LSTM (Weighted)": (10.0, 16.33, "#1f77b4"),
        "Compact Transformer": (30.0, 20.41, "#ff7f0e"),
        "Representation Hybrid": (80.95, 57.89, "#e377c2"),
        "Relative S2 (MAD Norm)": (43.33, 37.41, "#2ca02c"),
        "Linear Blend (0.8 S2 + 0.2 S0)": (36.67, 24.49, "#17becf"),
        "Mamdani Fuzzy FIS": (66.67, 48.30, "#bcbd22"),
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Acceptable operational region (Gate 1: Recall >= 70%, Gate 2: FAR <= 10%)
    ax.axhspan(70, 100, xmin=0, xmax=0.10, color="green", alpha=0.15, label="Operational Success Zone (Gates 1 & 2)")
    ax.axhline(70, color="blue", linestyle="--", alpha=0.7, label="Gate 1 Threshold: Episode Recall ≥ 70%")
    ax.axvline(10, color="red", linestyle="--", alpha=0.7, label="Gate 2 Threshold: Entity FAR ≤ 10%")

    for name, (rec, far, color) in models.items():
        ax.scatter(far, rec, color=color, s=150, zorder=5, edgecolors="black", linewidth=1.2)
        offset_y = 2.5 if name != "Representation Hybrid" else -4.0
        offset_x = 1.0 if name != "Temporal LSTM (Unweighted)" else 1.5
        ax.annotate(name, (far + offset_x, rec + offset_y), fontsize=9.5, weight="bold" if "Blend" in name or "S2" in name else "normal")

    ax.set_xlabel("Pure-Normal Entity False Alarm Rate (%) [Lower is Better]", fontsize=12)
    ax.set_ylabel("Episode Recall (%) [Higher is Better]", fontsize=12)
    ax.set_title("Operational Reliability Trade-Off: Recall vs. False Alarm Rate\n(Demonstrating the Sensitivity-Specificity Dead Zone)", pad=15)
    ax.set_xlim(-2, 75)
    ax.set_ylim(-5, 100)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "operational_gates_tradeoff.png")
    plt.close()
    print("Saved operational_gates_tradeoff.png")

# ---------------------------------------------------------------------------
# Figure 3: Relative Telemetry Breakthrough (Simpson's Paradox Resolution)
# ---------------------------------------------------------------------------
def plot_relative_vs_absolute():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Metric 1: Healthy Batch Job 6251277720 False Alarm Rate
    conditions = ["Absolute Telemetry\n(S0 Baseline)", "Combined Telemetry\n(S1 Representation)", "Relative Telemetry\n(S2 MAD-Normalized)"]
    job_far = [90.74, 90.11, 6.10]
    colors_job = ["#d62728", "#ff7f0e", "#2ca02c"]

    bars1 = ax1.bar(conditions, job_far, color=colors_job, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("False Alarm Rate on Job 6251277720 (%)", fontsize=11)
    ax1.set_title("False Alarm Rate on Healthy High-Resource Workload\n(Job 6251277720: 940 Pure-Normal Windows)", fontsize=12)
    ax1.set_ylim(0, 105)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=11, weight="bold")
    ax1.axhline(10.0, color="gray", linestyle=":", label="Operational Ceiling (10%)")
    ax1.legend(loc="upper right")

    # Metric 2: ROC-AUC Generalization (Train-OOF vs. Validation)
    x = np.arange(len(conditions))
    width = 0.35
    train_auc = [0.3066, 0.3139, 0.5080]
    val_auc = [0.4244, 0.4435, 0.5864]

    rects1 = ax2.bar(x - width/2, train_auc, width, label="Train Out-of-Fold ROC-AUC", color="#1f77b4", edgecolor="black")
    rects2 = ax2.bar(x + width/2, val_auc, width, label="Validation ROC-AUC", color="#9467bd", edgecolor="black")

    ax2.set_ylabel("Binary Failure ROC-AUC", fontsize=11)
    ax2.set_title("Cross-Job Generalization ROC-AUC Improvement\n(Overcoming Inverted Predictive Scores)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.set_ylim(0, 0.75)
    ax2.axhline(0.50, color="red", linestyle="--", label="Random Guess Baseline (0.50)")
    ax2.legend(loc="upper left")

    for bar in rects1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=9.5)
    for bar in rects2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "relative_vs_absolute_telemetry.png")
    plt.close()
    print("Saved relative_vs_absolute_telemetry.png")

# ---------------------------------------------------------------------------
# Figure 4: Historical Phase 6 Confirmatory Holdout Comparison
# ---------------------------------------------------------------------------
def plot_phase6_confirmatory():
    models = ["Random Forest", "Temporal LSTM", "Compact Transformer", "Representation Hybrid"]
    macro_f1 = [0.6111, 0.4425, 0.4479, 0.3924]
    accuracy = [78.06, 70.95, 76.21, 64.09]
    window_fpr = [21.39, 26.88, 20.27, 34.65]
    entity_far = [35.35, 41.92, 31.82, 64.14]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(models))
    width = 0.20

    ax.bar(x - 1.5 * width, [f * 100 for f in macro_f1], width, label="Macro-F1 (×100)", color="#3b528b", edgecolor="black")
    ax.bar(x - 0.5 * width, accuracy, width, label="Overall Accuracy (%)", color="#21918c", edgecolor="black")
    ax.bar(x + 0.5 * width, window_fpr, width, label="Window FPR (%)", color="#fde725", edgecolor="black")
    ax.bar(x + 1.5 * width, entity_far, width, label="Pure-Normal Entity FAR (%)", color="#e41a1c", edgecolor="black")

    ax.axhline(5.0, color="#fde725", linestyle="--", linewidth=1.5, label="Window FPR Ceiling (≤ 5%)")
    ax.axhline(12.0, color="#e41a1c", linestyle="--", linewidth=1.5, label="Entity FAR Ceiling (≤ 12%)")

    ax.set_ylabel("Percentage / Scaled Score", fontsize=11)
    ax.set_title("Phase 6 Confirmatory Holdout Evaluation (5,511 Quarantined Windows)\nClassical Tree Baselines Outperformed Deep Sequential Models", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_confirmatory_evaluation.png")
    plt.close()
    print("Saved phase6_confirmatory_evaluation.png")

if __name__ == "__main__":
    plot_model_comparison()
    plot_operational_tradeoff()
    plot_relative_vs_absolute()
    plot_phase6_confirmatory()
    print("All publication figures successfully generated.")

