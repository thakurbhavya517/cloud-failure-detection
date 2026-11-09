import json

p8_file = r'c:\AntiGravity Files\cloud-failure-detection\results\phase8_research_integrity\phase8_dataset_integrity_audit.json'
with open(p8_file, encoding='utf-8') as f:
    p8 = json.load(f)

print('=== PHASE 8 MODELS SUMMARY ===')
for m in p8.get('models_comparison', []):
    name = m.get('model_architecture')
    f1 = m.get('val_macro_f1')
    ep_rec = m.get('episode_recall_pct')
    crit_rec = m.get('genuine_critical_recall_pct')
    far = m.get('pure_normal_entity_far_pct')
    fpr = m.get('normal_window_fpr_pct')
    print(f"{name:45s} | F1: {f1} | EpRec: {ep_rec}% | CritRec: {crit_rec}% | FAR: {far}% | FPR: {fpr}%")

print('\n=== PHASE 6 CONFIRMATORY METRICS ===')
with open(r'c:\AntiGravity Files\cloud-failure-detection\results\phase6\confirmatory_metrics_l5.json', encoding='utf-8') as f:
    p6 = json.load(f)
for model, metrics in p6.items():
    if isinstance(metrics, dict):
        print(f"Model: {model}")
        print(f"  Macro-F1: {metrics.get('macro_f1')}")
        print(f"  Accuracy: {metrics.get('accuracy')}")
        print(f"  Window FPR: {metrics.get('window_fpr')}")
        print(f"  Entity FAR: {metrics.get('entity_far')}")
        print(f"  Episode Recall: {metrics.get('episode_recall')}")
        print(f"  Critical Recall: {metrics.get('critical_recall')}")
