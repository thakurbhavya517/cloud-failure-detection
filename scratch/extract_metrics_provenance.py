import json
from pathlib import Path

files = [
    'results/phase7d/validation_metrics.json',
    'results/phase7d/test_evaluation_results.json',
    'results/phase6/phase6_4_confirmatory_metrics.json',
    'results/phase8_research_integrity/phase8_dataset_integrity_audit.json'
]

for p in files:
    f = Path(p)
    if f.exists():
        print(f"\n=== {p} ===")
        with open(f, 'r', encoding='utf-8') as fl:
            data = json.load(fl)
        
        if 'models' in data:
            for m, vals in data['models'].items():
                print(f"  {m}: Macro-F1 = {vals.get('macro_f1')}, Accuracy = {vals.get('accuracy')}")
        elif 'model_metrics' in data:
            for m, vals in data['model_metrics'].items():
                print(f"  {m}: Macro-F1 = {vals.get('macro_f1')}, Accuracy = {vals.get('accuracy')}")
        elif 'models_comparison' in data:
            for m, vals in data['models_comparison'].items():
                print(f"  {m}: Macro-F1 = {vals.get('macro_f1')}, Episode-Recall = {vals.get('episode_recall_pct')}")
        elif 'phase7d_validation_results' in data:
            for m, vals in data['phase7d_validation_results'].items():
                print(f"  {m}: Macro-F1 = {vals.get('macro_f1')}")
        else:
            print("  Top-level keys:", list(data.keys())[:10])
            for k in list(data.keys())[:5]:
                val = data[k]
                if isinstance(val, dict):
                    print(f"    Subkey {k}: {list(val.keys())[:5]}")

