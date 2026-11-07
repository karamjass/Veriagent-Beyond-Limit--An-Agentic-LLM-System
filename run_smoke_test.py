"""
Smoke test: N=40 (seed 42) for configs 1, 9, 12.
Stale caches for cfg9 and cfg12 were deleted by patch_tasks_ab.py;
runs start fresh with the fixed text truncation and compliance checks.
"""
import json
import model_core

print("=== Running N=40 Smoke Test (Seed 42) ===")

# Config 1: Solo Generator Baseline
print("Running Config 1: Solo Generator Baseline...")
res1 = model_core.run_benchmark_eval(config_id=1, limit=40, seed=42)
print(f"Config 1: {res1['correct']}/{res1['sample_size']} ({res1['accuracy_pct']}%), "
      f"CI95={res1['ci_95']}, Latency={res1['avg_latency_sec']}s")

baseline_preds = res1["predictions_map"]

# Config 9: Structured-prompt only
print("\nRunning Config 9: Structured Prompt Only...")
res9 = model_core.run_benchmark_eval(config_id=9, limit=40, seed=42, baseline_predictions=baseline_preds)
print(f"Config 9: {res9['correct']}/{res9['sample_size']} ({res9['accuracy_pct']}%), "
      f"Fixed={res9['fixed']}, Broken={res9['broken']}, "
      f"Compliance={res9['format_compliance_pct']}%, Latency={res9['avg_latency_sec']}s")

# Config 12: + PoT / SymPy
print("\nRunning Config 12: Structured Prompt + SymPy PoT...")
res12 = model_core.run_benchmark_eval(config_id=12, limit=40, seed=42, baseline_predictions=baseline_preds)
print(f"Config 12: {res12['correct']}/{res12['sample_size']} ({res12['accuracy_pct']}%), "
      f"Fixed={res12['fixed']}, Broken={res12['broken']}, "
      f"Compliance={res12['format_compliance_pct']}%, Latency={res12['avg_latency_sec']}s")

summary = {"config_1": res1, "config_9": res9, "config_12": res12}
with open("results/smoke_test_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSmoke test complete. Saved to results/smoke_test_summary.json")
