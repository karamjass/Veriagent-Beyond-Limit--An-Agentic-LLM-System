import json
import os
import glob

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # dedupe by index, keeping last
    deduped = {}
    for r in rows:
        deduped[r.get("index", r.get("id"))] = r
    return list(deduped.values())

print("=== 1. GSM8K RAW FILES ===")
gsm8k_files = [
    ("cfg1", "results/raw/gsm8k_cfg1_N40_seed42.jsonl"),
    ("cfg2", "results/raw/gsm8k_cfg2_N40_seed42.jsonl"),
    ("cfg3", "results/raw/gsm8k_cfg3_N40_seed42.jsonl"),
    ("cfg4", "results/raw/gsm8k_cfg4_N40_seed42.jsonl"),
    ("cfg9", "results/raw/gsm8k_cfg9_N40_seed42.jsonl"),
    ("cfg12", "results/raw/gsm8k_cfg12_N40_seed42.jsonl"),
]

for cfg, path in gsm8k_files:
    rows = load_jsonl(path)
    n = len(rows)
    correct = sum(1 for r in rows if r.get("is_correct") is True)
    acc = (correct / n) * 100 if n else 0
    latencies = [r.get("latency", 0) for r in rows if "latency" in r]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    llm_calls = [r.get("llm_calls", 0) for r in rows if "llm_calls" in r]
    avg_calls = sum(llm_calls) / len(llm_calls) if llm_calls else 0
    backend = set(r.get("backend") for r in rows)
    print(f"{cfg}: N={n}, correct={correct}/{n} ({acc:.1f}%), avg_lat={avg_lat:.2f}s, avg_calls={avg_calls:.2f}, backend={backend}")

print("\n=== 2. HALLUCINATION PROBE RAW FILES ===")
raw_probe_rows = load_jsonl("results/hallucination_probe_raw.jsonl")
print(f"hallucination_probe_raw.jsonl: total rows={len(raw_probe_rows)}")
false_p = [r for r in raw_probe_rows if r.get("type") == "false_premise"]
ctrls = [r for r in raw_probe_rows if r.get("type") == "control"]
print(f"  false_premise count={len(false_p)}, controls count={len(ctrls)}")
fp_rule_halluc = sum(1 for r in false_p if r.get("rule_label") == "HALLUCINATED")
fp_llama3_halluc = sum(1 for r in false_p if r.get("llama3_label") == "HALLUCINATED")
ctrl_rule_over_ref = sum(1 for r in ctrls if r.get("rule_label") == "REFUSED")
ctrl_llama3_over_ref = sum(1 for r in ctrls if r.get("llama3_label") == "REFUSED")
print(f"  Raw probe: FP rule halluc={fp_rule_halluc}/{len(false_p)}, LLaMA3 halluc={fp_llama3_halluc}/{len(false_p)}")
print(f"  Raw probe: Control rule over-ref={ctrl_rule_over_ref}/{len(ctrls)}, LLaMA3 over-ref={ctrl_llama3_over_ref}/{len(ctrls)}")

print("\n--- Probe Baseline N30 ---")
base_probe = load_jsonl("results/raw/probe_baseline_N30.jsonl")
base_fp = [r for r in base_probe if r.get("type") == "false_premise"]
base_ctrl = [r for r in base_probe if r.get("type") == "control"]
base_fp_halluc = sum(1 for r in base_fp if r.get("llama3_label") == "HALLUCINATED" or r.get("rule_label") == "HALLUCINATED")
base_ctrl_over_ref = sum(1 for r in base_ctrl if r.get("llama3_label") == "REFUSED" and r.get("rule_label") == "REFUSED")
base_lat = sum(r.get("latency", 0) for r in base_probe) / len(base_probe)
base_calls = sum(r.get("llm_calls", 0) for r in base_probe) / len(base_probe)
print(f"Baseline N30: N={len(base_probe)}, FP halluc={base_fp_halluc}/{len(base_fp)}, Ctrl over-ref={base_ctrl_over_ref}/{len(base_ctrl)}, lat={base_lat:.2f}s, calls={base_calls:.1f}")

print("\n--- Probe Pipeline N30 ---")
pipe_probe = load_jsonl("results/raw/probe_pipeline_N30.jsonl")
pipe_fp = [r for r in pipe_probe if r.get("type") == "false_premise"]
pipe_ctrl = [r for r in pipe_probe if r.get("type") == "control"]
pipe_fp_halluc = sum(1 for r in pipe_fp if r.get("llama3_label") == "HALLUCINATED" or r.get("rule_label") == "HALLUCINATED")
pipe_ctrl_over_ref = sum(1 for r in pipe_ctrl if r.get("llama3_label") == "REFUSED" and r.get("rule_label") == "REFUSED")
pipe_lat = sum(r.get("latency", 0) for r in pipe_probe) / len(pipe_probe)
pipe_calls = sum(r.get("llm_calls", 0) for r in pipe_probe) / len(pipe_probe)
print(f"Pipeline N30: N={len(pipe_probe)}, FP halluc={pipe_fp_halluc}/{len(pipe_fp)}, Ctrl over-ref={pipe_ctrl_over_ref}/{len(pipe_ctrl)}, lat={pipe_lat:.2f}s, calls={pipe_calls:.1f}")

print("\n=== 3. TRUTHFULQA MC RAW FILES ===")
tqa_base = load_jsonl("results/raw/truthfulqa_baseline_N40_seed42.jsonl")
tqa_perm = load_jsonl("results/raw/truthfulqa_perm_N40_seed42.jsonl")
base_corr = sum(1 for r in tqa_base if r.get("is_correct") is True)
perm_corr = sum(1 for r in tqa_perm if r.get("is_correct") is True)
base_lat = sum(r.get("latency", 0) for r in tqa_base) / len(tqa_base)
perm_lat = sum(r.get("latency", 0) for r in tqa_perm) / len(tqa_perm)
uncertain = sum(1 for r in tqa_perm if r.get("prediction") == "uncertain" or r.get("is_uncertain") is True)
# per-permutation accuracy
p0 = sum(1 for r in tqa_perm if r.get("perm_results", [{}])[0].get("is_correct") is True) if "perm_results" in tqa_perm[0] else None
p1 = sum(1 for r in tqa_perm if len(r.get("perm_results", [])) > 1 and r.get("perm_results")[1].get("is_correct") is True) if "perm_results" in tqa_perm[0] else None
p2 = sum(1 for r in tqa_perm if len(r.get("perm_results", [])) > 2 and r.get("perm_results")[2].get("is_correct") is True) if "perm_results" in tqa_perm[0] else None
print(f"TruthfulQA MC Base: N={len(tqa_base)}, correct={base_corr}/{len(tqa_base)} ({base_corr/len(tqa_base)*100:.1f}%), lat={base_lat:.2f}s")
print(f"TruthfulQA MC Perm: N={len(tqa_perm)}, correct={perm_corr}/{len(tqa_perm)} ({perm_corr/len(tqa_perm)*100:.1f}%), lat={perm_lat:.2f}s, uncertain={uncertain}/{len(tqa_perm)} ({uncertain/len(tqa_perm)*100:.1f}%)")
if p0 is not None:
    print(f"Per-permutation accuracies: [{p0/len(tqa_perm)*100:.1f}%, {p1/len(tqa_perm)*100:.1f}%, {p2/len(tqa_perm)*100:.1f}%]")

print("\n=== 4. TRUTHFULQA OPEN-ENDED RAW FILES ===")
oe_base = load_jsonl("results/raw/truthfulqa_openended_baseline_N40_seed42.jsonl")
oe_pipe = load_jsonl("results/raw/truthfulqa_openended_pipeline_N40_seed42.jsonl")

oe_base_truthful = sum(1 for r in oe_base if r.get("truthful") is True)
oe_base_info = sum(1 for r in oe_base if r.get("informative") is True)
oe_base_unc = sum(1 for r in oe_base if r.get("not_certain") is True)
oe_base_lat = sum(r.get("latency", 0) for r in oe_base) / len(oe_base)
oe_base_calls = sum(r.get("llm_calls", 0) for r in oe_base) / len(oe_base)

oe_pipe_truthful = sum(1 for r in oe_pipe if r.get("truthful") is True)
oe_pipe_info = sum(1 for r in oe_pipe if r.get("informative") is True)
oe_pipe_unc = sum(1 for r in oe_pipe if r.get("not_certain") is True)
oe_pipe_lat = sum(r.get("latency", 0) for r in oe_pipe) / len(oe_pipe)
oe_pipe_calls = sum(r.get("llm_calls", 0) for r in oe_pipe) / len(oe_pipe)

print(f"Open-Ended Baseline: N={len(oe_base)}, truthful={oe_base_truthful}/{len(oe_base)} ({oe_base_truthful/len(oe_base)*100:.1f}%), info={oe_base_info}/{len(oe_base)} ({oe_base_info/len(oe_base)*100:.1f}%), not_certain={oe_base_unc}/{len(oe_base)} ({oe_base_unc/len(oe_base)*100:.1f}%), lat={oe_base_lat:.2f}s, calls={oe_base_calls:.1f}")
print(f"Open-Ended Pipeline: N={len(oe_pipe)}, truthful={oe_pipe_truthful}/{len(oe_pipe)} ({oe_pipe_truthful/len(oe_pipe)*100:.1f}%), info={oe_pipe_info}/{len(oe_pipe)} ({oe_pipe_info/len(oe_pipe)*100:.1f}%), not_certain={oe_pipe_unc}/{len(oe_pipe)} ({oe_pipe_unc/len(oe_pipe)*100:.1f}%), lat={oe_pipe_lat:.2f}s, calls={oe_pipe_calls:.1f}")

