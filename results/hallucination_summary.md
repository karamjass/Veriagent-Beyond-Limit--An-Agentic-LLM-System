# Hallucination Evaluation Summary

**Evaluation Environment**: Local Ollama (`qwen3.5:9b-q4_K_M` generator, `llama3:latest` independent judge)  
**Datasets**: Hand-crafted false-premise & control probe (`data/hallucination_probe.json`, $N=30$)  
**Methodology Note**: Neither the internal self-consistency check nor the cross-model LLaMA-3 check constitutes external ground truth. All LLM-based judgments represent proxy metrics, not human evaluation.

---

## 1. Task 1: Probe Audit & Multi-Judge Disagreement Analysis

The 30-question probe was executed with full, unconstrained answer text retention (saved in `results/hallucination_probe_raw.jsonl`). Each generation was independently assessed under two evaluation regimes without cross-override:
1. **Rule-Based Regex Scanner**: Evaluates exact phrase matches against `REFUSE_PATTERNS` and `CONFIDENT_PATTERNS`.
2. **Independent LLaMA-3 LLM Judge**: Assesses whether the model fabricated claims, refused appropriately, or correctly debunked false premises.

### Headline Probe Metrics ($N=30$)

| Evaluation System | False-Premise Hallucination (k/20) | False-Premise 95% Wilson CI | Control Over-Refusal (k/10) | Control 95% Wilson CI |
|:---|:---:|:---:|:---:|:---:|
| **Rule-Based Regex** | **0 / 20 (0.0%)** | [0.0%, 16.1%] | **0 / 10 (0.0%)** | [0.0%, 27.8%] |
| **LLaMA-3 Judge** | **0 / 20 (0.0%)** | [0.0%, 16.1%] | **0 / 10 (0.0%)** | [0.0%, 27.8%] |

### Label Disagreements (14 / 30 Questions)
* **Agreement Rate**: 16 / 30 (53.3% raw token agreement).
* **Nature of Disagreements**: Evaluator divergence occurred exclusively on false-premise questions due to label ontology:
  - On 14 questions, Qwen explicitly rejected the false premise (e.g., *"There was no such study"*, *"no such paper exists"*, *"General Fatima did not exist"*).
  - The Rule Scanner categorized explicit refutations as `REFUSED`.
  - The LLaMA-3 Judge categorized explicit refutations as `OK` (matching the judge rubric: *"correctly identified the false premise without inventing details"*).
* **Core Takeaway**: **Zero hallucinations were detected under either evaluator.** Both systems agree that Qwen 3.5 9B never affirmed any of the 20 false premises as factual.

---

## 2. Task 2: Raw Qwen Baseline vs. VeriAgent Pipeline ($N=30$)

Sequential benchmark execution comparing single-pass generator inference (Config 1 style) against the multi-agent VeriAgent pipeline (Draft $\to$ Fact-Checker $\to$ Targeted Reviser):

| Metric | Raw Qwen Baseline (`probe_baseline_N30.jsonl`) | VeriAgent Pipeline (`probe_pipeline_N30.jsonl`) | Comparison Verdict |
|:---|:---:|:---:|:---:|
| **False-Premise Hallucination Rate** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** | **Inconclusive** |
| 95% Wilson Score CI | [0.0%, 16.1%] | [0.0%, 16.1%] | (Identical point estimates; 0 pp diff) |
| **Control Over-Refusal Rate** | **0 / 10 (0.0%)** | **0 / 10 (0.0%)** | **Inconclusive** |
| 95% Wilson Score CI | [0.0%, 27.8%] | [0.0%, 27.8%] | (Identical point estimates; 0 pp diff) |
| **Average Latency (s/question)** | **8.60 s** | **13.68 s** | Pipeline is **1.59x slower** |
| **LLM Calls per Question** | **1.0** | **3.0** | 3x compute cost |

### Plain-Language Verdict: Inconclusive
* **Reason**: Both the solo generator and the multi-agent pipeline achieved ceiling performance ($0.0\%$ hallucinations on false premises, $0.0\%$ over-refusal on controls).
* Because the solo foundation model already explicitly recognized and debunked all 20 false premises, the multi-agent fact-checker and reviser loops had zero baseline hallucinations to correct.
* The pipeline introduced a $5.08\,\text{s}$ ($+59\%$) latency overhead and tripled token consumption without measurable accuracy or safety gains on this probe set.

---

## 3. TruthfulQA Validation Benchmark Status

### Historical 57.5% Comparison Audit
* A rigorous audit of repository artifacts, git commits, and cache directories confirms that **no raw cached rows, random seeds, or split definitions exist for the historical 57.5% figure**.
* The 57.5% ($23/40$) was referenced only as an earlier slide estimate; whether it evaluated generation choice vs. legacy log-likelihood MC1 scoring cannot be verified.
* **Standardization Policy**: Direct comparison to the unverified 57.5% figure is scientifically invalid. Legitimate benchmarking requires fresh sequential evaluation on pinned revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490` under identical sample sizes ($N=40$, seed 42).

### Current Status of TruthfulQA Runs
* **B1 Multiple-Choice (Original vs. 3-Permutation Voting)**: Infrastructure implemented (`evaluate_truthfulqa_permutation()`); **planned, awaiting execution**.
* **B2 Open-Ended Generation (LLaMA-3 Judge)**: Infrastructure implemented (`evaluate_truthfulqa_openended()`); **planned, awaiting execution**.

---

## 4. Ground-Truth & Provenance Disclaimer

> **Important Scientific Labeling**:
> 1. The VeriAgent Fact-Checker utilizes the same foundation model weights as the primary generator. It represents an **epistemic self-consistency audit**, not an external ground-truth validation.
> 2. The cross-model judge (`llama3:latest`) belongs to an independent parameter family, mitigating correlated generation blind spots. However, it remains an **LLM-judged proxy**, not human ground truth.
