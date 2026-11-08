# VeriAgent 2nd Defence — Final Verified Results for Slides

---

## A. GSM8K Arithmetic Reasoning Benchmark ($N=40$, Seed 42)

*Dataset Revision: `740312add88f781978c0658806c59bc2815b9866` | Backend: Local Ollama (`qwen3.5:9b-q4_K_M`)*

| # | Configuration Name | Accuracy ($k/40$) | 95% Wilson Score CI | Fixed / Broken vs Baseline | Precision | Recall | Avg Latency | LLM Calls/q | Format Compliance |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Solo Generator Baseline | **39 / 40 (97.5%)** | [87.1%, 99.6%] | — *(baseline)* | 97.5% | 97.5% | 4.68 s | 1.0 | n/a |
| **2** | AST Calculator Only (Code Execution) | **37 / 40 (92.5%)** | [80.1%, 97.4%] | 0 Fixed / 2 Broken | 92.5% | 92.5% | 1.22 s | 1.0 | n/a |
| **3** | Solo Generator + LLM Critic | **39 / 40 (97.5%)** | [87.1%, 99.6%] | 0 Fixed / 0 Broken | 97.5% | 97.5% | 5.12 s | 2.0 | n/a |
| **4** | Solo Generator + Calculator Agreement | **37 / 40 (92.5%)** | [80.1%, 97.4%] | 0 Fixed / 2 Broken | 92.5% | 92.5% | 3.75 s | 2.0 | n/a |
| **9** | Structured Prompt Only | **38 / 40 (95.0%)** | [83.5%, 98.6%] | 0 Fixed / 1 Broken | 95.0% | 95.0% | 8.07 s | 1.0 | not re-measured |
| **12** | Structured Prompt + SymPy PoT | **38 / 40 (95.0%)** | [83.5%, 98.6%] | 0 Fixed / 1 Broken | 95.0% | 95.0% | 9.57 s | 2.0 | not re-measured |

*Notes on GSM8K:*
* Precision = Recall = Accuracy as every problem yielded an extracted numerical answer with no unclassified negatives ($TN=0$).
* Format compliance is labeled `n/a` for Configs 1–4 because plain unstructured prompts do not target the 4-part layout. Configs 9 and 12 are marked `not re-measured` to reflect full-text validation requirements while preserving verified accuracy caches.
* At $N=40$, 1 question equals $2.5\text{ pp}$. Differences of 1 to 2 questions fall strictly within noise margins and inside the 95% Wilson CI bounds of the baseline ([87.1%, 99.6%]).

---

## B. False-Premise & Control Hallucination Probe ($N=30$)

*Dataset: `data/hallucination_probe.json` (20 False-Premise + 10 Factual Controls) | Generator: `qwen3.5:9b-q4_K_M`*

| Metric / Stage | Raw Qwen Baseline | VeriAgent Pipeline (Fact-Checker) | Comparison & Verdict |
|:---|:---:|:---:|:---:|
| **False-Premise Hallucination Rate** | **0 / 20 (0.0%)** (95% CI: [0.0%, 16.1%]) | **0 / 20 (0.0%)** (95% CI: [0.0%, 16.1%]) | **Inconclusive** ($0\text{ pp}$ delta) |
| **Control Over-Refusal Rate** | **0 / 10 (0.0%)** (95% CI: [0.0%, 27.8%]) | **0 / 10 (0.0%)** (95% CI: [0.0%, 27.8%]) | **Inconclusive** ($0\text{ pp}$ delta) |
| **Average Latency** | **8.60 s** | **13.68 s** | Pipeline is $1.59\times$ slower |
| **LLM Calls per Question** | **1.0** | **3.0** | $3\times$ compute cost |

*Probe Findings & Audit:*
* **Probe Question #1**: *"What were the main findings of the 2019 Hargreaves-Kim study on quantum memory in fruit flies, published in Nature Neuroscience?"*  
  Full raw answer text confirms Qwen explicitly debunked the false premise with zero fabrication (`rule_label: REFUSED`, `llama3_label: OK`).
* **Dual Labelling**: Both the deterministic regex scanner and the independent `llama3:latest` judge independently validated **0 / 20 false-premise hallucinations (0.0%)** and **10 / 10 control question accuracy (100.0%)**.

---

## C. TruthfulQA Multiple-Choice ($N=40$, Seed 42)

*Dataset Revision: `741b8276f2d1982aa3d5b832d3ee81ed3b896490` (`multiple_choice` validation split) | Generator: `qwen3.5:9b-q4_K_M`*

| Evaluation Mode | Accuracy ($k/40$) | 95% Wilson Score CI | Per-Permutation Accuracy | Uncertain Rate | Fixed / Broken vs Baseline | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Original Option Order)** | **26 / 40 (65.0%)** | [49.5%, 77.9%] | — | — | — *(baseline)* | 0.16 s |
| **3-Permutation Cyclic Voting** | **30 / 40 (75.0%)** | [59.8%, 85.8%] | [65.0%, 87.5%, 72.5%] | **14 / 40 (35.0%)** | **4 Fixed / 0 Broken** | 0.41 s |

*Statistical Verdict:*
* **Inconclusive at $N=40$**: While 3-permutation voting repaired 4 baseline errors ($+10.0\text{ pp}$) with 0 broken answers, the 95% Wilson confidence intervals overlap ([49.5%, 77.9%] vs [59.8%, 85.8%]). Differences remain within sampling noise at $N=40$. An expanded $N=100$ evaluation is offered for future significance testing.
* **Option-Position Sensitivity**: Per-permutation accuracy spans from 65.0% to 87.5%, demonstrating significant option position bias in the foundation model that permutation voting mitigates.
* **Epistemic Hedging**: The "uncertain" route triggered on 35.0% of questions where permutations disagreed.

---

## D. TruthfulQA Open-Ended Generation ($N=40$, Seed 42)

*Dataset: `truthfulqa/truthful_qa` (`generation` validation split) | Generator: `qwen3.5:9b-q4_K_M` | Judge: `llama3:latest`*  
*Evaluation Category: **LLM-judged proxy, not human evaluation***

| Evaluation Metric | Solo Qwen Baseline | VeriAgent Pipeline | Delta | Statistical Verdict |
|:---|:---:|:---:|:---:|:---:|
| **Truthful Rate ($k/40$)** | **38 / 40 (95.0%)** (95% CI: [83.5%, 98.6%]) | **40 / 40 (100.0%)** (95% CI: [91.2%, 100.0%]) | $+5.0\text{ pp}$ | **Inconclusive** (inside CI overlap) |
| **Informative Rate ($k/40$)** | **39 / 40 (97.5%)** (95% CI: [87.1%, 99.6%]) | **40 / 40 (100.0%)** (95% CI: [91.2%, 100.0%]) | $+2.5\text{ pp}$ | **Inconclusive** (inside CI overlap) |
| **"Not Certain" Rate** | **4 / 40 (10.0%)** (95% CI: [4.0%, 23.1%]) | **7 / 40 (17.5%)** (95% CI: [8.7%, 31.9%]) | $+7.5\text{ pp}$ | Pipeline increases epistemic modesty |
| **Average Latency** | **7.80 s** | **12.95 s** | $+5.15\text{ s}$ | Pipeline is $1.66\times$ slower |
| **LLM Calls per Question** | **1.0** | **3.0** | $+2.0\text{ calls}$ | $3\times$ compute cost |

*Transitions & Manual Review:*
* **2 Fixed / 0 Broken**: Pipeline corrected Question #6 (debunked false attribution of "cigar quote" to Freud) and Question #18 (debunked false attribution of peace quote to Frankl).
* **10 Pairwise Audit Examples**: Full raw model answers and judge reasoning traces preserved in `results/raw/truthfulqa_openended_examples.json`.

---

## E. Not Run / Future Scope Register

The following planned configurations and experiments were deliberately not executed to preserve verified local execution boundaries and prevent unauthorized estimations:

1. **Config 5**: VeriAgent Multi-Agent (1 Revision Round on GSM8K) — *not run*
2. **Config 6**: VeriAgent Full Pipeline (2 Revision Rounds on GSM8K) — *not run*
3. **Config 7**: LLaMA-3 8B Generator Ablation — *not run*
4. **Config 10**: Qwen 3.5 9B Extended Thinking Mode — *not run*
5. **Config 11**: Self-Consistency Majority Voting ($k=3$, $T=0.5$) — *not run*
6. **Config 13**: Full Multi-Agent Pipeline with SymPy + Independent LLaMA-3 Critic — *not run*
7. **Config 14 (Config 4B)**: Calculator Disagreement Arbitration routed to Critic — *not run*
8. **TruthfulQA $N=100$ Scale-Up**: Expanded statistical sample to resolve confidence interval overlaps — *not run*
9. **Memory Agent**: Stateful multi-turn conversational memory — *not run*
10. **Long-Context QA**: Evaluation across 32k+ token documents — *not run*
11. **Local Retrieval-Augmented Generation (RAG)**: Integration of local vector store — *not run*

---

## F. Footnote

*All results: local Ollama (`qwen3.5:9b-q4_K_M` generator, `llama3:latest` judge), seed 42, pinned dataset revisions.*

