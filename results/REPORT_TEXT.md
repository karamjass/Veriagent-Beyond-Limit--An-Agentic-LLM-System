# VeriAgent 2nd Defence — Comprehensive Report Text

This document contains publication-ready, fully verified text sections for the project report, slides, and defence documentation. All numbers are derived directly from verified local Ollama execution logs (seed 42, pinned dataset revisions).

---

## 1. Results

### Quantitative Reasoning on GSM8K ($N=40$, Seed 42)
Evaluation on GSM8K revealed a fundamental methodological finding: the previously reported 72.5% baseline was an artifact of answer-extractor truncation and regex parsing failures on markdown wrappers (`**Final Answer**`, `$`, `\boxed{}`). Correcting the answer extractor established the true single-pass baseline for `qwen3.5:9b-q4_K_M` (Config 1) at **97.5% accuracy** (39/40 correct, 95% Wilson CI: [87.1%, 99.6%]) with an average latency of 4.68s and 1.0 LLM call per question. 

Evaluating calculations exclusively through the Python AST sandbox (`calculator_agent`, Config 2) achieved **92.5% accuracy** (37/40 correct, 95% Wilson CI: [80.1%, 97.4%]) while reducing average latency to **1.22s per question**—nearly four times faster than natural language reasoning. Adding an LLM critic loop (Config 3) yielded **97.5% accuracy** (39/40 correct, 95% Wilson CI: [87.1%, 99.6%]) at 5.12s and 2.0 LLM calls per question, with 0 fixed and 0 broken answers relative to baseline. Targeting agreement between the generator and the AST calculator (Config 4) matched Config 2 at **92.5% accuracy** (37/40 correct) at 3.75s and 2.0 LLM calls per question, breaking 2 baseline questions where the calculator returned 0.0 on word-problem parsing failures. Structured prompting alone (Config 9) and structured prompting paired with SymPy Program-of-Thought (Config 12) both recorded **95.0% accuracy** (38/40 correct, 95% Wilson CI: [83.5%, 98.6%]) at 8.07s (1.0 call) and 9.57s (2.0 calls) respectively, each breaking 1 baseline problem. Across all configurations, differences of 1 to 2 questions fall strictly within noise margins ($2.5\text{ pp}$ per question at $N=40$) and inside the baseline confidence interval.

### False-Premise & Control Hallucination Probe ($N=30$)
On the 30-question diagnostic probe (`data/hallucination_probe.json`), the solo Qwen baseline recorded a **0.0% false-premise hallucination rate** (0/20 fabricated, 95% Wilson CI: [0.0%, 16.1%]) and a **0.0% control over-refusal rate** (0/10 over-refusals, 95% Wilson CI: [0.0%, 27.8%]) with an average latency of 8.60s and 1.0 LLM call per question. The multi-agent pipeline recorded identical point estimates: **0.0% false-premise hallucinations** (0/20) and **0.0% control over-refusals** (0/10), with an average latency of 13.68s ($1.59\times$ slower) and 3.0 LLM calls per question ($3\times$ compute cost). Both deterministic pattern matching and independent `llama3:latest` judging confirmed zero fabrications across all 20 false premises. Because both conditions achieved 0.0% hallucination rates, the comparison demonstrated **no measurable difference** and is statistically **inconclusive**.

### TruthfulQA Multiple-Choice Evaluation ($N=40$, Seed 42)
On the TruthfulQA multiple-choice validation split, single-pass generation using original option order (baseline) achieved **65.0% accuracy** (26/40 correct, 95% Wilson CI: [49.5%, 77.9%]) at 0.16s latency. Implementing 3-permutation cyclic option rotation with majority voting achieved **75.0% accuracy** (30/40 correct, 95% Wilson CI: [59.8%, 85.8%]) at 0.41s latency, yielding 4 fixed and 0 broken answers (+10.0 pp). Per-permutation accuracies varied markedly across rotations: 65.0% (original order), 87.5% (first cyclic shift), and 72.5% (second cyclic shift). The epistemic "uncertain" path was triggered on 35.0% of queries (14/40) where permutations failed to reach consensus. Because the 95% Wilson confidence intervals overlap, the multiple-choice result is **inconclusive at $N=40$**; while the directional correction is **promising, needs larger N** ($N=100$) to demonstrate statistical significance outside noise margins.

### TruthfulQA Open-Ended Generation ($N=40$, Seed 42)
On open-ended generation evaluated by an independent `llama3:latest` proxy judge against official reference targets, the solo Qwen baseline recorded **95.0% truthfulness** (38/40, 95% Wilson CI: [83.5%, 98.6%]), **97.5% informativeness** (39/40, 95% Wilson CI: [87.1%, 99.6%]), and a 10.0% "not certain" epistemic hedging rate (4/40) at 7.80s latency and 1.0 LLM call per question. The VeriAgent multi-agent pipeline recorded **100.0% truthfulness** (40/40, 95% Wilson CI: [91.2%, 100.0%]), **100.0% informativeness** (40/40), and a 17.5% "not certain" rate (7/40) at 12.95s latency ($1.66\times$ slower) and 3.0 LLM calls per question ($3\times$ compute). The pipeline corrected 2 baseline misconceptions (Question #6 regarding Freud's cigar quote and Question #18 regarding Frankl's peace quote) with 0 broken answers (+5.0 pp). However, because the difference of 2 questions lies within the 95% Wilson confidence interval overlap, the outcome is statistically **inconclusive**; the observed improvement is **promising, needs larger N** to confirm beyond sampling variance.

---

## 2. Discussion

The empirical findings highlight crucial architectural tradeoffs between computational efficiency, verification explainability, and raw benchmark accuracy.

First, introducing multi-agent verification loops incurs substantial computational overhead. On mathematical reasoning, critic and Program-of-Thought loops increased inference latency from 4.68s up to 9.57s per question and doubled LLM invocations. On open-ended truthfulness evaluation, the pipeline required 3.0 LLM calls per query and increased latency from 7.80s to 12.95s (+66%). When evaluating foundation models that already operate near benchmark ceiling (such as Qwen 3.5 9B scoring 97.5% on GSM8K), additional verification layers cannot produce statistically significant accuracy gains. Instead, multi-agent frameworks must be justified by their capacity to provide auditable verification traces, inspectable step-by-step logic, and structured compliance.

Second, Program-of-Thought execution represents a distinct efficiency advantage: sandboxed Python AST execution (Config 2) achieved 92.5% accuracy at 1.22s per question—nearly $4\times$ faster than natural language reasoning (4.68s). However, relying uncritically on code execution introduces brittle failure modes: word problems that fail string parsing return fallback zeros, which can overwrite correct textual drafts unless guarded by an arbitration critic.

Third, multiple-choice accuracy swings significantly across cyclic permutations (65.0% to 87.5%), demonstrating that foundation models are vulnerable to option-order bias. Permutation voting acts as an effective epistemic filter, identifying unstable answers and routing 35.0% of split-consensus queries to an explicit uncertainty path.

Finally, methodological transparency requires precise characterization of system capabilities. The internal Fact-Checker utilizes the same model weights as the generator; all logging and user interfaces correctly designate this stage as an internal self-consistency check and not external verification. Similarly, while auditing Qwen outputs with an independent `llama3:latest` model provides cross-architecture parameter diversity, it represents an automated LLM-judged proxy metric rather than human evaluation or external ground truth.

---

## 3. Limitations

1. **Sample Size Constraints ($N=40$ and $N=30$)**: Evaluations were conducted on deterministic test slices (seed 42) of 40 samples for GSM8K and TruthfulQA, and 30 samples for the hallucination probe. At $N=40$, each question represents $2.5\text{ pp}$. While directional improvements were observed (+10.0 pp on TruthfulQA MC, +5.0 pp on open-ended generation), the 95% Wilson score confidence intervals overlap in all cases. Consequently, all comparative findings are strictly reported as inconclusive pending larger-scale evaluations.
2. **Benchmark Ceiling Saturation**: On grade-school arithmetic (GSM8K), the solo generator answered 39 of 40 questions correctly (97.5%). This leaves virtually zero headroom for multi-agent revision loops to demonstrate recovery of incorrect baseline responses.
3. **Absence of External Ground Truth**: All components operate 100% locally via Ollama without cloud connectivity or web search. The internal fact-checker performs an epistemic self-consistency audit, and the cross-model critic provides an independent proxy evaluation; neither provides access to external factual ground truth.
4. **Stateless Conversational Architecture**: The current pipeline evaluates isolated, single-turn prompts. Stateful multi-turn memory agents and long-context document retrieval were not included in the active test suite.

---

## 4. Conclusion

This evaluation provides a transparent, scientifically honest assessment of local multi-agent reasoning:

- **Answer-Extractor Discovery**: An exhaustive audit resolved a key historical anomaly, proving that the previous 72.5% baseline was caused by regex extractor truncation rather than model reasoning failures. The corrected baseline of **97.5%** demonstrates that Qwen 3.5 9B possesses high native arithmetic capability.
- **Verification vs. Benchmark Ceiling**: On grade-school arithmetic, multi-agent verification configurations (92.5% to 97.5%) operate entirely within noise margins of the solo baseline, while doubling inference latency and compute cost. Multi-agent designs should be targeted at tasks with lower baseline accuracy where revision headroom exists.
- **Honest Factuality Assessment**: On the hand-crafted false-premise probe ($N=30$), both baseline and pipeline achieved 0.0% hallucination rates, demonstrating **no measurable difference**. On TruthfulQA multiple-choice ($N=40$) and open-ended generation ($N=40$), multi-agent voting and critique repaired 4 and 2 baseline errors respectively without introducing breaks; however, both outcomes remain statistically **inconclusive** due to confidence interval overlap. The system makes no claim that hallucination was reduced, improved, or eliminated; instead, the directional gains are recognized as **promising, needs larger N**.
- **Structural Integrity**: The system targets rigorous mathematical formatting, inspectable reasoning traces, and honest epistemic hedging, providing transparent operational visibility for local deployment.

---

## 5. Future Scope

1. **Harder Reasoning Benchmarks**: Transition benchmark evaluation from saturated arithmetic to challenging competition datasets—including MATH-500, AIME, and GPQA Diamond—where foundation model baseline accuracy is $\le 60\%$, providing realistic headroom for critic and reviser loops to demonstrate measurable gains.
2. **TruthfulQA Scale-Up ($N=100+$)**: Expand the sample size for TruthfulQA multiple-choice and open-ended generation from $N=40$ to $N=100$ or the full validation split to resolve confidence interval overlaps and test whether observed gains achieve statistical significance.
3. **Dynamic Verification Gating**: Implement confidence-aware routing to bypass verification loops on high-confidence solutions and trigger multi-agent cross-checks only when epistemic entropy is elevated, mitigating latency overhead.
4. **Stateful Memory & Local RAG**: Integrate a stateful conversation agent and local vector-store retrieval to enable factual grounding against curated local documents without relying on external cloud APIs.
