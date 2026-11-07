# VeriAgent 2nd Defence - Slide-Ready Content

## 1. Conclusion & Future Scope

### Conclusion
- **Answer Extractor Root-Cause Discovery**: Rigorous audit revealed the earlier reported 72.5% baseline was an artifact of extractor truncation and regex failures on markdown wrappers (`**Final Answer**`, `$`, `\boxed{}`). Fixing the extractor demonstrated the true Qwen 3.5 9B solo baseline is **97.5%** (39/40 correct), replacing the legacy 72.5% -> 85.0% multi-agent improvement claim.
- **Benchmark Ceiling Saturation**: On GSM8K ($N=40$), the foundation model's intrinsic mathematical reasoning is already near ceiling (97.5%), meaning multi-agent critique and calculator agreement (92.5% - 97.5%) operate within noise bounds rather than providing large percentage-point leaps.
- **Program-of-Thought (PoT) Efficiency**: Evaluating calculations via sandboxed Python AST execution (`calculator_agent`, Config 2) achieves **92.5% accuracy at 1.22s/question** — nearly **4x faster** than natural language chain-of-thought (4.68s).
- **Honest Metric Transparency**: Multi-agent verification targets structural adherence and produces auditable verification traces, but adds measurable latency overhead (up to 2x) without significant accuracy gains on elementary arithmetic benchmarks.

### Future Scope
- **Evaluation on Harder Reasoning Benchmarks**: Transition benchmark evaluation to collegiate and Olympiad datasets (MATH-500, AIME, GPQA) where foundation generator accuracy is $\le 60\%$, providing realistic headroom for critic and reviser loops.
- **Cross-Model Critic Diversity**: Deploy distinct parameter families (e.g. `llama3:latest` auditing `qwen3.5:9b`) to prevent correlated blind spots and self-critique bias.
- **Dynamic Verification Gating**: Implement uncertainty-aware routing to bypass verification loops on high-confidence questions and invoke multi-agent cross-checks only when epistemic entropy is elevated.
- **Stateful Memory & External RAG**: Implement conversational long-context memory and external knowledge-base retrieval to expand beyond isolated closed-book queries.

## 2. Honest Observation Box (For Results Slide)

> ### Critical Observations & Findings

> * **Differences Within Noise at N=40**: At $N=40$, each question represents $2.5\text{ pp}$. The difference between Config 1 (97.5%, 39/40) and Config 2 / Config 4 (92.5%, 37/40) is exactly 2 questions ($5.0\text{ pp}$), which falls strictly inside the 95% Wilson confidence interval ($80.1\% - 97.4\%$). No configuration demonstrates a statistically significant accuracy difference over baseline.
> * **GSM8K Ceiling Effect**: With 39 of 40 questions answered correctly by the solo generator, GSM8K is saturated for Qwen 3.5 9B. Multi-agent revisions have virtually zero incorrect baseline solutions available to 'fix', making benchmark progression asymptotic.
> * **Latency vs. Verification Tradeoff**: Adding structured critic loops doubles inference time ($4.68\text{s} \to 5.12\text{s} - 9.57\text{s}$) and requires 2 LLM calls per question. Multi-agent architectures must be justified by verification explainability and safety rather than raw benchmark accuracy gains on simple math.

## 3. Dashboard Screenshot Captions

- **Caption 1 (Step-by-Step Reasoning)**: *'VeriAgent multi-agent reasoning bubble displaying clean step-by-step problem understanding, calculation flow, and internal sanity checks with reasoning traces stripped from user view.'*
- **Caption 2 (Single Final Answer Formatting)**: *'Deterministic output formatting ensuring exactly one bold `**Final Answer: <val>**` terminates the response without redundant repetitions.'*
- **Caption 3 (Honest Provenance Badge)**: *'Sub-bubble status indicator displaying `"Self-consistency check (not external verification)"` to honestly reflect shared model weights without misrepresenting external truth access.'*

## 4. Limitations Summary

- **Evaluation Sample ($N=40$)**: Evaluation uses a deterministic 40-sample test slice (seed 42) rather than the complete 1,319 GSM8K test set, introducing sampling variance.
- **GSM8K Benchmark Saturation**: Foundation model capability on grade-school math leaves negligible room for multi-agent revision to demonstrate recovery.
- **TruthfulQA Status**: Factuality evaluation planned; internal fact-checker utilizes identical model weights, serving as an epistemic self-consistency check rather than external ground-truth validation.
- **Memory Agent Status**: Long-context memory and conversational retrieval are not implemented in the current stateless pipeline.

## 5. Hallucination Evaluation Status

### B1 — TruthfulQA MC (Permutation Voting)
- Infrastructure complete: `evaluate_truthfulqa_permutation()` implemented in `model_core.py`
- 3-permutation cyclic option rotation with majority voting and "uncertain" path
- **Status: planned, not completed** — requires Ollama active during evaluation run
- Old 57.5% comparison point: not directly comparable (different seed/sample unknown)

### B2 — Open-Ended Generation with LLM Judge
- Infrastructure complete: `evaluate_truthfulqa_openended()` implemented
- **Judge: llama3:latest** (different model from Qwen generator — required by design)
- Label: *"LLM-judged proxy, not human evaluation. Neither model is external ground truth."*
- **Status: planned, not completed**

### B3 — False-Premise / Unanswerable Probe
- Probe set: `data/hallucination_probe.json` (20 false-premise + 10 control questions)
- **Status: planned, not completed** — `run_hallucination_probe()` ready to execute

### B4 — Cross-Model Fact-Check (Config 15)
- Qwen generates → llama3 verifies claims
- Label: *"Neither self-consistency nor cross-model check is external ground truth"*
- **Status: not run** — requires llama3:latest available

### Fact-Checker Labelling (Complete)
- All pipeline outputs now carry: `"Self-consistency check (not external verification)"`
- Removed any residual "Internal fact-checker pass" labels (confirmed clean by grep)

---
