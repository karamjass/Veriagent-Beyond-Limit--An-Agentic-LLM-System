# Latency and Cost Analysis: VeriAgent Benchmark (N=40, Seed 42)

## 1. Per-Config Latency, Resource Breakdown & LLM Invocations

| Config | Model | Accuracy | 95% Wilson CI | Avg Latency | Relative to Cfg 1 | LLM Calls/Q | Multi-Agent Component Breakdown |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1. Qwen 3.5 9B (Solo Generator Baseline) | Qwen 3.5 9B | 39/40 (97.5%) | (87.1%, 99.6%) | 4.68s | 1.00x | 1.0 | 100% Draft Generator (single-pass reasoning, SOLVER_SYS) |
| 2. AST Calculator Only (Code Execution) | Qwen 3.5 9B | 37/40 (92.5%) | (80.1%, 97.4%) | 1.22s | 0.26x | 1.0 | Code generation LLM call (1.2s) + local Python AST executor (<2ms) |
| 3. Qwen 3.5 9B + LLM Critic (No Calculator) | Qwen 3.5 9B | 39/40 (97.5%) | (87.1%, 99.6%) | 5.12s | 1.10x | 2.0 | Draft Generator (~2.8s) + Structured Critic (~2.3s); reviser not triggered |
| 4. Qwen 3.5 9B + AST Calculator Agreement | Qwen 3.5 9B | 37/40 (92.5%) | (80.1%, 97.4%) | 3.75s | 0.80x | 2.0 | Draft Generator (~2.5s) + Calculator agent (~1.2s); numeric agreement check |
| 5. VeriAgent Multi-Agent (1 Revision Round) | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |
| 6. VeriAgent Full Pipeline (2 Revision Rounds) | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |
| 7. LLaMA-3 8B (VeriAgent Pipeline Ablation) | LLaMA-3 8B | not run | not run | not run | not run | not run | Not executed |
| 8. TruthfulQA Validation (Factuality MC Mode) | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |
| 9. Structured Prompt (Upgraded) | Qwen 3.5 9B | 38/40 (95.0%) | (83.5%, 98.6%) | 8.07s | 1.73x | 1.0 | Structured prompt draft generator (detailed 4-part schema) |
| 10. + Thinking Mode | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |
| 11. + Self-Consistency (k=3) | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |
| 12. + SymPy / PoT | Qwen 3.5 9B | 38/40 (95.0%) | (83.5%, 98.6%) | 9.57s | 2.05x | 2.0 | Structured draft generator (~6.8s) + SymPy PoT code execution (~2.8s) |
| 13. Full Pipeline + Independent Critic | Qwen 3.5 9B | not run | not run | not run | not run | not run | Not executed |

## 2. Cost vs. Benefit Analysis

1. **Baseline Speed & Saturation**: Config 1 (Solo Generator) finishes in **4.68s** with **97.5% accuracy** (39/40 correct). On grade-school math (GSM8K), Qwen 3.5 9B possesses exceptionally strong intrinsic reasoning, creating a severe ceiling effect with only 1 failed problem in the sample.
2. **AST Calculator Efficiency**: Config 2 (PoT / AST Calculator alone) executes in **1.22s** (nearly **4x faster** than Config 1, relative latency **0.26x**) while achieving **92.5% accuracy** (37/40). For quantitative arithmetic, generating concise Python/SymPy code evaluated by a sandboxed AST engine is substantially faster and highly accurate.
3. **Multi-Agent Latency Overhead**: Config 3 (Critic) runs in **5.12s** (2 LLM calls, 1.09x Config 1) and Config 4 (AST Agreement) runs in **3.75s** (2 LLM calls, 0.80x Config 1). In both cases, the multi-agent pipeline adds an additional LLM verification step.
4. **Statistical Significance vs. 95% Wilson CI**: Config 1 has a 95% Wilson CI of **[87.12%, 99.56%]**. Config 3 ties Config 1 at 97.5% (difference = 0 questions, 0.0 pp). Config 2 and Config 4 achieve 92.5% (difference = 2 questions, 5.0 pp, CI: [80.14%, 97.42%]). **Every configuration's accuracy falls entirely within the 95% confidence interval of Config 1.** There is no statistically significant gain from multi-agent critique or calculator agreement on GSM8K at N=40.

## 3. Second Table: 95% Wilson Confidence Intervals & Format Compliance

| Config | Setup | Completed Questions | Accuracy | 95% Wilson Confidence Interval | Strict Format Compliance | Avg Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Qwen 3.5 9B (Solo Generator Baseline) | 40/40 | 39/40 (97.5%) | (87.1%, 99.6%) | 0/40 (0.0%) | 4.68s |
| 2 | AST Calculator Only (Code Execution) | 40/40 | 37/40 (92.5%) | (80.1%, 97.4%) | 0/40 (0.0%) | 1.22s |
| 3 | Qwen 3.5 9B + LLM Critic (No Calculator) | 40/40 | 39/40 (97.5%) | (87.1%, 99.6%) | 0/40 (0.0%) | 5.12s |
| 4 | Qwen 3.5 9B + AST Calculator Agreement | 40/40 | 37/40 (92.5%) | (80.1%, 97.4%) | 0/40 (0.0%) | 3.75s |
| 5 | VeriAgent Multi-Agent (1 Revision Round) | 0/40 | not run | not run | not run | not run |
| 6 | VeriAgent Full Pipeline (2 Revision Rounds) | 0/40 | not run | not run | not run | not run |
| 7 | LLaMA-3 8B (VeriAgent Pipeline Ablation) | 0/40 | not run | not run | not run | not run |
| 8 | TruthfulQA Validation (Factuality MC Mode) | 0/40 | not run | not run | not run | not run |
| 9 | Structured Prompt (Upgraded) | 40/40 | 38/40 (95.0%) | (83.5%, 98.6%) | 39/40 (97.5%) | 8.07s |
| 10 | + Thinking Mode | 0/40 | not run | not run | not run | not run |
| 11 | + Self-Consistency (k=3) | 0/40 | not run | not run | not run | not run |
| 12 | + SymPy / PoT | 40/40 | 38/40 (95.0%) | (83.5%, 98.6%) | 26/40 (65.0%) | 9.57s |
| 13 | Full Pipeline + Independent Critic | 0/40 | not run | not run | not run | not run |
