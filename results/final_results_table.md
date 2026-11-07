# VeriAgent Results — GSM8K N=40 seed=42

| # | Config | Model | Accuracy | Fixed / Broken | Recall | Precision | Avg Latency | LLM calls/q |
|---|--------|-------|----------|----------------|--------|-----------|-------------|-------------|
| 1 | Solo Generator Baseline | Qwen 3.5 9B | 39/40 (97.5%) | — (baseline) | 97.5% | 97.5% | 4.68s | 1 |
| 2 | AST Calculator Only (Code Execution) | Qwen 3.5 9B | 37/40 (92.5%) | 0 Fixed / 2 Broken | 92.5% | 92.5% | 1.22s | 1 |
| 3 | Qwen 3.5 9B + LLM Critic (No Calculator) | Qwen 3.5 9B | 39/40 (97.5%) | 0 Fixed / 0 Broken | 97.5% | 97.5% | 5.12s | 2 |
| 4 | Qwen 3.5 9B + AST Calculator Agreement | Qwen 3.5 9B | 37/40 (92.5%) | 0 Fixed / 2 Broken | 92.5% | 92.5% | 3.75s | 2 |
| 5 | VeriAgent Multi-Agent (1 Revision Round) | — | *not run* | — | — | — | — | — |
| 6 | VeriAgent Full Pipeline (2 Revision Rounds) | — | *not run* | — | — | — | — | — |
| 7 | LLaMA-3 8B (VeriAgent Pipeline Ablation) | — | *not run* | — | — | — | — | — |
| 8 | TruthfulQA Validation (Factuality MC Mode) | — | *not run* | — | — | — | — | — |
| 9 | Structured Prompt (Qwen 3.5 9B) | Qwen 3.5 9B | 38/40 (95.0%) | 0 Fixed / 1 Broken | 95.0% | 95.0% | 8.07s | 1 |
| 10 | + Thinking Mode | — | *not run* | — | — | — | — | — |
| 11 | + Self-Consistency (k=3) | — | *not run* | — | — | — | — | — |
| 12 | + SymPy / PoT | Qwen 3.5 9B | 38/40 (95.0%) | 0 Fixed / 1 Broken | 95.0% | 95.0% | 9.57s | 2 |
| 13 | Full Pipeline + Independent Critic | — | *not run* | — | — | — | — | — |
| 14 | Config 4B — Disagreement → Critic | — | *not run* | — | — | — | — | — |

*All results: local Ollama, N=40, seed=42, GSM8K revision 740312add88f781978c0658806c59bc2815b9866. Local Ollama (qwen3.5:9b-q4_K_M). Fixed = baseline wrong, now correct. Broken = baseline correct, now wrong. Precision = Recall = Accuracy (every question answered, TN = 0). Differences of 1–2 questions are within noise.*

---

## Extended Metrics — 95% Wilson CI · Format Compliance

| # | Accuracy | 95% Wilson CI | Format Compliance | Avg Latency | LLM calls/q |
|---|----------|---------------|-------------------|-------------|-------------|
| 1 (Solo Generator Baseline) | 39/40 (97.5%) | (87.1%, 99.6%) | n/a | 4.68s | 1 |
| 2 (AST Calculator Only (Code Execution)) | 37/40 (92.5%) | (80.1%, 97.4%) | n/a | 1.22s | 1 |
| 3 (Qwen 3.5 9B + LLM Critic (No Calculator)) | 39/40 (97.5%) | (87.1%, 99.6%) | n/a | 5.12s | 2 |
| 4 (Qwen 3.5 9B + AST Calculator Agreement) | 37/40 (92.5%) | (80.1%, 97.4%) | n/a | 3.75s | 2 |
| 5 (VeriAgent Multi-Agent (1 Revision Round)) | — | — | — | — | — |
| 6 (VeriAgent Full Pipeline (2 Revision Rounds)) | — | — | — | — | — |
| 7 (LLaMA-3 8B (VeriAgent Pipeline Ablation)) | — | — | — | — | — |
| 8 (TruthfulQA Validation (Factuality MC Mode)) | — | — | — | — | — |
| 9 (Structured Prompt (Qwen 3.5 9B)) | 38/40 (95.0%) | (83.5%, 98.6%) | not re-measured | 8.07s | 1 |
| 10 (+ Thinking Mode) | — | — | — | — | — |
| 11 (+ Self-Consistency (k=3)) | — | — | — | — | — |
| 12 (+ SymPy / PoT) | 38/40 (95.0%) | (83.5%, 98.6%) | not re-measured | 9.57s | 2 |
| 13 (Full Pipeline + Independent Critic) | — | — | — | — | — |
| 14 (Config 4B — Disagreement → Critic) | — | — | — | — | — |

**Noise note (N=40, 1 question = 2.5 pp):** The 95% Wilson CI for Config 1 (97.5%) is [87.1%, 99.6%]. All other configs fall within this interval. No statistically significant improvement over the solo baseline is claimed. Configs 5, 6, 7, 8, 10, 11, 13, 14 require additional run time and are marked "not run".