# VeriAgent: Verified Multi-Agent Reasoning Pipeline

VeriAgent is a 100% local, privacy-first multi-agent reasoning and verification framework. It integrates domain-specialized few-shot prompts, a zero-trust AST and SymPy calculation sandbox, self-consistency majority voting, and an independent cross-model critic.

Created and owned by **Karamjass Kaur**, VeriAgent is built upon the **Qwen 3.5 9B** foundation model running locally via Ollama with zero paid external API dependencies.

---

## Key Capabilities

1. **4-Stage Mathematical Formatting**:
   - **Problem understanding**: Goal and given parameters clearly identified.
   - **Step-by-step solution**: Numbered derivations explaining *what* is done, *why*, and the explicit calculation.
   - **Verification**: Sanity check, unit validation, and dimensional analysis.
   - **Final Answer**: Terminating line formatted strictly as `**Final Answer: <numeric_value>**`.

2. **Domain-Specific System Prompts (`prompts.py`)**:
   - Dedicated prompts and non-overlapping few-shot examples for **Math**, **Logical Reasoning**, **Factual/General Knowledge**, **Coding**, and **Identity**.
   - Verified 0% overlap with held-out benchmark splits (GSM8K test split and TruthfulQA validation split).

3. **Zero-Trust Safe AST & SymPy Evaluator**:
   - Strict AST node whitelist preventing arbitrary code execution.
   - Rejects dunder traversal (`__class__`), file system access (`open()`), system calls (`os.system`), and code evaluation (`eval()`, `exec()`).
   - Thread-bounded timeout (2.0s) and bounded power operations ($b^e \le 1000$).
   - Natively evaluates fractions (`Rational`), algebra (`solve`, `Eq`), and linear systems.

4. **Self-Consistency Majority Voting**:
   - Samples $k \in [3, 5]$ candidate solutions at temperature $T=0.5$.
   - Aggregates votes for the final numeric answer.
   - **Always runs calculator check**: Even under unanimous consensus, calculator disagreement marks the response as low-confidence.

5. **Independent Cross-Model Critic & "Do No Harm" Revision**:
   - The critic runs on a separate local model (`llama3:latest` by default via `CRITIC_MODEL`).
   - Emits structured JSON step-by-step audits:
     ```json
     {"valid": false, "first_wrong_step": 2, "reason": "Arithmetic calculation error", "corrected_answer": null}
     ```
   - Targeted revision only updates the flawed step and will never overwrite an answer that already matches the calculator result.

---

## System Requirements

- **Processor**: Multi-core x86_64 CPU (tested on Intel Core Ultra 9 275HX).
- **GPU**: NVIDIA GPU with $\ge 8$ GB VRAM (tested on RTX 5070 Ti 12GB).
- **Memory**: $\ge 16$ GB RAM.
- **Local Model Daemon**: [Ollama](https://ollama.ai) installed and running locally (`http://127.0.0.1:11434`).
- **Required Models**:
  ```bash
  ollama pull qwen3.5:9b-q4_K_M
  ollama pull llama3:latest
  ```

---

## Installation & Setup

```bash
# 1. Clone repository and navigate to folder
cd beyond_limit(veriagent)

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Web Assistant

To launch the local web server:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

---

## Running the Benchmark Evaluation

Run the evaluation harness with per-sample caching and resume capability:

```bash
# Smoke test (40 samples, seed 42)
python run_smoke_test.py

# Full benchmark evaluation on GSM8K
python -c "import model_core; print(model_core.run_benchmark_eval(config_id=13, limit=200, seed=42))"

# Run all unit tests
python -m unittest discover -s tests
```

---

## Configuration Reference

| Config ID | Description | Generator Model | Thinking Mode | Verification Method |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Solo Generator Baseline | Qwen 3.5 9B | Off | None (Single pass) |
| **2** | Calculator Alone | None | N/A | Safe AST / SymPy Interpreter |
| **3** | Generator + Self-Critic | Qwen 3.5 9B | Off | Self-Criticism (Same weights) |
| **4** | Generator + Calculator | Qwen 3.5 9B | Off | Direct AST Agreement |
| **5** | VeriAgent Round 1 | Qwen 3.5 9B | Off | Single Revision Round |
| **6** | VeriAgent Full (Round 2) | Qwen 3.5 9B | Off | Up to 2 Revision Rounds |
| **7** | LLaMA-3 Ablation | LLaMA-3 8B | Off | Full VeriAgent Pipeline |
| **8** | TruthfulQA Validation MC | Qwen 3.5 9B | Off | 3-Permutation Majority Vote |
| **9** | Structured Prompt Only | Qwen 3.5 9B | Off | 4-Stage Prompting |
| **10** | + Thinking Mode | Qwen 3.5 9B | **On** | Extended Scratchpad |
| **11** | + Self-Consistency ($k=3$) | Qwen 3.5 9B | Off | Majority Vote ($T=0.5$) |
| **12** | + SymPy Program-of-Thought | Qwen 3.5 9B | Off | Whitelisted SymPy Engine |
| **13** | Full New Pipeline | Qwen 3.5 9B | Off | Self-Consistency + SymPy + Independent Critic (`llama3`) |

## Hallucination Evaluation

VeriAgent includes infrastructure for three honesty evaluations:

### B1 — TruthfulQA Multiple-Choice (Permutation Voting)
Eliminates position bias by cycling through 3 option permutations and taking a majority vote.
- Run: `model_core.evaluate_truthfulqa_permutation(examples, limit=40, seed=42)`
- Label: *"Generation-based; same local model weights for all permutations — not external verification."*

### B2 — Open-Ended Generation with LLM Judge
Generates free-text answers with Qwen and judges truthfulness with **llama3:latest** (a different model family).
- Run: `model_core.evaluate_truthfulqa_openended(examples, judge_model="llama3:latest")`
- **Important**: The judge model must differ from the generator. `ValueError` is raised otherwise.
- Label: *"LLM-judged proxy, not human evaluation. Neither model is external ground truth."*

### B3 — False-Premise / Unanswerable Probe
Measures how often the model invents facts on 20 false-premise questions vs. correctly answers 10 real controls.
- Probe set: `data/hallucination_probe.json` (hand-written, not from TruthfulQA)
- Run: `model_core.run_hallucination_probe("data/hallucination_probe.json")`
- Results saved to: `results/raw/hallucination_probe_results.json`

### Results
See `results/hallucination_summary.md` for the current status table and verdicts.

> **Note**: The internal fact-checker and critic share the same model weights as the generator.
> All pipeline provenance labels read *"Self-consistency check (not external verification)"*
> to prevent misleading claims of external factual grounding.

---
