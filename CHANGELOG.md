# Changelog

All notable changes and architectural improvements in this release of VeriAgent.

## [2.5.0] - 2026-11-09

### Security (basic controls, not penetration-tested)
* **Access Control & Tunnel Protection (H2 Fix)**:
  - In demo/public mode (`VERIAGENT_ENV=demo` or `VERIAGENT_ENV=public`), disabled automatic `remote_addr` localhost logins completely.
  - Required `DEMO_ACCESS_CODE` environment variable with constant-time verification (`hmac.compare_digest`) before issuing session cookies.
  - Added support for `TRUSTED_PROXY=1`; `X-Forwarded-For` is strictly ignored when unset to prevent client IP spoofing.
  - Added secret key validation refusing to boot with default placeholder keys in demo and production environments.
* **Abuse Protection & Rate Limiting (H1 & H3 Fixes)**:
  - Integrated `Flask-Limiter` with client IP and authenticated user keying (10 req/min for chat, 2 req/min for benchmark runs), returning clear HTTP 429 status codes. In-memory single-worker storage (`memory://`).
  - Added active generation concurrency cap (maximum 2 concurrent reasoning tasks), returning HTTP 503 (`"Server is busy processing other reasoning requests"`) to prevent VRAM exhaustion.
  - Capped maximum prompt length to 2,000 characters (`MAX_PROMPT_LENGTH`) and request body to 1 MB (`MAX_CONTENT_LENGTH`).
* **Emergency Administration**:
  - Implemented `ADMIN_KILL_SWITCH=1` returning immediate HTTP 503 on all generation endpoints.
* **Network Isolation & Asset Protection (M3 & M5 Fixes)**:
  - Restricted static file serving strictly to an allowlist of root assets (`styles.css`, `app.js`, `favicon.ico`, `robots.txt`) and `static/` directory with safe file extensions. Requests to `results/`, `data/`, source code (`*.py`), and `.env` files are blocked with HTTP 403 Forbidden.
  - Added startup safety check warning loudly if `OLLAMA_BASE_URL` is bound to a non-loopback address (`0.0.0.0`).
* **Browser Protections & Security Headers (M4 Fix)**:
  - Configured `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and strict `Cache-Control: no-store` on API routes.
  - Added Content-Security-Policy (CSP) restricting frame embedding and cross-origin access. (Known limitation: `'unsafe-inline'` documented in `results/security_audit.md` for inline UI compatibility).
* **SafeMathEvaluator AST Defenses**:
  - Hardened AST execution against attribute traversal (`__class__`), string formatting, excessive exponents ($> 1000$), AST bombs ($> 150$ nodes), and oversized expressions ($> 2000$ chars).
* **Automated Security Test Suite**:
  - Added 21 automated security test cases in `tests/test_security.py` covering auth rejection, tunnel simulation, rate limits, kill switch, security headers, and AST injection defenses. Full suite: 65 tests, 100% passing.
* **Security Documentation**:
  - Documented audit findings in `results/security_audit.md`, automated scans in `results/security_scan.md`, and prompt injection attacks in `results/security_injection_tests.md`.

## [2.4.0] - 2026-11-08

### Hallucination Probe Audit & Multi-Judge Infrastructure
* **Full Answer Text Preservation**: Re-ran the 30-question hallucination probe (`data/hallucination_probe.json`) with unlimited answer retention, resolving previous 400-character truncation across 13 samples.
* **Dual-Label Classification (Rule-Based + LLaMA-3 Judge)**: Evaluated all 30 probe outputs under both pattern-matching rules and an independent `llama3:latest` LLM judge without overrides. Disagreements are cataloged explicitly:
  - Both evaluators confirmed **0/20 hallucinations** on false-premise questions (0.0%).
  - Both evaluators confirmed **10/10 correct answers** on factual control questions (100.0%, 0.0% over-refusal).
  - Evaluator divergence occurred exclusively on semantic labeling: the rule scanner marked premise refutations as `REFUSED` while the LLaMA-3 judge categorized them as `OK` (correctly identifying false premises without fabrication).
* **Baseline vs. VeriAgent Pipeline Comparison**: Executed sequential benchmark runs (`results/raw/probe_baseline_N30.jsonl` vs. `results/raw/probe_pipeline_N30.jsonl`) measuring hallucination rates, control over-refusals, per-stage latencies, and LLM call counts.

### Table Wording & Metric Standardization
* **Format Compliance Labeling**:
  - Replaced misleading `0.0%` compliance values with `n/a` for Configurations 1 to 4 (`SOLVER_SYS` plain-prompt configs that do not target the 4-part structured layout).
  - Marked Configurations 9 and 12 format compliance as `not re-measured` to reflect full-text validation requirements while preserving verified accuracy caches.
* **Terminology Alignment**:
  - Replaced occurrences of `enforces` and `guarantees` with `targets` in `slides_content.md` to maintain rigorous scientific phrasing.
  - Eliminated speculative wording regarding noise margins in presentation slides.

### Automated Test Suite
* Added 3 new regression tests in `tests/test_veriagent.py` (total: 44 tests, 43 passed, 1 skipped for optional dependencies):
  - `test_judge_model_never_equals_generator`: Enforces strict model divergence between judge and generator.
  - `test_probe_raw_file_has_30_rows`: Validates integrity and required fields across all 30 probe outputs in `results/hallucination_probe_raw.jsonl`.
  - `test_compliance_shows_na_for_non_structured_configs`: Asserts `n/a` format compliance in `results/final_results_table.csv` for non-structured configs.

## [2.3.0] - 2026-11-08

### Part A — GSM8K Verification & Config 4B

**A1 — llm_calls verification:**
- Confirmed Config 2: 1 LLM call/question (calculator only), Config 3: 2 LLM calls/question (draft + critic), Config 4: 2 LLM calls/question (draft + calculator). All 40 rows verified. All gates passed.

**A2 — Config 4 root-cause analysis:**
- Investigation revealed both broken questions (indices 8, 28) share the same failure mode: `calculator_agent()` returned `0.0` on word-problem parsing failure, and the agreement rule silently overrode a correct LLM draft (145→0, 18→0).
- Finding documented in `results/config4_analysis.md`.
- **Added Config 4B** (`config_id=14`): on calculator–draft disagreement, the critic arbitrates instead of blindly trusting the calculator. Added `c_val == 0.0` guard to avoid overriding correct drafts with parse-failure zeros. Requires ≥2 LLM calls, enforced by harness.

**Code changes to `model_core.py`:**
- Added `DEFAULT_CRITIC_MODEL` module-level constant (env-overridable via `CRITIC_MODEL`).
- Added Config 14 (`config_id == 14`) dispatch in `run_benchmark_eval()`.
- Added Config 14 llm_calls ≥ 2 enforcement.

### Part B — Hallucination Evaluation Infrastructure

**B1 — TruthfulQA Permutation Voting:**
- Added `evaluate_truthfulqa_permutation()` to `model_core.py` with 3-permutation cyclic option rotation, majority vote, and "uncertain" path.
- Caches to `results/raw/truthfulqa_perm_N{limit}_seed{seed}.jsonl`.

**B2 — Open-Ended TruthfulQA with LLM Judge:**
- Added `evaluate_truthfulqa_openended()` with `judge_model` parameter.
- Enforces `judge_model != MODEL` (raises `ValueError` if same) — tested by `test_judge_uses_different_model`.
- Labels all output: *"LLM-judged proxy, not human evaluation"*.
- Saves 10 raw examples to `results/raw/truthfulqa_openended_examples.json`.

**B3 — False-Premise Probe:**
- Created `data/hallucination_probe.json`: 20 hand-written false-premise questions (fake papers, fake people, impossible events) + 10 real control questions.
- Added `run_hallucination_probe()` to `model_core.py`.

**B4 — Cross-Model Fact-Check:**
- Config 15 (`config_id==15`) reserved for Qwen-generates / llama3-verifies pipeline.
- Fact-Checker label audit complete: no remaining "Internal fact-checker pass" strings in codebase (all replaced with "Self-consistency check (not external verification)").

**B5 — Local RAG gate:**
- Skips if `ENABLE_LOCAL_RAG != 1` or `data/local_knowledge.txt` missing. Writes "not run" to summary.

### Part C — Output Files

- Regenerated `results/final_results_table.md/.csv/.html` from verified 40-row caches.
- Written `results/hallucination_summary.md` with per-task verdicts (all "inconclusive" — not yet run).
- Updated `slides_content.md`: removed "guarantees" (replaced with "enforces"), added hallucination status section.

### Tests

- Added 7 new tests in `TestNewBenchmarkFeatures` class: `test_cfg3_llm_calls_ge_2`, `test_cfg4_llm_calls_ge_2`, `test_judge_uses_different_model`, `test_probe_set_loader`, `test_no_guarantees_in_slides`, `test_truthfulqa_perm_function_exists`, `test_default_critic_model_defined`.
- Added `skipTest` guard to `test_few_shot_no_overlap_with_benchmark_data` for environments without `datasets` package.
- **Total: 41 tests — 40 pass, 1 skipped (datasets not installed in system Python).**


## [2.2.0] - 2026-11-08

### Benchmark Harness Corrections & LLM Call Instrumentation
* **Benchmark Invalidation Notice**: Earlier evaluation results for **Config 2** (AST Calculator Only), **Config 3** (LLM Critic), **Config 4** (AST Calculator Agreement), and **Config 5** (VeriAgent 1 Revision Round) were invalidated due to harness dispatch bugs in `build_results_table.py`:
  - **Config 2**: Attempted to call `model_core.extract_expression()` and `model_core.safe_eval()`, neither of which existed. The unhandled `AttributeError` was caught silently, resulting in an artificial `0.0%` accuracy and `0.0s` latency without executing the AST calculator.
  - **Config 3**: Dispatched to `solve_question()` which was configured for single-pass draft generation; the critic and reviser agents were never executed.
  - **Config 4**: Dispatched an initial draft and then crashed on `extract_expression()`, silently falling back to the raw draft without executing `calculator_agent()`.
  - **Config 5**: Encountered an unexpected process crash at question index 25 (25/40 complete) and never finished.
* **Resolution**: All earlier cached rows for Configs 2, 3, 4, and 5 have been formally invalidated and removed. Config 1, Config 9, and Config 12 remain verified at 40/40.

### Fixes & Improvements
1. **Per-Question LLM Call Instrumentation**:
   - Added global thread-safe LLM call counter (`reset_llm_call_counter()`, `get_llm_call_count()`) integrated into `chat()`.
   - Every benchmark cache row now explicitly records `"llm_calls": int` alongside `"backend": "ollama"` and `"model": MODEL`.
   - Verified execution constraints: Config 3 enforces $\ge 2$ calls (draft + critic), and Config 4 enforces $\ge 2$ calls (draft + calculator).
2. **Canonical Dispatch in `run_benchmark_eval`**:
   - Config 2 correctly executes `calculator_agent(question)` via Python `ast` / `SafeMathEvaluator`.
   - Config 3 executes `draft_agent(question)` $\to$ `critic_agent(question, draft)` $\to$ `reviser_agent()` (if flagged).
   - Config 4 executes `draft_agent(question)` + `calculator_agent(question)` with numeric agreement check.
3. **Format & Text Integrity**:
   - Cache saving preserves full text up to 20,000 characters (eliminating the legacy 1500 character slice).
   - Implemented `ensure_single_final_answer()` to guarantee exactly one `**Final Answer: <val>**` appears as the very last line.
   - Verified cache deduplication by question index on load.
   - Asserted `backend == "ollama"` on all benchmark rows.

## [2.1.0] - 2026-11-06

### 1. Robust Answer Extraction & Formatting
- **Standardized Math Format**: Enforced a strict 4-section layout for quantitative prompts: (1) Problem understanding, (2) Step-by-step solution with calculation rationale, (3) Sanity-check verification, and (4) Terminating `**Final Answer: <numeric_value>**`.
- **Enhanced Answer Extractor**: Upgraded `extract_answer()` in `model_core.py` to robustly parse markdown wrappers (`**Final Answer: X**`, `**Final Answer:** $X`), boxed numbers (`\boxed{X}`), currency symbols, fractions (`3/4` -> `0.75`), and negative numbers.
- **Answer Retry Mechanism**: Added an automated single-retry prompt if a candidate generation fails to output an extractable `Final Answer` line.

### 2. Domain Router & Dedicated Prompts (`prompts.py`)
- **Independent Prompt Module**: Extracted domain prompts into `prompts.py` covering Math, Logic, Factual, Coding, and Identity.
- **Out-of-Distribution Few-Shot Examples**: Hand-crafted 2 worked examples per domain that are programmatically verified to have 0% overlap with held-out GSM8K test and TruthfulQA validation splits.
- **Safe Domain Router**: Implemented `classify_query()` with prioritized keyword and pattern matching, ensuring math, logic, and coding queries are never misrouted to identity prompts.

### 3. Zero-Trust Safe AST & SymPy Evaluator
- **SafeMathEvaluator**: Replaced basic calculator with a whitelisted AST parser and SymPy executor supporting algebra (`solve`, `Eq`), linear systems, and fractional arithmetic (`Rational`).
- **Sandboxed Security**: Strictly disallows `ast.Attribute` (blocking dunder traversal), `ast.Import`, `ast.ImportFrom`, `ast.While`, and code execution builtins (`eval`, `exec`, `open`).
- **Bounded Resource Limits**: Enforced 2.0s thread timeout and bounded exponents ($b^e \le 1000$). Verified against 8 malicious injection test cases.

### 4. Verification Upgrades
- **Self-Consistency Majority Voting**: Added `solve_self_consistency()` sampling $k \in [3, 5]$ candidate solutions at $T=0.5$ and voting on the final numeric answer.
- **Unanimous Calculator Cross-Check**: Enforced mandatory calculator evaluation even when candidate votes are unanimous; if the calculator disagrees, the response is explicitly marked as low-confidence.
- **Independent Cross-Model Critic**: Configured `DEFAULT_CRITIC_MODEL = llama3:latest` (configurable via `CRITIC_MODEL`) to audit candidate reasoning steps and return structured JSON reviews.
- **Do No Harm Revision**: Targeted revisions address only the identified erroneous step and will never overwrite a solution that already matches the calculator result.
- **Honest Provenance Labeling**: Renamed `"Internal fact-checker pass"` to `"Self-consistency check (not external verification)"` to prevent misleading claims of external factuality.

### 5. TruthfulQA Bias Mitigation
- **Permutation Shuffling & Voting**: Implemented 3-permutation cyclic option rotation for multiple-choice questions to eliminate position bias.
- **Epistemic Hedging**: Added uncertainty paths for split votes and unverified factual claims.

### 6. Resumable Evaluation Harness
- **Per-Sample JSONL Caching**: Raw predictions are cached per sample in `results/raw/`, allowing instant resume upon interruption.
- **Comprehensive Metrics**: Added 95% Wilson score confidence intervals, format compliance percentage tracking, average latency logging, and Fixed/Broken transition analysis vs baseline.

