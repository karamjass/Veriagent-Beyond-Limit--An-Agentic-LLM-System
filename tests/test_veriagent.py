import json
import math
import os
import re
import unittest
import model_core
import prompts


class TestAnswerExtraction(unittest.TestCase):
    def test_standard_final_answer(self):
        text = "The calculated result is:\nFinal answer: 42"
        self.assertEqual(model_core.extract_answer(text), 42.0)

    def test_bold_final_answer(self):
        text = "Step 1: 10 + 5 = 15\n**Final Answer: 15**"
        self.assertEqual(model_core.extract_answer(text), 15.0)

    def test_bold_final_answer_with_colon_inside(self):
        text = "Step 1: 20 * 2 = 40\n**Final Answer:** $40.50"
        self.assertEqual(model_core.extract_answer(text), 40.5)

    def test_negative_and_decimal(self):
        text = "Net change is negative.\n**Final Answer: -12.75**"
        self.assertEqual(model_core.extract_answer(text), -12.75)

    def test_gsm8k_delimiter(self):
        text = "Janet had 16 eggs.\n#### 16"
        self.assertEqual(model_core.extract_answer(text), 16.0)

    def test_gsm8k_currency_delimiter(self):
        text = "Total cost is $85.\n#### $85"
        self.assertEqual(model_core.extract_answer(text), 85.0)

    def test_boxed_final_answer(self):
        text = "The solution is complete.\nFinal Answer: \\boxed{100}"
        self.assertEqual(model_core.extract_answer(text), 100.0)

    def test_fraction_final_answer(self):
        text = "Probability calculation.\n**Final Answer: 3/4**"
        self.assertAlmostEqual(model_core.extract_answer(text), 0.75)

    def test_comma_formatted_number(self):
        text = "Population after 5 years.\n**Final Answer: 1,250,500**"
        self.assertEqual(model_core.extract_answer(text), 1250500.0)

    def test_steps_do_not_override_final_answer(self):
        text = (
            "Step 1: Add 50 and 20 to get 70.\n"
            "Step 2: Multiply 70 by 3 to get 210.\n"
            "Step 3: Subtract 10 to get 200.\n"
            "**Final Answer: 200**"
        )
        self.assertEqual(model_core.extract_answer(text), 200.0)

    def test_empty_or_no_number(self):
        self.assertIsNone(model_core.extract_answer(""))
        self.assertIsNone(model_core.extract_answer("There are no digits here."))


class TestDomainRouterAndPrompts(unittest.TestCase):
    def test_math_queries_not_routed_to_identity(self):
        math_queries = [
            "Who has more apples if John has 5 and Mary has 8?",
            "Calculate who created more profit: Company A ($500) or Company B ($400)?",
            "Solve 2x + 5 = 15 for x.",
            "Janet has 16 eggs. She eats 3. How many are left?",
            "What is the integral of x^2 from 0 to 3?",
        ]
        for q in math_queries:
            domain = model_core.classify_query(q)
            self.assertEqual(domain, "math", f"Query '{q}' was routed to '{domain}', expected 'math'")
            self.assertFalse(model_core.is_identity_question(q), f"Query '{q}' mistakenly flagged as identity")

    def test_coding_queries_not_routed_to_identity(self):
        coding_queries = [
            "Write a Python function to sort a list of strings.",
            "Implement a binary search tree in Python.",
            "Write code to reverse a linked list.",
            "Debug this function: def foo(x): return x + 1",
        ]
        for q in coding_queries:
            domain = model_core.classify_query(q)
            self.assertEqual(domain, "coding", f"Query '{q}' was routed to '{domain}', expected 'coding'")
            self.assertFalse(model_core.is_identity_question(q), f"Query '{q}' mistakenly flagged as identity")

    def test_logic_queries_not_routed_to_identity(self):
        logic_queries = [
            "All humans are mortal. Socrates is human. Deduce the conclusion using Modus Ponens.",
            "If it rains, the grass is wet. The grass is not wet. Is it raining?",
            "Consider two premises P1 and P2: can we find a counterexample?",
        ]
        for q in logic_queries:
            domain = model_core.classify_query(q)
            self.assertEqual(domain, "logic", f"Query '{q}' was routed to '{domain}', expected 'logic'")
            self.assertFalse(model_core.is_identity_question(q), f"Query '{q}' mistakenly flagged as identity")

    def test_identity_queries_routed_correctly(self):
        identity_queries = [
            "Who is the owner of VeriAgent?",
            "Who created you and what model do you use?",
            "Who built this application?",
            "Who is Karamjass Kaur?",
        ]
        for q in identity_queries:
            domain = model_core.classify_query(q)
            self.assertEqual(domain, "identity", f"Query '{q}' was routed to '{domain}', expected 'identity'")
            self.assertTrue(model_core.is_identity_question(q), f"Query '{q}' should be flagged as identity")

    def test_few_shot_no_overlap_with_benchmark_data(self):
        """Programmatic check that few-shot questions don't overlap with GSM8K test or TruthfulQA."""
        try:
            from datasets import load_dataset
        except ImportError:
            self.skipTest("datasets package not installed in test environment; skipping overlap check")

        few_shot_questions = re.findall(r"Question:\s*(.+?)(?=\nAnswer:)", prompts.MATH_SOLVER_SYS, flags=re.S)
        few_shot_questions += re.findall(r"Question:\s*(.+?)(?=\nAnswer:)", prompts.LOGIC_SOLVER_SYS, flags=re.S)
        few_shot_questions += re.findall(r"Question:\s*(.+?)(?=\nAnswer:)", prompts.FACTUAL_SOLVER_SYS, flags=re.S)
        few_shot_questions += re.findall(r"Question:\s*(.+?)(?=\nAnswer:)", prompts.CODING_SOLVER_SYS, flags=re.S)

        self.assertGreater(len(few_shot_questions), 0, "No few-shot questions extracted from prompts.py")

        # Load held-out datasets
        gsm8k = load_dataset("openai/gsm8k", "main", split="test", revision=model_core.GSM8K_DATASET_REVISION)
        truthfulqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision=model_core.TRUTHFULQA_DATASET_REVISION)

        gsm8k_texts = {ex["question"].strip().lower() for ex in gsm8k}
        truthfulqa_texts = {ex["question"].strip().lower() for ex in truthfulqa}

        for q in few_shot_questions:
            clean_q = q.strip().lower()
            self.assertNotIn(clean_q, gsm8k_texts, f"Few-shot question overlaps with GSM8K test: {q}")
            self.assertNotIn(clean_q, truthfulqa_texts, f"Few-shot question overlaps with TruthfulQA validation: {q}")


if __name__ == "__main__":
    unittest.main()


class TestSafeMathEvaluator(unittest.TestCase):
    def test_basic_arithmetic(self):
        code = "a = 14 * 8\nb = 12 * 6\nc = 40 - 9\nans = a + b + c"
        self.assertEqual(model_core.safe_math_eval(code), 215.0)

    def test_fractions_rational(self):
        code = "ans = Rational(3, 4) + Rational(1, 4)"
        self.assertEqual(model_core.safe_math_eval(code), 1.0)

    def test_algebraic_equation_solver(self):
        code = "x = Symbol('x')\neq = Eq(3*x - 9, 21)\nsols = solve(eq, x)\nans = sols[0]"
        self.assertEqual(model_core.safe_math_eval(code), 10.0)

    def test_linear_system_solver(self):
        code = "x, y = symbols('x y')\neq1 = Eq(x + y, 10)\neq2 = Eq(x - y, 4)\nsols = solve((eq1, eq2), (x, y))\nans = sols[x]"
        self.assertEqual(model_core.safe_math_eval(code), 7.0)

    def test_built_in_math_funcs(self):
        code = "ans = sqrt(144) + gcd(24, 36)"
        self.assertEqual(model_core.safe_math_eval(code), 24.0)

    # ── Malicious / Hostile Input Tests ──────────────────────────────────────
    def test_malicious_os_import_blocked(self):
        code = "__import__('os').system('echo pwned')"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_import_statement_blocked(self):
        code = "import os\nos.system('echo pwned')"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_open_file_blocked(self):
        code = "open('requirements.txt').read()"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_dunder_attribute_traversal_blocked(self):
        code = "x = ().__class__.__base__.__subclasses__()"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_eval_call_blocked(self):
        code = "eval('1 + 1')"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_exec_call_blocked(self):
        code = "exec('x = 5')"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_while_loop_blocked(self):
        code = "while True: pass"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_malicious_exponential_bomb_blocked(self):
        code = "ans = 2 ** 100000000"
        self.assertIsNone(model_core.safe_math_eval(code))

    def test_no_truncation_in_saved_text(self):
        """Verify saved benchmark rows preserve >1500 chars without [:1500] slice."""
        long_text = "A" * 2500
        test_path = "results/raw/_test_truncation_check.jsonl"
        try:
            record = {
                "index": 999,
                "config_id": 999,
                "backend": "ollama",
                "model": "test",
                "text": long_text,
            }
            model_core._append_cache(test_path, record)
            loaded = model_core._load_cache(test_path)
            self.assertIn(999, loaded)
            self.assertEqual(len(loaded[999]["text"]), 2500)
            self.assertGreater(len(loaded[999]["text"]), 1500)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_single_final_answer_line(self):
        """Verify ensure_single_final_answer produces exactly one final answer line at the end."""
        text_with_duplicates = (
            "Step 1: Calculate.\n"
            "Final answer: 42\n\n"
            "**Final Answer: 42**\n"
            "Final Answer: 42"
        )
        cleaned = model_core.ensure_single_final_answer(text_with_duplicates, 42.0)
        final_lines = [l for l in cleaned.splitlines() if "final answer" in l.lower()]
        self.assertEqual(len(final_lines), 1)
        self.assertTrue(cleaned.endswith("**Final Answer: 42**"))

    def test_dedupe_on_load(self):
        """Verify cache loader keeps the last record when duplicate question indices appear."""
        test_path = "results/raw/_test_dedupe_check.jsonl"
        try:
            r1 = {"index": 10, "prediction": 1.0, "is_correct": False}
            r2 = {"index": 10, "prediction": 2.0, "is_correct": True}
            model_core._append_cache(test_path, r1)
            model_core._append_cache(test_path, r2)
            loaded = model_core._load_cache(test_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[10]["prediction"], 2.0)
            self.assertTrue(loaded[10]["is_correct"])
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_calculator_only_returns_prediction(self):
        """Verify calculator_agent executes code via safe evaluator and returns numeric output."""
        res = model_core.calculator_agent("If John has 5 apples and eats 2, how many does he have left?")
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 3.0, places=2)

    def test_llm_call_counter(self):
        """Verify LLM call counter resets and increments properly."""
        model_core.reset_llm_call_counter()
        self.assertEqual(model_core.get_llm_call_count(), 0)
        # Calling chat increments counter
        model_core.chat("You are a helper.", "Reply with 1.", max_new_tokens=10)
        self.assertEqual(model_core.get_llm_call_count(), 1)



class TestNewBenchmarkFeatures(unittest.TestCase):
    """Tests for Part A/B additions: llm_calls, probe loader, judge model check, slides."""

    def test_cfg3_llm_calls_ge_2(self):
        """Every row in the cfg3 cache must have llm_calls >= 2 (draft + critic)."""
        path = os.path.join("results", "raw", "gsm8k_cfg3_N40_seed42.jsonl")
        if not os.path.exists(path):
            self.skipTest("cfg3 cache not found; run benchmark first")
        cached = model_core._load_cache(path)
        self.assertEqual(len(cached), 40, "cfg3 cache must have exactly 40 rows")
        bad = [idx for idx, r in cached.items() if r.get("llm_calls", 0) < 2]
        self.assertEqual(bad, [], f"cfg3 rows with < 2 llm_calls: {bad}")

    def test_cfg4_llm_calls_ge_2(self):
        """Every row in the cfg4 cache must have llm_calls >= 2 (draft + calculator)."""
        path = os.path.join("results", "raw", "gsm8k_cfg4_N40_seed42.jsonl")
        if not os.path.exists(path):
            self.skipTest("cfg4 cache not found; run benchmark first")
        cached = model_core._load_cache(path)
        self.assertEqual(len(cached), 40, "cfg4 cache must have exactly 40 rows")
        bad = [idx for idx, r in cached.items() if r.get("llm_calls", 0) < 2]
        self.assertEqual(bad, [], f"cfg4 rows with < 2 llm_calls: {bad}")

    def test_judge_uses_different_model(self):
        """evaluate_truthfulqa_openended must raise if judge_model == MODEL."""
        # Temporarily set MODEL to a known value and confirm ValueError is raised.
        original_model = model_core.MODEL
        try:
            model_core.MODEL = "test_model"
            with self.assertRaises(ValueError) as ctx:
                model_core.evaluate_truthfulqa_openended(
                    [],
                    judge_model="test_model",  # same as MODEL -> must raise
                )
            self.assertIn("judge_model must differ", str(ctx.exception))
        finally:
            model_core.MODEL = original_model

    def test_probe_set_loader(self):
        """data/hallucination_probe.json must have exactly 20 false_premise + 10 control entries."""
        probe_path = os.path.join("data", "hallucination_probe.json")
        self.assertTrue(os.path.exists(probe_path), "data/hallucination_probe.json must exist")
        with open(probe_path, encoding="utf-8") as f:
            probe = json.load(f)
        false_premise = [q for q in probe if q.get("type") == "false_premise"]
        controls = [q for q in probe if q.get("type") == "control"]
        self.assertEqual(len(false_premise), 20, "Must have exactly 20 false_premise questions")
        self.assertEqual(len(controls), 10, "Must have exactly 10 control questions")

    def test_no_guarantees_in_slides(self):
        """slides_content.md must not contain the word 'guarantees'."""
        slides_path = "slides_content.md"
        if not os.path.exists(slides_path):
            self.skipTest("slides_content.md not found")
        with open(slides_path, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("guarantees", content.lower(),
                         "slides_content.md must not use 'guarantees' — use measured/verified language")

    def test_truthfulqa_perm_function_exists(self):
        """evaluate_truthfulqa_permutation must exist in model_core."""
        self.assertTrue(
            hasattr(model_core, "evaluate_truthfulqa_permutation"),
            "model_core must expose evaluate_truthfulqa_permutation()"
        )

    def test_default_critic_model_defined(self):
        """DEFAULT_CRITIC_MODEL must be defined and must differ from DEFAULT_MODEL."""
        self.assertTrue(
            hasattr(model_core, "DEFAULT_CRITIC_MODEL"),
            "model_core must define DEFAULT_CRITIC_MODEL"
        )
        self.assertNotEqual(
            model_core.DEFAULT_CRITIC_MODEL,
            model_core.DEFAULT_MODEL,
            "DEFAULT_CRITIC_MODEL must differ from DEFAULT_MODEL (generator)"
        )

    def test_judge_model_never_equals_generator(self):
        """Assert judge model can never be the same as the generator model."""
        self.assertNotEqual(model_core.DEFAULT_CRITIC_MODEL, model_core.MODEL)
        with self.assertRaises(ValueError):
            model_core._judge_truthfulness(
                "q", "a", ["c"], ["i"],
                judge_model="same_model",
                generator_model="same_model"
            )

    def test_probe_raw_file_has_30_rows(self):
        """results/hallucination_probe_raw.jsonl must exist and contain exactly 30 rows."""
        path = os.path.join("results", "hallucination_probe_raw.jsonl")
        self.assertTrue(os.path.exists(path), f"File {path} must exist")
        with open(path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 30, "hallucination_probe_raw.jsonl must have exactly 30 rows")
        row1 = json.loads(lines[0])
        self.assertIn("rule_label", row1)
        self.assertIn("llama3_label", row1)
        self.assertIn("answer", row1)

    def test_compliance_shows_na_for_non_structured_configs(self):
        """final_results_table.csv must show 'n/a' for non-structured configs (1 to 4)."""
        csv_path = os.path.join("results", "final_results_table.csv")
        self.assertTrue(os.path.exists(csv_path), f"File {csv_path} must exist")
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines[1:5]:  # rows 1 to 4
            parts = line.split(",")
            cfg_id = int(parts[0])
            self.assertIn(cfg_id, (1, 2, 3, 4))
            compliance = parts[-1]
            self.assertEqual(compliance, "n/a", f"Config {cfg_id} format compliance must be 'n/a', got '{compliance}'")
