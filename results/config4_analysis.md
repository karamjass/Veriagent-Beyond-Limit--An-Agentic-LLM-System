# Config 4 Analysis — Calculator Agreement Agreement Rule Investigation

**Date:** 2026-11-08  
**Data:** results/raw/gsm8k_cfg4_N40_seed42.jsonl (40 rows verified)  
**Baseline (Config 1):** 39/40 correct (97.5%)  
**Config 4:** 37/40 correct (92.5%)  
**Fixed vs Cfg1:** 0 questions  
**Broken vs Cfg1:** 2 questions (indices 8, 28)

---

## Root Cause: Calculator Returns 0 and Overrides Correct Draft

Both broken questions share the identical failure mode:

| Step | What happened |
|---|---|
| Draft agent | Generated a **correct** step-by-step solution with the right final answer |
| Calculator agent | Failed to extract a meaningful expression from the word problem → returned **0** |
| Agreement rule (Config 4) | Since `draft_val != 0`, the calculator disagreed → rule applied `calculator_val` (0) as final answer |
| Result | Correct draft (145 or 18) was silently overwritten by the wrong calculator output (0) |

This is a **calculator extraction failure**, not a reasoning failure. The draft was correct in both cases.

---

## Question #8 — Full Detail

**Question:**  
Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.

| | Value |
|---|---|
| Gold answer | 145 |
| Cfg1 (draft only) | **145** ✅ |
| Cfg4 draft prediction | 145 (correct!) |
| Calculator output | **0** (failed — could not parse word problem) |
| Agreement rule output | 0 (calculator overrides draft) ❌ |

**Agreement rule applied:**  
Since `draft_val (145) != calc_val (0)` and `calc_val is not None`, the rule outputs `calc_val = 0`.  
This directly contradicts the "do no harm" principle — the calculator broke a correct answer.

---

## Question #28 — Full Detail

**Question:**  
Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?

| | Value |
|---|---|
| Gold answer | 18 |
| Cfg1 (draft only) | **18** ✅ |
| Cfg4 draft prediction | 18 (correct!) |
| Calculator output | **0** (failed — could not parse multi-step word problem) |
| Agreement rule output | 0 (calculator overrides draft) ❌ |

---

## Finding

> **The Config 4 calculator-agreement rule can override a correct LLM draft when the calculator fails to parse a word problem and returns 0 (or any wrong value). This causes accuracy regressions on questions where the LLM draft is correct and the calculator cannot extract a meaningful code snippet.**

The rule as implemented:
```python
if c_val is not None and d_val is not None and close(c_val, d_val):
    text = ensure_single_final_answer(draft, c_val)   # agreement: use draft
elif c_val is not None:
    text = f"... **Final Answer: {fmt(c_val)}**"      # disagreement: use CALCULATOR ← BUG
else:
    text = ensure_single_final_answer(draft, d_val)   # no calc: use draft
```

The `elif c_val is not None` branch blindly trusts the calculator even when it returns 0 from a failed parse.

---

## Proposed Config 4B — Disagreement → Critic (config_id = 14)

Instead of defaulting to the calculator on disagreement, call the structured critic to arbitrate:

```python
elif config_id == 14:
    draft = draft_agent(question)
    c_val = calculator_agent(question)
    d_val = extract_answer(draft)
    if c_val is not None and d_val is not None and close(c_val, d_val):
        text = ensure_single_final_answer(draft, c_val)      # agree: use draft
    elif c_val is not None and d_val is not None:
        # Disagreement: ask critic to arbitrate between draft and calculator
        crit = critic_agent(question, draft)
        if critic_says_wrong(crit):
            # Critic agrees something is wrong; prefer calculator if sane, else draft
            text = ensure_single_final_answer(draft, c_val if c_val != 0.0 else d_val)
        else:
            # Critic says draft is fine → keep draft
            text = ensure_single_final_answer(draft, d_val)
    else:
        text = ensure_single_final_answer(draft, d_val)      # no calc: use draft
```

This requires >= 3 LLM calls on disagreement questions (draft + critic + implicit).

---

## Summary

| Config | Accuracy | Fixed | Broken | Note |
|---|---|---|---|---|
| Config 1 (baseline) | 39/40 (97.5%) | — | — | Solo draft |
| Config 4 (calc agree) | 37/40 (92.5%) | 0 | 2 | Calculator-override bug |
| Config 4B (critic arbitrate) | TBD | TBD | TBD | Planned; smoke N=3 first |

**Conclusion:** Config 4 performs worse than Config 1 (−2 questions) solely due to calculator failures returning 0. The agreement rule provides no benefit on these questions and actively hurts accuracy. Config 4B adds critic arbitration on disagreement.
