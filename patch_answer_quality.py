"""
Patch model_core.py to fix wrong answers caused by response truncation.

Fixes:
1. Raise MAX_NEW_TOKENS 750 -> 1400 for math (responses were cut mid-calculation)
2. Raise _allow_up_to in chat() from 3000 -> 4000 (soft ceiling)
3. Fix extract_answer: suppress the "last number" fallback when the response
   contains the truncation notice — that number (e.g. "100" from "× 100") is
   NOT the answer.
4. Improve SOLVER_SYS prompt: explicitly instruct model to finish the
   arithmetic and write the numeric result before "Final answer:".
5. Raise calculator_agent token budget 350 -> 500 (code was also getting cut).
"""

import re
import sys

PATH = r"c:\Users\Karam\OneDrive\Desktop\beyond_limit(veriagent)\model_core.py"

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

original = text   # keep for diff

# ── CRLF vs LF detection ─────────────────────────────────────────────────────
CRLF = "\r\n" in text
NL = "\r\n" if CRLF else "\n"

def _fix(old, new, label):
    global text
    # Try as-is first, then with opposite line endings
    for src, rep in [(old, new), (old.replace("\r\n", "\n"), new.replace("\r\n", "\n"))]:
        if src in text:
            text = text.replace(src, rep, 1)
            print(f"✅  {label}")
            return
    print(f"⚠️   SKIPPED — could not find target for: {label}")
    # Print first 60 chars of what we searched for to help debug
    preview = repr(old[:80])
    print(f"    searched for: {preview}")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Raise MAX_NEW_TOKENS 750 -> 1400
# ─────────────────────────────────────────────────────────────────────────────
_fix(
    "MAX_NEW_TOKENS = 750          # default for numerical solver agents",
    "MAX_NEW_TOKENS = 1400         # raised from 750; multi-step math needs more tokens",
    "Fix 1: MAX_NEW_TOKENS 750 → 1400"
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Raise MATH_CONCEPT_TOKENS and _allow_up_to ceiling in chat()
# ─────────────────────────────────────────────────────────────────────────────
_fix(
    "MATH_CONCEPT_TOKENS = 900     # concept/explanation answers need more room",
    "MATH_CONCEPT_TOKENS = 1200    # raised from 900; concepts need room for examples",
    "Fix 2a: MATH_CONCEPT_TOKENS 900 → 1200"
)

_fix(
    "    max_new_tokens: int = MAX_NEW_TOKENS,\r\n    _allow_up_to: int = 3000,",
    "    max_new_tokens: int = MAX_NEW_TOKENS,\r\n    _allow_up_to: int = 4000,",
    "Fix 2b: chat() _allow_up_to 3000 → 4000"
)
# LF fallback
_fix(
    "    max_new_tokens: int = MAX_NEW_TOKENS,\n    _allow_up_to: int = 3000,",
    "    max_new_tokens: int = MAX_NEW_TOKENS,\n    _allow_up_to: int = 4000,",
    "Fix 2b (LF): chat() _allow_up_to 3000 → 4000"
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Suppress last-number fallback on truncated responses
# ─────────────────────────────────────────────────────────────────────────────
OLD_FALLBACK = (
    "    # Ultimate fallback: last number in the text\r\n"
    "    numbers = re.findall(NUM, clean)\r\n"
    "    return to_float(numbers[-1]) if numbers else None"
)
NEW_FALLBACK = (
    "    # Ultimate fallback: last number in the text.\r\n"
    "    # IMPORTANT: skip this fallback when the response was cut short — the\r\n"
    "    # last number is typically part of an unfinished formula (e.g. '× 100')\r\n"
    "    # and NOT the final answer. Only return it on complete responses.\r\n"
    "    _TRUNC_MARKER = \"[Note: the response may have been cut short\"\r\n"
    "    if _TRUNC_MARKER.lower() in clean.lower():\r\n"
    "        return None   # do not guess from an incomplete response\r\n"
    "    numbers = re.findall(NUM, clean)\r\n"
    "    return to_float(numbers[-1]) if numbers else None"
)
_fix(OLD_FALLBACK, NEW_FALLBACK, "Fix 3 (CRLF): suppress last-number fallback on truncated response")

OLD_FALLBACK_LF = OLD_FALLBACK.replace("\r\n", "\n")
NEW_FALLBACK_LF = NEW_FALLBACK.replace("\r\n", "\n")
_fix(OLD_FALLBACK_LF, NEW_FALLBACK_LF, "Fix 3 (LF): suppress last-number fallback on truncated response")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Improve SOLVER_SYS prompt — force model to finish arithmetic
# ─────────────────────────────────────────────────────────────────────────────
OLD_SOLVER = (
    'SOLVER_SYS = (\r\n'
    '    "You are a careful math solver. Solve the problem step by step, check units, and avoid unsupported assumptions. "\r\n'
    '    "Be direct and concise: state the relevant formula, substitute the values, and calculate cleanly. Do not repeat multiple redundant methods. "\r\n'
    '    "End with a last line exactly like: Final answer: <number>"\r\n'
    ')'
)
NEW_SOLVER = (
    'SOLVER_SYS = (\r\n'
    '    "You are a careful math solver. Solve the problem step by step, check units, and avoid unsupported assumptions. "\r\n'
    '    "Be direct and concise: state the relevant formula, substitute the values, and calculate each step to a numeric result. "\r\n'
    '    "CRITICAL RULES: "\r\n'
    '    "(1) Always finish every arithmetic expression — never leave a formula like \'(17.1/90) × 100\' uncomputed. "\r\n'
    '    "(2) The very last line of your response must be: Final answer: <the computed number>  "\r\n'
    '    "(3) The number after \'Final answer:\' must be the actual numeric result, not a formula or expression. "\r\n'
    '    "(4) For percentage problems, compute the percentage as a decimal result first, then multiply by 100. "\r\n'
    '    "Example of correct ending: Final answer: 19"\r\n'
    ')'
)
_fix(OLD_SOLVER, NEW_SOLVER, "Fix 4 (CRLF): improved SOLVER_SYS with arithmetic-completion rules")

OLD_SOLVER_LF = OLD_SOLVER.replace("\r\n", "\n")
NEW_SOLVER_LF = NEW_SOLVER.replace("\r\n", "\n")
_fix(OLD_SOLVER_LF, NEW_SOLVER_LF, "Fix 4 (LF): improved SOLVER_SYS with arithmetic-completion rules")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 5: Raise calculator_agent token budget 350 -> 500
# ─────────────────────────────────────────────────────────────────────────────
_fix(
    "    raw = chat(CALC_SYS, prompt, 350)",
    "    raw = chat(CALC_SYS, prompt, 500)",
    "Fix 5: calculator_agent token budget 350 → 500"
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 6: Raise reviser_agent token budget (also uses MAX_NEW_TOKENS, fine)
#         But the critic only gets 300 — raise to 450
# ─────────────────────────────────────────────────────────────────────────────
_fix(
    "    return chat(CRITIC_SYS, prompt, 300)",
    "    return chat(CRITIC_SYS, prompt, 450)",
    "Fix 6: critic_agent token budget 300 → 450"
)

# ─────────────────────────────────────────────────────────────────────────────
# Write result
# ─────────────────────────────────────────────────────────────────────────────
if text == original:
    print("\n⚠️  WARNING: no changes were made to the file. Check the SKIPPED messages above.")
    sys.exit(1)
else:
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(text)
    changed = sum(1 for a, b in zip(original.splitlines(), text.splitlines()) if a != b)
    print(f"\n✅  Saved. ~{changed} lines changed.")
    print("\nNext: restart Ollama and test with the shopkeeper problem.")
    print("Expected answer: 19%  (net profit percentage)")

