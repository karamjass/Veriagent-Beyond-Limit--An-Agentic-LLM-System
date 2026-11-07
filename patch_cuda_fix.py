"""
Patch model_core.py to fix the CUDA VRAM overflow crash:
1. Lower default OLLAMA_CONTEXT_TOKENS from 8192 to 4096
2. Add OLLAMA_NUM_GPU env-var support
3. Add retry-with-fallback in chat() when Ollama crashes with HTTP 500 CUDA error
4. Add _build_options() helper that injects num_gpu when set
"""
import re

src = "model_core.py"
with open(src, "r", encoding="utf-8") as f:
    text = f.read()

# ── 1. Lower default context tokens and add GPU layer cap ─────────────────────
old_ctx = 'OLLAMA_CONTEXT_TOKENS = int(os.environ.get("OLLAMA_CONTEXT_TOKENS", "8192"))\r\n_model_lock = threading.RLock()'
new_ctx = (
    '# Lowered default 8192 -> 4096 to prevent CUDA VRAM overflow on 12 GB GPU.\r\n'
    '# Raise via OLLAMA_CONTEXT_TOKENS=8192 only after confirming stability.\r\n'
    'OLLAMA_CONTEXT_TOKENS = int(os.environ.get("OLLAMA_CONTEXT_TOKENS", "4096"))\r\n'
    '# Set OLLAMA_NUM_GPU=0 to run on CPU; 1-100 to cap GPU layers and reduce VRAM.\r\n'
    '_OLLAMA_NUM_GPU: "int | None" = (\r\n'
    '    int(os.environ["OLLAMA_NUM_GPU"]) if "OLLAMA_NUM_GPU" in os.environ else None\r\n'
    ')\r\n'
    '_model_lock = threading.RLock()'
)

if old_ctx not in text:
    # Try with LF only
    old_ctx = old_ctx.replace("\r\n", "\n")
    new_ctx = new_ctx.replace("\r\n", "\n")

if old_ctx in text:
    text = text.replace(old_ctx, new_ctx)
    print("✅ Patch 1 applied: OLLAMA_CONTEXT_TOKENS default + OLLAMA_NUM_GPU")
else:
    print("⚠️  Patch 1 SKIPPED — could not find context tokens line")

# ── 2. Add _build_options() helper right before def chat( ─────────────────────
helper_fn = '''
def _build_options(num_predict: int, temperature: float) -> dict:
    """Build Ollama options dict, injecting GPU layer cap when OLLAMA_NUM_GPU is set."""
    opts: dict = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": max(2048, min(OLLAMA_CONTEXT_TOKENS, 32768)),
    }
    if _OLLAMA_NUM_GPU is not None:
        opts["num_gpu"] = _OLLAMA_NUM_GPU
    return opts

'''

marker_chat = "def chat("
if helper_fn.strip() not in text and "_build_options" not in text:
    # Insert before first occurrence of def chat(
    text = text.replace(marker_chat, helper_fn + marker_chat, 1)
    print("✅ Patch 2 applied: _build_options() helper added")
else:
    print("ℹ️  Patch 2 skipped — _build_options already present")

# ── 3. Rewrite the chat() function body to use _build_options + crash retry ──
old_chat_body = r"""    \"\"\"Run chat completion with optional thinking mode, temperature, and model override.\"\"\"
    target_model = model or MODEL
    predict_tokens = int(max_new_tokens)
    if thinking:
        predict_tokens = max(predict_tokens, 1800)
    capped = max(1, min(predict_tokens, _allow_up_to))
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": thinking,
        "options": {
            "temperature": temperature,
            "num_predict": capped,
            "num_ctx": max(4096, min(OLLAMA_CONTEXT_TOKENS, 32768)),
        },
    }
    with _model_lock:
        response = _ollama_request("/api/chat", payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    message = response.get("message") or {}
    content = str(message.get("content") or message.get("thinking") or "").strip()
    if not content:
        raise ModelUnavailable("Ollama returned an empty response. Try again or check Ollama logs.")
    done_reason = str(response.get("done_reason") or "").lower()
    was_cut = done_reason == "length" or (response.get("done") is True and done_reason == "")
    if was_cut and done_reason == "length":
        content = content + "\\n\\n[Note: the response may have been cut short by the output limit.]"
    clean_content, _ = strip_thinking_trace(content)
    return clean_content or content"""

# We'll do a simpler targeted replacement using unique strings
OLD_OPTIONS_LINE = '"num_ctx": max(4096, min(OLLAMA_CONTEXT_TOKENS, 32768)),'
NEW_CHAT_BLOCK = '''    """Run chat completion with optional thinking mode, temperature, and model override.

    On CUDA crashes (HTTP 500 containing "CUDA error" or "stack-based buffer"),
    waits 8 s and retries once with halved num_ctx to reduce VRAM pressure.
    """
    _CUDA_ERR_MARKERS = ("cuda error", "stack-based buffer", "exit status 0xc0000409", "llama-server")

    target_model = model or MODEL
    predict_tokens = int(max_new_tokens)
    if thinking:
        predict_tokens = max(predict_tokens, 1800)
    capped = max(1, min(predict_tokens, _allow_up_to))

    def _make_payload(ctx_tokens: int) -> dict:
        opts = _build_options(capped, temperature)
        opts["num_ctx"] = max(2048, min(ctx_tokens, 32768))
        return {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": thinking,
            "options": opts,
        }

    def _extract_content(response: dict) -> str:
        message = response.get("message") or {}
        content = str(message.get("content") or message.get("thinking") or "").strip()
        if not content:
            raise ModelUnavailable("Ollama returned an empty response. Try again or check Ollama logs.")
        done_reason = str(response.get("done_reason") or "").lower()
        if done_reason == "length":
            content += "\\n\\n[Note: the response may have been cut short by the output limit.]"
        clean_content, _ = strip_thinking_trace(content)
        return clean_content or content

    with _model_lock:
        try:
            response = _ollama_request("/api/chat", _make_payload(OLLAMA_CONTEXT_TOKENS),
                                       timeout=OLLAMA_TIMEOUT_SECONDS)
        except ModelUnavailable as exc:
            err_lower = str(exc).lower()
            is_cuda_crash = any(m in err_lower for m in _CUDA_ERR_MARKERS)
            if not is_cuda_crash:
                raise
            # CUDA crash: wait and retry once with reduced context window
            import sys
            print(f"[chat] CUDA crash detected — waiting 8 s then retrying with "
                  f"num_ctx={OLLAMA_CONTEXT_TOKENS // 2}", file=sys.stderr)
            time.sleep(8)
            fallback_ctx = max(2048, OLLAMA_CONTEXT_TOKENS // 2)
            response = _ollama_request("/api/chat", _make_payload(fallback_ctx),
                                       timeout=OLLAMA_TIMEOUT_SECONDS)
    return _extract_content(response)'''

# Replace by locating the exact docstring + body pattern
chat_start = '    """Run chat completion with optional thinking mode, temperature, and model override."""'
chat_end_marker = '    clean_content, _ = strip_thinking_trace(content)\n    return clean_content or content'
chat_end_marker_crlf = '    clean_content, _ = strip_thinking_trace(content)\r\n    return clean_content or content'

if chat_start in text:
    # Find the return statement after strip_thinking_trace
    for end_m in (chat_end_marker_crlf, chat_end_marker):
        idx_start = text.index(chat_start)
        idx_end = text.find(end_m, idx_start)
        if idx_end != -1:
            old_block = text[idx_start: idx_end + len(end_m)]
            text = text[:idx_start] + NEW_CHAT_BLOCK + text[idx_start + len(old_block):]
            print("✅ Patch 3 applied: chat() body upgraded with CUDA retry logic")
            break
    else:
        print("⚠️  Patch 3 SKIPPED — could not find end of chat() body")
else:
    print("⚠️  Patch 3 SKIPPED — could not find chat() docstring")

with open(src, "w", encoding="utf-8") as f:
    f.write(text)

print("\nAll patches complete. Run the unit tests to verify:")
print("  .venv\\Scripts\\python.exe -m unittest discover -s tests")

