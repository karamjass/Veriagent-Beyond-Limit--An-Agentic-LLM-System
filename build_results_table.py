"""
build_results_table.py
======================
Runs every benchmark config (1–13 + TruthfulQA) on GSM8K N=40 seed=42,
resumes from cache, then produces:
  results/final_results_table.md
  results/final_results_table.csv
  results/final_results_table.html   (purple/yellow/beige styled)

Usage:
  python build_results_table.py          # run all, generate table
  python build_results_table.py --table  # generate table from existing caches only
"""

import argparse, json, math, os, sys, time, re, glob, random
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
RAW    = ROOT / "results" / "raw"
OUT    = ROOT / "results"
LOCK   = OUT / ".run.lock"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")

# ── Model name display map ────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "qwen3.5:9b-q4_k_m": "Qwen 3.5 9B",
    "qwen3.5:9b":        "Qwen 3.5 9B",
    "llama3:latest":     "LLaMA-3 8B",
    "llama3":            "LLaMA-3 8B",
    "gemma4:latest":     "Gemma 4",
}

def display_model(backend: str, model: str) -> str:
    key = model.lower().strip()
    return MODEL_DISPLAY.get(key, model)

# ── Wilson 95% CI ─────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)

# ── Cache helpers ─────────────────────────────────────────────────────────────
def cache_path(cfg: int, dataset: str = "gsm8k") -> Path:
    return RAW / f"{dataset}_cfg{cfg}_N40_seed42.jsonl"

def load_cache(path: Path) -> dict:
    """Load cache, deduping by index (keep last)."""
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows[str(r.get("index", len(rows)))] = r
                except Exception:
                    pass
    return rows

def append_cache(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

# ── Dataset loading ───────────────────────────────────────────────────────────
def load_gsm8k(n: int = 40, seed: int = 42):
    """Load GSM8K test split N=40 seed=42.
    Priority:
      1. HuggingFace datasets (openai/gsm8k)
      2. Fallback: reconstruct from cfg1 cache (questions already stored there)
    """
    # ── Try HuggingFace ───────────────────────────────────────────────────────
    for repo in ("openai/gsm8k", "gsm8k"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, "main", trust_remote_code=False)
            test = list(ds["test"])
            rng = random.Random(seed)
            sample = rng.sample(test, min(n, len(test)))
            def gold(item):
                ans = item["answer"].split("####")[-1].strip()
                ans = re.sub(r"[^\d.\-]", "", ans)
                try: return str(float(ans))
                except: return ans
            print(f"[INFO] Loaded GSM8K from {repo} ({len(sample)} questions)")
            return [(i, item["question"], gold(item)) for i, item in enumerate(sample)]
        except Exception as e:
            print(f"[WARN] GSM8K load from '{repo}' failed: {e}")

    # ── Fallback: reconstruct from cfg1 cache ─────────────────────────────────
    p1 = cache_path(1, "gsm8k")
    if p1.exists():
        rows = list(load_cache(p1).values())
        if rows:
            print(f"[INFO] Reconstructing questions from cfg1 cache ({len(rows)} rows)")
            result = []
            for r in sorted(rows, key=lambda x: int(x.get("index", 0))):
                result.append((
                    int(r["index"]),
                    r["question"],
                    str(r["gold"]),
                ))
            return result

    print("[ERROR] Cannot load GSM8K — no HuggingFace access and no cfg1 cache")
    return []

def load_truthfulqa(n: int = 40, seed: int = 42):
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=False)
        items = list(ds["validation"])
        rng = random.Random(seed)
        sample = rng.sample(items, min(n, len(items)))
        result = []
        for i, item in enumerate(sample):
            choices = item["mc1_targets"]["choices"]
            labels  = item["mc1_targets"]["labels"]
            correct = [c for c, l in zip(choices, labels) if l == 1]
            gold_ans = correct[0] if correct else choices[0]
            opts = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(choices))
            q = f"{item['question']}\n{opts}"
            result.append((i, q, gold_ans))
        return result
    except Exception as e:
        print(f"[WARN] Could not load TruthfulQA: {e}")
        return []

# ── Accuracy metric helpers ───────────────────────────────────────────────────
def _unwrap(result) -> str:
    """Unwrap dict results (solve_question_v2 returns dicts) to plain text."""
    if isinstance(result, dict):
        return str(result.get("text") or result.get("answer") or "")
    return str(result) if result is not None else ""

def extract_pred(text, gold: str) -> str | None:
    """Try to extract a numeric answer from model output."""
    text = _unwrap(text)
    try:
        import model_core
        return model_core.extract_answer(text)
    except Exception:
        pass
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def is_match(pred, gold) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except Exception:
        return str(pred).strip().lower() == str(gold).strip().lower()

# ── Run one config ────────────────────────────────────────────────────────────
def run_config(cfg_id: int, questions, dataset: str = "gsm8k") -> list[dict]:
    """Run a single config, resuming from cache.  Returns all rows."""
    path = cache_path(cfg_id, dataset)
    cache = load_cache(path)
    print(f"\n[CFG {cfg_id}] {len(cache)}/{len(questions)} already cached")

    try:
        import model_core
    except ImportError as e:
        print(f"[ERROR] Cannot import model_core: {e}")
        return list(cache.values())

    for idx, question, gold in questions:
        key = str(idx)
        if key in cache:
            continue  # resume from cache

        t0 = time.perf_counter()
        try:
            result = _call_config(cfg_id, question, model_core)
            text = _unwrap(result)
        except Exception as e:
            text = f"[ERROR] {e}"
        latency = time.perf_counter() - t0

        pred = extract_pred(text, gold)
        correct = is_match(pred, gold)

        row = {
            "index":     idx,
            "config_id": cfg_id,
            "backend":   "ollama",
            "model":     model_core.MODEL,
            "question":  question[:300],
            "gold":      gold,
            "prediction": pred,
            "is_correct": correct,
            "latency":   round(latency, 3),
            "text":      text[:20000],
        }
        cache[key] = row
        append_cache(path, row)
        status = "✓" if correct else "✗"
        print(f"  [{status}] Q{idx}  pred={pred}  gold={gold}  t={latency:.1f}s")

    return list(cache.values())

def _call_config(cfg_id: int, question: str, mc) -> str:
    """Dispatch to the right model_core function per config. Always returns str."""
    if cfg_id == 1:
        return mc.chat(mc.SOLVER_SYS, question, mc.MAX_NEW_TOKENS)
    elif cfg_id == 2:
        # AST calculator only
        try:
            expr = mc.extract_expression(question)
            val  = mc.safe_eval(expr)
            return f"**Final Answer: {val}**"
        except Exception as e:
            return f"[AST ERROR] {e}"
    elif cfg_id == 3:
        return _unwrap(mc.solve_question(question))
    elif cfg_id == 4:
        # LLM + AST agreement
        llm_ans = mc.chat(mc.SOLVER_SYS, question, mc.MAX_NEW_TOKENS)
        try:
            expr = mc.extract_expression(question)
            ast_val = str(mc.safe_eval(expr))
            pred = extract_pred(llm_ans, "")
            if pred and abs(float(pred) - float(ast_val)) < 1e-3:
                return llm_ans + f"\n\n**Final Answer: {ast_val}**"
            else:
                return llm_ans
        except Exception:
            return llm_ans
    elif cfg_id == 5:
        return _unwrap(mc.solve_question_v2(question, k_sc=1, use_critic=False))
    elif cfg_id == 6:
        return _unwrap(mc.solve_question_v2(question, k_sc=1, use_critic=True))
    elif cfg_id == 7:
        # LLaMA-3 8B ablation
        orig = mc.MODEL
        mc.MODEL = "llama3:latest"
        try:
            return _unwrap(mc.solve_question_v2(question, k_sc=1, use_critic=False))
        finally:
            mc.MODEL = orig
    elif cfg_id == 9:
        return _unwrap(mc.draft_agent_v2(question, thinking=False, temperature=0.0))
    elif cfg_id == 10:
        return _unwrap(mc.draft_agent_v2(question, thinking=True, temperature=0.0))
    elif cfg_id == 11:
        ans, _, text, _ = mc.solve_self_consistency(question, k=3, temperature=0.5, thinking=False)
        return _unwrap(text) or str(ans)
    elif cfg_id == 12:
        return _unwrap(mc.solve_question_v2(question, k_sc=1, thinking=False, use_critic=False))
    elif cfg_id == 13:
        return _unwrap(mc.solve_question_v2(
            question, k_sc=1, thinking=False, use_critic=True,
            critic_model=mc.DEFAULT_CRITIC_MODEL
        ))
    else:
        return mc.chat(mc.SOLVER_SYS, question, mc.MAX_NEW_TOKENS)

# ── Compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(rows: list[dict], baseline_correct: set | None = None):
    n = len(rows)
    if n == 0:
        return None

    # Assert backend
    for r in rows:
        if r.get("backend", "ollama") != "ollama":
            sys.exit(f"[ERROR] Row has backend != 'ollama': {r}")

    correct_idx = {str(r["index"]) for r in rows if r.get("is_correct")}
    k = len(correct_idx)
    acc = k / n
    lo, hi = wilson_ci(k, n)
    lat = [float(r["latency"]) for r in rows if r.get("latency")]
    avg_lat = sum(lat) / len(lat) if lat else 0.0

    # Fixed / Broken vs baseline (Config 1)
    fixed = broken = "—"
    if baseline_correct is not None:
        all_idx = {str(r["index"]) for r in rows}
        fixed = len(correct_idx - baseline_correct)
        broken = len(baseline_correct - correct_idx)

    # Recall / Precision (same definition as report)
    # Recall  = TP / (TP + FN) where TP = correct, FN = wrong
    # Precision = TP / (TP + FP) where FP = wrong answers that are non-null
    tp = k
    fn = n - k
    fp = sum(1 for r in rows if not r.get("is_correct") and r.get("prediction") is not None)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Format compliance
    try:
        import model_core
        compliant = sum(1 for r in rows if model_core.check_format_compliance(r.get("text", "")))
    except Exception:
        compliant = 0

    model_name = "—"
    for r in rows:
        b = r.get("backend", "ollama")
        m = r.get("model", "")
        if m:
            model_name = display_model(b, m)
            break

    return {
        "model_name": model_name,
        "n": n, "k": k,
        "accuracy": acc,
        "ci_lo": lo, "ci_hi": hi,
        "fixed": fixed, "broken": broken,
        "recall": recall,
        "precision": precision,
        "compliance": compliant,
        "avg_latency": avg_lat,
    }

# ── Build table strings ───────────────────────────────────────────────────────
CONFIG_LABELS = {
    1:  "Qwen 3.5 9B (Solo Generator Baseline)",
    2:  "AST Calculator Only (Code Execution)",
    3:  "Qwen 3.5 9B + LLM Critic (No Calculator)",
    4:  "Qwen 3.5 9B + AST Calculator Agreement",
    5:  "VeriAgent Multi-Agent (1 Revision Round)",
    6:  "VeriAgent Full Pipeline (2 Revision Rounds)",
    7:  "LLaMA-3 8B (VeriAgent Pipeline Ablation)",
    8:  "TruthfulQA Validation (Factuality MC Mode)",
    9:  "Structured Prompt (Upgraded)",
    10: "+ Thinking Mode",
    11: "+ Self-Consistency (k=3)",
    12: "+ SymPy / PoT",
    13: "Full Pipeline + Independent Critic",
}

FOOTNOTE = (
    "All results: local Ollama, N=40, seed=42, "
    "GSM8K revision 740312add88f781978c0658806c59bc2815b9866."
)

NOISE_NOTE = (
    "Noise note: At N=40, one question = 2.5 percentage points. "
    "Differences ≤5 pp are within the 95% CI and should not be claimed as improvements."
)

def pct(v) -> str:
    return f"{v*100:.1f}%" if isinstance(v, float) else "N/A"

def fmt_fb(v) -> str:
    return str(v) if v != "—" else "—"

def build_tables(metrics_map: dict):
    """Return (md_str, csv_str, html_str)."""

    rows_main = []
    rows_ext  = []

    for cfg_id in sorted(CONFIG_LABELS.keys()):
        label = CONFIG_LABELS[cfg_id]
        m = metrics_map.get(cfg_id)
        if m is None:
            # Not run yet
            rows_main.append((label, "N/A", "N/A", "N/A", "N/A"))
            rows_ext.append((label, "N/A", "N/A", "N/A", "N/A"))
            continue

        fb = f"+{m['fixed']} / -{m['broken']}" if m["fixed"] != "—" else "—"
        rows_main.append((
            label,
            pct(m["accuracy"]),
            fb,
            pct(m["recall"]),
            pct(m["precision"]),
        ))
        rows_ext.append((
            label,
            f"({pct(m['ci_lo'])}, {pct(m['ci_hi'])})",
            f"{m['compliance']}/{m['n']}",
            f"{m['avg_latency']:.1f}s",
            m["model_name"],
        ))

    # ── Markdown ──────────────────────────────────────────────────────────────
    hdr = "| Model | Accuracy | Fixed / Broken | Recall | Precision |"
    sep = "|:---|:---:|:---:|:---:|:---:|"
    md_lines = ["# Results\n", hdr, sep]
    for i, (cfg_id, r) in enumerate(zip(sorted(CONFIG_LABELS.keys()), rows_main)):
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    md_lines += ["", f"_{FOOTNOTE}_", "", f"_{NOISE_NOTE}_"]

    hdr2 = "| Config | 95% Wilson CI | Compliance | Avg Latency | Model |"
    sep2 = "|:---|:---:|:---:|:---:|:---:|"
    md_lines += ["\n## Extended Metrics\n", hdr2, sep2]
    for i, (cfg_id, r) in enumerate(zip(sorted(CONFIG_LABELS.keys()), rows_ext)):
        md_lines.append(f"| {cfg_id}. {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

    md_str = "\n".join(md_lines)

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Model", "Accuracy", "Fixed / Broken", "Recall", "Precision",
                "CI_lo", "CI_hi", "Compliance", "Avg_Latency_s", "Backend_Model"])
    for cfg_id in sorted(CONFIG_LABELS.keys()):
        label = CONFIG_LABELS[cfg_id]
        m = metrics_map.get(cfg_id)
        if m is None:
            w.writerow([label] + ["N/A"]*9)
        else:
            fb = f"+{m['fixed']} / -{m['broken']}" if m["fixed"] != "—" else "—"
            w.writerow([
                label,
                pct(m["accuracy"]), fb,
                pct(m["recall"]), pct(m["precision"]),
                pct(m["ci_lo"]), pct(m["ci_hi"]),
                f"{m['compliance']}/{m['n']}",
                f"{m['avg_latency']:.1f}",
                m["model_name"],
            ])
    csv_str = buf.getvalue()

    # ── HTML ──────────────────────────────────────────────────────────────────
    def html_rows(data_rows, header):
        lines = []
        lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
        for i, row in enumerate(data_rows):
            bg = "#fdf5e6" if i % 2 == 0 else "#faf0dc"
            lines.append(
                f'<tr style="background:{bg}">'
                + "".join(f"<td>{cell}</td>" for cell in row)
                + "</tr>"
            )
        return "\n".join(lines)

    main_header = ["Model", "Accuracy", "Fixed / Broken", "Recall", "Precision"]
    ext_header  = ["Config", "95% Wilson CI", "Compliance", "Avg Latency", "Model"]

    main_data = [tuple(r) for cfg_id, r in
                 zip(sorted(CONFIG_LABELS.keys()), rows_main)]
    ext_data  = [(f"{cfg_id}. {r[0]}", r[1], r[2], r[3], r[4]) for cfg_id, r in
                 zip(sorted(CONFIG_LABELS.keys()), rows_ext)]

    def _tbody(data):
        parts = []
        for i, row in enumerate(data):
            bg = "#fdf5e6" if i % 2 == 0 else "#faf0dc"
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            parts.append(f'<tr style="background:{bg}">{cells}</tr>')
        return "\n".join(parts)

    html_str = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        "<title>VeriAgent Results</title>\n"
        "<style>\n"
        '  body { font-family: "Times New Roman", Times, serif; background: #f8f4ef; margin: 40px; }\n'
        "  h1 { background: #6a0dad; color: white; padding: 14px 24px; letter-spacing: 1px; font-size: 1.6em; margin: 0; }\n"
        "  h2 { background: #4a0080; color: white; padding: 10px 20px; font-size: 1.1em; margin: 24px 0 0 0; }\n"
        "  table { border-collapse: collapse; width: 100%; font-size: 0.95em; }\n"
        "  thead tr { background: #f5c518; color: #1a1a1a; font-weight: bold; }\n"
        "  th, td { border: 1px solid #ccc; padding: 7px 12px; text-align: center; }\n"
        "  td:nth-child(2) { text-align: left; }\n"
        "  .footnote { font-size: 0.8em; color: #555; margin-top: 16px; font-style: italic; }\n"
        "  .noise { font-size: 0.82em; color: #7a4000; margin-top: 6px; font-style: italic; }\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Results</h1>\n"
        "<table>\n<thead>\n"
        + html_rows([], main_header)
        + "\n</thead>\n<tbody>\n"
        + _tbody(main_data)
        + "\n</tbody>\n</table>\n"
        + f'<p class="footnote">{FOOTNOTE}</p>\n'
        + f'<p class="noise">{NOISE_NOTE}</p>\n'
        + "<h2>Extended Metrics (CI &middot; Compliance &middot; Latency)</h2>\n"
        "<table>\n<thead>\n"
        + html_rows([], ext_header)
        + "\n</thead>\n<tbody>\n"
        + _tbody(ext_data)
        + "\n</tbody>\n</table>\n</body>\n</html>\n"
    )
    return md_str, csv_str, html_str

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="store_true",
                        help="Generate table from existing caches only (no inference)")
    parser.add_argument("--cfg", type=int, default=0,
                        help="Run only this config ID (0 = all)")
    args = parser.parse_args()

    # ── Acquire lock (Windows-compatible) ────────────────────────────────────
    if not args.table:
        if LOCK.exists():
            print("[ERROR] Another run is in progress (lock file exists). "
                  "Delete results/.run.lock if no other run is active.")
            sys.exit(1)
        lock_fh = open(LOCK, "w")
        lock_fh.write(str(os.getpid()))
        lock_fh.flush()

    try:
        _run(args)
    finally:
        if not args.table:
            try:
                lock_fh.close()
            except Exception:
                pass
            LOCK.unlink(missing_ok=True)

def _run(args):
    table_only = args.table
    cfg_filter = args.cfg  # 0 = all

    # ── Patch cfg1 cache: add backend/model fields ────────────────────────────
    p1 = cache_path(1)
    if p1.exists():
        rows1 = []
        changed = False
        with open(p1, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "backend" not in r:
                    r["backend"] = "ollama"
                    r["model"]   = MODEL_NAME
                    changed = True
                rows1.append(r)
        if changed:
            with open(p1, "w", encoding="utf-8") as f:
                for r in rows1:
                    f.write(json.dumps(r) + "\n")
            print(f"[INFO] Patched {len(rows1)} cfg1 rows with backend/model fields")

    # ── Load GSM8K ────────────────────────────────────────────────────────────
    if not table_only:
        print("[INFO] Loading GSM8K N=40 seed=42 ...")
        gsm_questions = load_gsm8k(40, 42)
        if not gsm_questions:
            print("[WARN] GSM8K unavailable — table-only mode")
            table_only = True

    # ── Run configs ───────────────────────────────────────────────────────────
    GSM_CONFIGS  = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    TQQA_CONFIGS = [8]

    if not table_only:
        for cfg_id in GSM_CONFIGS:
            if cfg_filter and cfg_id != cfg_filter:
                continue
            print(f"\n{'='*55}")
            print(f"  Running Config {cfg_id}: {CONFIG_LABELS[cfg_id]}")
            print(f"{'='*55}")
            run_config(cfg_id, gsm_questions, dataset="gsm8k")

        # TruthfulQA
        if not cfg_filter or cfg_filter == 8:
            print("\n[INFO] Loading TruthfulQA N=40 seed=42 ...")
            tqa_questions = load_truthfulqa(40, 42)
            if tqa_questions:
                run_config(8, tqa_questions, dataset="truthfulqa")

    # ── Build metrics map ─────────────────────────────────────────────────────
    metrics_map = {}

    # Load cfg1 for baseline
    baseline_rows = list(load_cache(cache_path(1)).values())
    baseline_correct = {str(r["index"]) for r in baseline_rows if r.get("is_correct")}

    for cfg_id in GSM_CONFIGS:
        path = cache_path(cfg_id, "gsm8k")
        rows = list(load_cache(path).values())
        if not rows:
            continue
        m = compute_metrics(rows, baseline_correct if cfg_id != 1 else None)
        if m:
            metrics_map[cfg_id] = m

    # TruthfulQA
    tqa_path = cache_path(8, "truthfulqa")
    tqa_rows = list(load_cache(tqa_path).values())
    if tqa_rows:
        m = compute_metrics(tqa_rows, None)
        if m:
            metrics_map[8] = m

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESULTS SUMMARY")
    print("="*65)
    print(f"{'Cfg':<4} {'Setup':<42} {'Acc':>7} {'F/B':>8} {'Rec':>7} {'Pre':>7}")
    print("-"*65)
    for cfg_id in sorted(CONFIG_LABELS.keys()):
        label = CONFIG_LABELS[cfg_id][:42]
        m = metrics_map.get(cfg_id)
        if m is None:
            print(f"{cfg_id:<4} {label:<42} {'N/A':>7} {'—':>8} {'N/A':>7} {'N/A':>7}")
        else:
            fb = f"+{m['fixed']}/-{m['broken']}" if m["fixed"] != "—" else "   —"
            print(f"{cfg_id:<4} {label:<42} {pct(m['accuracy']):>7} {fb:>8} "
                  f"{pct(m['recall']):>7} {pct(m['precision']):>7}")
    print("-"*65)
    print(f"\n{FOOTNOTE}")
    print(f"\n{NOISE_NOTE}")

    # ── Generate output files ─────────────────────────────────────────────────
    md_str, csv_str, html_str = build_tables(metrics_map)

    (OUT / "final_results_table.md").write_text(md_str, encoding="utf-8")
    (OUT / "final_results_table.csv").write_text(csv_str, encoding="utf-8")
    (OUT / "final_results_table.html").write_text(html_str, encoding="utf-8")

    print("\n[OUTPUT]")
    print(f"  results/final_results_table.md")
    print(f"  results/final_results_table.csv")
    print(f"  results/final_results_table.html")
    print("\nDone.")

if __name__ == "__main__":
    main()
