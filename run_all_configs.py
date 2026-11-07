"""
run_all_configs.py
Run configs 3-13 sequentially (resume from cache).
Safe to Ctrl+C and restart — it resumes where it left off.
"""
import subprocess, sys, os

PYTHON = r".venv\Scripts\python.exe"
CONFIGS = [3, 4, 5, 6, 9, 10, 11, 12, 13]  # 7=llama3 (optional), 8=TruthfulQA

for cfg in CONFIGS:
    print(f"\n{'='*50}")
    print(f"  STARTING CONFIG {cfg}")
    print(f"{'='*50}")
    ret = subprocess.call([PYTHON, "build_results_table.py", "--cfg", str(cfg)])
    if ret != 0:
        print(f"[WARN] Config {cfg} exited with code {ret} — continuing...")

# Config 7: LLaMA-3 (needs llama3:latest model)
print("\n" + "="*50)
print("  CONFIG 7: LLaMA-3 8B (needs llama3:latest)")
print("="*50)
ret7 = subprocess.call([PYTHON, "build_results_table.py", "--cfg", "7"])

# Config 8: TruthfulQA
print("\n" + "="*50)
print("  CONFIG 8: TruthfulQA")
print("="*50)
ret8 = subprocess.call([PYTHON, "build_results_table.py", "--cfg", "8"])

# Final table
print("\n" + "="*50)
print("  GENERATING FINAL TABLE")
print("="*50)
subprocess.call([PYTHON, "build_results_table.py", "--table"])

