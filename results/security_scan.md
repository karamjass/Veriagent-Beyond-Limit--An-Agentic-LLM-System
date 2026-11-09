# VeriAgent Automated Security Scan Report

**Scan Date**: November 9, 2026  
**Tools**: Bandit 1.9.4 (Static AST Code Analysis), pip-audit 2.10.1 (PyPI Vulnerability Database)  
**Scope**: `app.py`, `model_core.py`, `prompts.py`, `tests/`, and Python virtual environment dependencies (`.venv`)  
**Security Posture**: **Basic Controls** (Targeted defenses for demonstration via reverse tunnels; not penetration-tested)

---

## 1. Executive Summary

| Category | High Severity | Medium Severity | Low Severity | Total | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Code Vulnerabilities (Bandit)** | **0** | **2** | **10** | 12 | **Passed** (No Highs; Mediums verified as low-risk/false positive) |
| **Dependency Vulnerabilities (pip-audit)** | 0 | 1 | 2 (packages) | 3 | **Reviewed** (Runtime core packages 0 vulns; non-exploitable transitive deps) |

---

## 2. Static AST Code Analysis (Bandit)

Bandit scanned 3,926 lines of Python code across `app.py`, `model_core.py`, `prompts.py`, and `tests/`.

### Findings Breakdown

#### A. High Severity Findings: 0
No hard-coded credentials, unvalidated raw `eval`/`exec`, shell command injections (`subprocess.call(shell=True)`), or unsafe deserializations were detected in application logic.

#### B. Medium Severity Findings: 2

1. **`B310: urllib.request.urlopen` audit warning**
   - **Location**: [`model_core.py:79`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/model_core.py#L79)
   - **Description**: Bandit flags `urllib.request.urlopen` because unvalidated URLs could theoretically accept `file://` or custom URI schemes.
   - **Triage & Justification**: All URLs passed to `_ollama_request` are strictly constructed using the server-controlled `OLLAMA_BASE_URL` prefixed to hardcoded API paths (e.g. `/api/chat`, `/api/tags`). Furthermore, `check_ollama_host_safety()` enforces loopback host validation (`127.0.0.1`, `localhost`). External attacker-supplied URLs are never parsed by `urlopen`.
   - **Risk Assessment**: **Low / Controlled**.

2. **`B608: Hardcoded SQL expressions (Heuristic False Positive)`**
   - **Location**: [`model_core.py:1390`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/model_core.py#L1390)
   - **Description**: Bandit's heuristic flags a string containing `"Select the most accurate answer..."` as a potential SQL injection vector.
   - **Triage & Justification**: This is an English instructional prompt sent to the LLM generator for TruthfulQA multiple-choice evaluation, not an SQL query. Database persistence in `app.py` exclusively uses SQLAlchemy ORM parameter binding; no raw SQL string concatenation is used.
   - **Risk Assessment**: **False Positive**.

#### C. Low Severity Findings: 10
- **`B110: Try, Except, Pass`** (2 occurrences): Used in JSON cache parsing fallbacks (`app.py:433`).
- **`B105: Hardcoded password string`** (6 occurrences): Located strictly inside `tests/test_security.py` where placeholder test strings (e.g. `"a-strong-test-secret-key-32-chars-long"`, `"your-long-random-secret-key-min-32-chars"`) are used to verify password validation logic. No real credentials exist in source code.

---

## 3. Dependency Vulnerability Scan (pip-audit)

The virtual environment was audited against the Open Source Vulnerability (OSV) and PyPI advisory databases.

### Core Web & Reasoning Packages: CLEAN (0 Vulnerabilities)
The following primary runtime packages have **zero known vulnerabilities**:
- `Flask` (3.1.x)
- `Flask-Limiter` (4.1.x)
- `SQLAlchemy` (2.0.x)
- `Werkzeug` (3.1.x)
- `requests` (2.32.x)
- `urllib3` (2.8.x)
- `sympy` (1.14.x)
- `cryptography` (50.0.x)

### Flagged Transitive / Build Packages

| Package | Installed Version | Flagged Advisory | Upstream Fix | Triage & Context |
|:---|:---:|:---|:---:|:---|
| `filelock` | 4.1.0 | CVE-2026-112906 (CVSS 4.7 Medium) | 4.2.1 | Transitive dependency of HuggingFace `datasets`. Involves local race condition in shared `/tmp` file lock eviction between local system users. VeriAgent runs in single-user local deployment; this race is not exploitable over HTTP/tunnels. |
| `jinja2` | 3.1.5 | Legacy Template Sandbox CVEs | 3.1.6+ | VeriAgent does not use server-side Jinja templates for dynamic user-supplied templates. The UI is a static SPA (`index.html`) interacting via JSON APIs. |
| `setuptools` | 65.5.0 | Legacy Wheel Namespace Traversal | 84.1.0+ | Build-time packaging utility; not loaded during runtime request handling. |

---

## 4. Security Verification Summary

1. **Zero High-Severity Findings**: No critical code injection, privilege escalation, or hardcoded credentials detected.
2. **Runtime Isolation**: The application binds strictly to `127.0.0.1`. Inference requires a locally hosted Ollama server; no third-party cloud APIs or external socket dependencies are invoked during standard operation.
3. **Defense-in-Depth**:
   - `DEMO_ACCESS_CODE` with constant-time verification (`hmac.compare_digest`).
   - In-memory rate limiting via `Flask-Limiter` (single-worker fixed-window).
   - Strict static asset allowlisting blocking `results/`, `data/`, source files, and secrets.
   - Comprehensive security headers (`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Content-Security-Policy`).

