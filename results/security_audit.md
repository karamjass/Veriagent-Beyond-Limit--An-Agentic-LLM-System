# VeriAgent Security Audit — Pre-Public Demo Assessment

**Scope**: Security posture review of `app.py`, `model_core.py`, `prompts.py`, and infrastructure for a small public demo via a secure tunnel (e.g., Cloudflare Tunnel, ngrok).  
**Security Classification**: **Basic Controls** (Targeted protections against accidental exposure and opportunistic abuse; *not penetration-tested or certified for high-threat environments*).  
**Operating Principle**: 100% local Ollama inference; zero external/cloud LLM APIs; strict local isolation; fail-closed defaults.

---

## 1. Route-by-Route Security & Auth Inventory

| HTTP Method | Route Endpoint | Current Auth Required? | User Inputs Accepted | Return Payload & Sensitivity |
|:---:|:---|:---:|:---|:---|
| `GET` | `/` | **No** (Public) | None | Serves `index.html` (Static SPA interface). |
| `GET` | `/<path:filename>` | **No** (Public) | URL path `filename` | Serves static assets (`.js`, `.css`, `.png`). Blocks `.py`, `.db`, `.env`, `.git`, but exposes root directory non-code files. |
| `GET` | `/api/health` | **No** (Public) | None | System status, database engine, model name, and connection status. *May leak internal socket/URL errors*. |
| `GET` | `/api/auth/config` | **No** (Public) | None | Public authentication parameters (`auth_mode`, `supabase_url`, `supabase_anon_key`). |
| `POST`| `/api/auth/session` | **No** (Login Gate) | JSON: `access_token`, `email`, `display_name` | Issues signed session cookie + CSRF token. *In local dev, relies on `request.remote_addr` check which can be spoofed behind reverse tunnels*. |
| `POST`| `/api/auth/logout` | **No** (Public) | None | Clears session cookie; returns status `logged_out`. |
| `GET` | `/api/auth/user` | **Yes** (Cookie / Bearer) | Ambient session cookie / Bearer token | Profile data (`id`, `email`, `display_name`, `avatar_url`) + fresh CSRF token. |
| `GET` | `/api/conversations` | **Yes** (Cookie / Bearer) | None | JSON array of user's past 50 conversations. |
| `POST`| `/api/conversations` | **Yes** (CSRF Required) | JSON: `title` (max 90 chars) | Created conversation record. |
| `GET` | `/api/conversations/<id>` | **Yes** (Cookie / Bearer) | URL parameter `id` | Full conversation history with messages. Scoped to user ID. |
| `DELETE`| `/api/conversations/<id>` | **Yes** (CSRF Required) | URL parameter `id` | Confirmation of deletion. Scoped to user ID. |
| `POST`| `/api/conversations/import`| **Yes** (CSRF Required) | JSON: `conversations` list (up to 20 items) | Imported count confirmation. |
| `POST`| `/api/chat` | **Yes** (CSRF Required) | JSON: `question` (max 4000 chars), `conversation_id` | Full model response, reasoning trace, verification label, elapsed time. *High compute cost*. |
| `GET` | `/api/benchmarks` | **Yes** (Cookie / Bearer) | None | Historical benchmark records for caller. |
| `POST`| `/api/benchmarks/run` | **Yes** (CSRF Required) | JSON: `gsm8k_limit`, `truthfulqa_limit`, `seed` | Benchmark pilot execution results. *Extremely high compute cost*. |

---

## 2. Vulnerability Findings Ranked by Severity

### High Severity Findings

#### H1: Absence of Rate Limiting & Abuse Throttling
- **Location**: `app.py` (all endpoints, specifically `/api/chat` and `/api/benchmarks/run`).
- **Risk**: An unauthenticated or single authenticated client can flood the server with inference requests, monopolizing the GPU/CPU, saturating the Ollama queue, and causing Denial of Service (DoS) for all demo participants.
- **Remediation**: Integrate `Flask-Limiter` with client IP and user ID keying (default: 10 requests/minute for chat; 2 requests/minute for benchmark runs), returning clear HTTP 429 status codes.
- **Implementation Note & Scope**: Rate limiting is strictly **in-memory** and **single-worker** (`memory://` backend). It requires no external Redis instance, making it lightweight for demo deployment, but limits do not synchronize across multi-process clusters.

#### H2: Tunnel Authentication Bypass via `remote_addr`
- **Location**: `app.py` lines 742–744 (`auth_session`).
- **Risk**: When exposed over Cloudflare Tunnel or ngrok, the reverse proxy connection may report `request.remote_addr` as `127.0.0.1` unless `ProxyFix` is configured. Any internet visitor could automatically obtain a valid session without entering any credentials.
- **Remediation**: Introduce a mandatory `DEMO_ACCESS_CODE` environment variable for demo mode. In demo/public mode (`VERIAGENT_ENV=demo` or `VERIAGENT_ENV=public` or when `DEMO_ACCESS_CODE` is set), **completely disable `remote_addr` auto-login** and always require `DEMO_ACCESS_CODE`. Parse `X-Forwarded-For` only when `TRUSTED_PROXY=1` is explicitly enabled. A dedicated test will simulate a tunnel request arriving with `remote_addr=127.0.0.1` and no code, confirming it is strictly rejected with HTTP 401.

#### H3: Unbounded Concurrent Generations & VRAM Crash Risk
- **Location**: `app.py` (`/api/chat`), `model_core.py` (`_model_lock`).
- **Risk**: While `_model_lock` serializes Ollama inference calls, incoming HTTP requests spawn concurrent Flask threads that accumulate in memory, queueing up unbounded workloads that can exceed system memory and time out tunnels.
- **Remediation**: Implement an active generation semaphore/counter with a strict concurrency cap (e.g., max 2 concurrent generations), immediately rejecting excess requests with HTTP 503 (`"Server busy processing existing requests; please retry in a moment"`).

#### H4: Missing `.gitignore` & Risk of Credential Leakage
- **Location**: Repository root.
- **Risk**: There is currently no `.gitignore` file in the workspace. Any local `.env` file containing `FLASK_SECRET_KEY`, `DEMO_ACCESS_CODE`, or database credentials would be staged and committed by default.
- **Remediation**: Add `.gitignore` explicitly ignoring `.env`, `instance/`, `*.db`, `__pycache__/`, and `.secret_key`. Create `.env.example` with non-sensitive placeholder templates.

---

### Medium Severity Findings

#### M1: Absence of Immediate Emergency Kill Switch
- **Location**: `app.py` (`/api/chat`, `/api/benchmarks/run`).
- **Risk**: In the event of an active prompt-injection attack or resource exhaustion during a live demo, the operator has no mechanism to halt generation without killing the underlying process.
- **Remediation**: Add `ADMIN_KILL_SWITCH=1` environment variable support that immediately causes all inference routes to return HTTP 503 (`"Service temporarily paused by administrator"`).

#### M2: Verbose Error Responses Revealing Internal Architecture
- **Location**: `app.py` lines 1010–1021 (`mc.ModelUnavailable`), 1106–1107 (`Benchmark evaluation error`).
- **Risk**: Uncaught exceptions and `ModelUnavailable` errors echo internal connection URLs (e.g., `http://127.0.0.1:11434`) and raw socket error messages directly to the client JSON response.
- **Remediation**: Intercept all exceptions in route handlers, log full stack traces server-side, and return sanitized, generic error messages to clients.

#### M3: Overly Broad Static File Route
- **Location**: `app.py` lines 600–618 (`static_files`).
- **Risk**: `send_from_directory(BASE_DIR, filename)` serves files from the repository root. While blocking `.py`, `.db`, and `.env`, it allows arbitrary reading of `.md`, `.json`, `.csv`, `.html`, and `.txt` files across subdirectories (e.g., `results/`, `data/`).
- **Remediation**: Restrict static file serving strictly to an allowlisted root asset set (`styles.css`, `app.js`, `favicon.ico`, `robots.txt`) and dedicated static assets inside `static/` with safe file extensions. Explicitly forbid serving `results/`, `data/`, documentation (`*.md`), test files, and repository root files, returning HTTP 403 Forbidden.

#### M4: Missing Clickjacking & Content-Security-Policy (CSP) Headers
- **Location**: `app.py` lines 63–71 (`add_security_headers`).
- **Risk**: Current response headers lack `X-Frame-Options` and `Content-Security-Policy`. An attacker could embed the demo interface in a malicious `<iframe>` to execute clickjacking attacks.
- **Remediation**: Add `X-Frame-Options: DENY`, `Content-Security-Policy` with `default-src 'self'`, `frame-ancestors 'none'`, and strict CORS headers.
- **Known Limitation**: `'unsafe-inline'` is maintained in `script-src` and `style-src` as an explicit, documented trade-off to permit the existing single-page application's inline UI event handlers, SVGs, and dynamic theme styles without requiring a full frontend refactor to nonces or build bundling.

#### M5: Unchecked External Ollama Daemon Binding
- **Location**: `model_core.py` lines 20, 44–76 (`_ollama_request`).
- **Risk**: If the host environment binds Ollama to `0.0.0.0` or a public interface, external actors could bypass Flask entirely and interact with the unauthenticated Ollama API directly on port 11434.
- **Remediation**: Add a startup check in `app.py` and `model_core.py` verifying that `OLLAMA_BASE_URL` resolves strictly to loopback addresses (`127.0.0.1` or `localhost`), logging a loud critical warning if bound to a non-local address.

---

### Low Severity Findings

#### L1: Cookie `Secure` Flag Conditional on `request.is_secure`
- **Location**: `app.py` line 785.
- **Risk**: When operating behind a TLS-terminating tunnel without `ProxyFix`, `request.is_secure` evaluates to `False`, issuing cookies without the `Secure` flag over what is externally an HTTPS connection.
- **Remediation**: Enforce `Secure=True` whenever running in production or demo mode, or when `X-Forwarded-Proto: https` is forwarded.

#### L2: Missing Privacy Notice on Data Processing
- **Location**: `index.html` (UI chat interface).
- **Risk**: Users joining a public demo may inadvertently submit private or sensitive personal information without realizing queries are processed and logged.
- **Remediation**: Add a prominent, concise privacy disclaimer banner in the UI stating that queries are processed locally and users should not submit confidential data.

#### L3: Unstructured Logging of Client Requests
- **Location**: `app.py` (request handling).
- **Risk**: Flask default logs lack structured attribution (e.g., hashed client IP, route, status code, latency) while running the risk of logging full prompt bodies if debug logging is enabled.
- **Remediation**: Implement a lightweight request logging hook recording timestamp, pseudonymized client hash, route, HTTP status, and elapsed time, while strictly suppressing request payloads and credentials (`LOG_PROMPTS=0`).

---

## 3. Known Limitations & Scope of Basic Controls

1. **Basic Controls Classification**: These security controls are intended for controlled demonstration over reverse tunnels (Cloudflare Tunnel, ngrok) to prevent opportunistic abuse, denial-of-service, and inadvertent leakage. The system is **not** penetration-tested, hardened for multi-tenant SaaS, or certified against state-sponsored adversarial attacks.
2. **In-Memory Rate Limiting**: The rate limiter is backed by `memory://` storage. Limits are enforced per-process in a single worker. Multiple workers or containerized replicas would each maintain separate memory stores; Redis would be required for clustered deployments.
3. **CSP `'unsafe-inline'`**: Inline styles and scripts are allowed to maintain compatibility with `index.html` inline UI interactions. While clickjacking (`frame-ancestors 'none'`) and foreign origin scripts (`default-src 'self'`) are blocked, nonced or hashed CSP scripts are left for future production redesign.
4. **Local Host Loopback Assumption**: VeriAgent relies on Ollama running locally at `127.0.0.1:11434`. It defends the HTTP boundary but assumes the host OS running Python and Ollama is a trusted environment.

---

## 4. Recommended Remediation & File List

| File Path | Planned Modifications |
|:---|:---|
| [`app.py`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/app.py) | Add `DEMO_ACCESS_CODE` auth gate, `ADMIN_KILL_SWITCH`, `Flask-Limiter` rate limiting, concurrency cap (semaphore), hardened security headers (CSP, X-Frame-Options), sanitized error responses, and structured audit logging. |
| [`model_core.py`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/model_core.py) | Add Ollama non-local host startup check, enforce server-side `num_predict` and context bounds, and expand `SafeMathEvaluator` AST defenses. |
| [`.env.example`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/.env.example) | Template with placeholder environment variables (`DEMO_ACCESS_CODE`, `FLASK_SECRET_KEY`, `ADMIN_KILL_SWITCH`, `RATE_LIMIT_PER_MINUTE`). |
| [`.gitignore`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/.gitignore) | Explicitly ignore `.env`, `instance/`, `*.db`, `__pycache__/`, `.secret_key`, and temporary run files. |
| [`requirements.txt`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/requirements.txt) | Add `Flask-Limiter>=3.5.0` for in-memory rate limiting. |
| [`index.html`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/index.html) | Add public demo access code prompt modal and local processing privacy notice banner. |
| [`tests/test_security.py`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/tests/test_security.py) | New automated test suite covering auth rejection, rate limiting (429), prompt limits (400), kill switch (503), security headers, secret leak prevention, and AST injection defenses. |
| [`results/security_injection_tests.md`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/results/security_injection_tests.md) | Audit report of 10+ prompt injection and system-prompt extraction attacks against the local model. |
| [`results/security_scan.md`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/results/security_scan.md) | Results of `bandit -r .` and `pip-audit` scans with detailed risk analysis. |
| [`README.md`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/README.md) & [`CHANGELOG.md`](file:///c:/Users/Karam/OneDrive/Desktop/beyond_limit(veriagent)/CHANGELOG.md) | Document basic security controls, demo configuration instructions, and security notes. |

