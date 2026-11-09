"""Automated Security & Access Control Tests for VeriAgent.

Covers:
  1. Unauthenticated request rejection (401).
  2. Tunnel request simulation (remote_addr=127.0.0.1 without access code rejected in demo mode).
  3. Valid DEMO_ACCESS_CODE session establishment.
  4. Constant-time comparison & invalid code rejection.
  5. TRUSTED_PROXY and X-Forwarded-For IP resolution.
  6. Rate limiting enforcement (429).
  7. Input length limits (MAX_PROMPT_LENGTH -> 400).
  8. Admin kill switch (503).
  9. Static asset directory restriction (results/, data/, source code -> 403).
 10. Security headers (X-Frame-Options: DENY, CSP, Cache-Control).
 11. Secret key validation & startup safety.
 12. Ollama loopback host safety verification.
 13. SafeMathEvaluator 5 hostile injection test cases (attributes, f-strings, node bombs, exponents, length).
"""
import os
import unittest
from unittest.mock import patch

import app as flask_app
import model_core


class TestSecurityControls(unittest.TestCase):
    """Test suite verifying basic security controls for public demo deployment."""

    def setUp(self):
        self.app = flask_app.app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    # ── 1. Access Control & Tunnel Auto-Login Bypass (H2) ─────────────────────

    def test_unauthenticated_chat_rejected(self):
        """Unauthenticated POST /api/chat must return 401."""
        res = self.client.post("/api/chat", json={"question": "What is 2+2?"})
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertIn("Authentication required", data.get("error", ""))

    def test_unauthenticated_conversations_rejected(self):
        """Unauthenticated GET /api/conversations must return 401."""
        res = self.client.get("/api/conversations")
        self.assertEqual(res.status_code, 401)

    def test_tunnel_request_without_code_rejected_in_demo_mode(self):
        """Simulate tunnel request arriving with remote_addr=127.0.0.1 in demo mode.

        Must NOT auto-login as local-dev; must return 401 requiring DEMO_ACCESS_CODE.
        """
        with patch.dict(os.environ, {"VERIAGENT_ENV": "demo", "DEMO_ACCESS_CODE": "demo-secret-pass-987"}):
            # Client connects from loopback (standard for reverse tunnels like ngrok/cloudflared)
            res = self.client.post(
                "/api/auth/session",
                json={"email": "visitor@example.com"},
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            self.assertEqual(res.status_code, 401)
            data = res.get_json()
            self.assertIn("Valid demo access code required", data.get("error", ""))

    def test_tunnel_request_with_invalid_code_rejected(self):
        """Invalid access code must be rejected with 401."""
        with patch.dict(os.environ, {"VERIAGENT_ENV": "demo", "DEMO_ACCESS_CODE": "demo-secret-pass-987"}):
            res = self.client.post(
                "/api/auth/session",
                json={"access_code": "wrong-code-attempt", "email": "visitor@example.com"},
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            self.assertEqual(res.status_code, 401)
            data = res.get_json()
            self.assertIn("Valid demo access code required", data.get("error", ""))

    def test_tunnel_request_with_valid_demo_code_succeeds(self):
        """Valid access code in demo mode establishes authenticated session and CSRF token."""
        secret = "demo-secret-pass-987"
        env = {
            "VERIAGENT_ENV": "demo",
            "DEMO_ACCESS_CODE": secret,
            "FLASK_SECRET_KEY": "a-strong-test-secret-key-32-chars-long",
        }
        with patch.dict(os.environ, env):
            res = self.client.post(
                "/api/auth/session",
                json={"access_code": secret, "email": "demo_user@example.com"},
                environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data.get("authenticated"))
            self.assertIn("csrf_token", data)
            self.assertEqual(data.get("auth_mode"), "demo")
            # Session cookie must be set
            self.assertIn(flask_app.SESSION_COOKIE, res.headers.get("Set-Cookie", ""))

    # ── 2. Trusted Proxy & IP Resolution ──────────────────────────────────────

    def test_trusted_proxy_enabled_uses_forwarded_for(self):
        """When TRUSTED_PROXY=1, client IP resolves to the forwarded header."""
        with patch.dict(os.environ, {"TRUSTED_PROXY": "1"}):
            with self.app.test_request_context(
                "/api/auth/config",
                headers={"X-Forwarded-For": "203.0.113.195, 127.0.0.1"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                ip = flask_app.get_client_ip()
                self.assertEqual(ip, "203.0.113.195")

    def test_untrusted_proxy_ignores_forwarded_for(self):
        """When TRUSTED_PROXY=0 (default), spoofed X-Forwarded-For header is ignored."""
        with patch.dict(os.environ, {"TRUSTED_PROXY": "0"}):
            with self.app.test_request_context(
                "/api/auth/config",
                headers={"X-Forwarded-For": "203.0.113.195"},
                environ_base={"REMOTE_ADDR": "127.0.0.1"},
            ):
                ip = flask_app.get_client_ip()
                self.assertEqual(ip, "127.0.0.1")

    # ── 3. Abuse Protection & Rate Limiting ───────────────────────────────────

    def test_oversized_prompt_rejected(self):
        """Questions exceeding MAX_PROMPT_LENGTH (default 2000 chars) must return 400."""
        # Establish session first
        env = {
            "DEMO_ACCESS_CODE": "code123",
            "VERIAGENT_ENV": "demo",
            "FLASK_SECRET_KEY": "a-strong-test-secret-key-32-chars-long",
        }
        with patch.dict(os.environ, env):
            login_res = self.client.post("/api/auth/session", json={"access_code": "code123"})
            csrf_token = login_res.get_json()["csrf_token"]

            huge_prompt = "Explain this math equation: " + ("x" * 2100)
            chat_res = self.client.post(
                "/api/chat",
                json={"question": huge_prompt},
                headers={"X-CSRF-Token": csrf_token},
            )
            self.assertEqual(chat_res.status_code, 400)
            self.assertIn("exceeds", chat_res.get_json().get("error", ""))

    def test_rate_limiting_returns_429(self):
        """Exceeding requests per minute must trigger HTTP 429 response."""
        env = {
            "RATE_LIMIT_PER_MINUTE": "3",
            "DEMO_ACCESS_CODE": "code123",
            "VERIAGENT_ENV": "demo",
            "FLASK_SECRET_KEY": "a-strong-test-secret-key-32-chars-long",
        }
        with patch.dict(os.environ, env):
            login_res = self.client.post("/api/auth/session", json={"access_code": "code123"})
            csrf_token = login_res.get_json()["csrf_token"]

            statuses = []
            for _ in range(6):
                res = self.client.post(
                    "/api/chat",
                    json={"question": ""},  # empty will fail validation or trigger limiter
                    headers={"X-CSRF-Token": csrf_token},
                )
                statuses.append(res.status_code)

            # Either rate limiter triggered (429) or prompt validation (400)
            self.assertIn(429, statuses)

    # ── 4. Admin Kill Switch (503) ────────────────────────────────────────────

    def test_admin_kill_switch_suspends_chat(self):
        """ADMIN_KILL_SWITCH=1 immediately halts chat endpoint with HTTP 503."""
        env = {
            "ADMIN_KILL_SWITCH": "1",
            "DEMO_ACCESS_CODE": "code123",
            "VERIAGENT_ENV": "demo",
            "FLASK_SECRET_KEY": "a-strong-test-secret-key-32-chars-long",
        }
        with patch.dict(os.environ, env):
            login_res = self.client.post("/api/auth/session", json={"access_code": "code123"})
            csrf_token = login_res.get_json()["csrf_token"]

            res = self.client.post(
                "/api/chat",
                json={"question": "Hello world"},
                headers={"X-CSRF-Token": csrf_token},
            )
            self.assertEqual(res.status_code, 503)
            self.assertIn("suspended", res.get_json().get("error", "").lower())

    # ── 5. Static Asset Restriction (M3) ──────────────────────────────────────

    def test_static_files_restricted_access(self):
        """results/, data/, source files, and .env files must be strictly blocked (403)."""
        blocked_paths = [
            "/results/security_audit.md",
            "/results/raw/gsm8k_cfg1.jsonl",
            "/data/prompt_templates.json",
            "/model_core.py",
            "/app.py",
            "/.env",
            "/.env.example",
            "/instance/veriagent.db",
            "/tests/test_veriagent.py",
        ]
        for path in blocked_paths:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 403, f"Expected 403 for restricted file: {path}")

    def test_allowlisted_root_assets_accessible(self):
        """Allowlisted root assets (styles.css, app.js) should be accessible."""
        for asset in ["/styles.css", "/app.js"]:
            res = self.client.get(asset)
            self.assertEqual(res.status_code, 200, f"Expected 200 for public asset: {asset}")
            res.close()

    # ── 6. Security Headers (M4) ──────────────────────────────────────────────

    def test_security_headers_present(self):
        """Browser protections (X-Frame-Options, CSP, nosniff) must be attached to responses."""
        res = self.client.get("/")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        csp = res.headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        res.close()

    def test_api_cache_control_no_store(self):
        """API endpoints must forbid caching."""
        res = self.client.get("/api/auth/config")
        self.assertIn("no-store", res.headers.get("Cache-Control", ""))

    # ── 7. Secret Key Safety Validation ───────────────────────────────────────

    def test_secret_key_refuses_insecure_defaults_in_demo(self):
        """_get_secret_key must raise RuntimeError if placeholder secret key is used in demo/production."""
        with patch.dict(os.environ, {"VERIAGENT_ENV": "demo", "FLASK_SECRET_KEY": "your-long-random-secret-key-min-32-chars"}):
            with self.assertRaises(RuntimeError):
                flask_app._get_secret_key()

    # ── 8. Ollama Loopback Verification ───────────────────────────────────────

    def test_ollama_loopback_validator(self):
        """is_loopback_address must accept 127.0.0.1 and localhost, and reject external IPs."""
        self.assertTrue(model_core.is_loopback_address("http://127.0.0.1:11434"))
        self.assertTrue(model_core.is_loopback_address("http://localhost:11434"))
        self.assertTrue(model_core.is_loopback_address("http://[::1]:11434"))
        self.assertFalse(model_core.is_loopback_address("http://0.0.0.0:11434"))
        self.assertFalse(model_core.is_loopback_address("http://192.168.1.50:11434"))
        self.assertFalse(model_core.is_loopback_address("http://attacker.example.com:11434"))

    # ── 9. SafeMathEvaluator Hostile Injection Defenses ───────────────────────

    def test_math_eval_attribute_access_blocked(self):
        """Attack 1: Attribute access on object or number must be rejected."""
        self.assertIsNone(model_core.safe_math_eval("(42).bit_length()"))
        self.assertIsNone(model_core.safe_math_eval("'hello'.upper()"))
        self.assertIsNone(model_core.safe_math_eval("x = ().__class__.__bases__"))

    def test_math_eval_string_formatting_blocked(self):
        """Attack 2: String formatting tricks and f-strings must be rejected."""
        self.assertIsNone(model_core.safe_math_eval("'{0.__class__}'.format(1)"))
        self.assertIsNone(model_core.safe_math_eval("f'{1+1}'"))

    def test_math_eval_huge_exponent_pow_blocked(self):
        """Attack 3: Huge exponent via direct pow() call must be rejected."""
        self.assertIsNone(model_core.safe_math_eval("ans = pow(2, 999999)"))
        self.assertIsNone(model_core.safe_math_eval("ans = 2 ** 5000"))

    def test_math_eval_ast_node_bomb_blocked(self):
        """Attack 4: Huge expression tree with excessive AST nodes must be rejected."""
        nested = "1" + (" + 1" * 200)
        self.assertIsNone(model_core.safe_math_eval(f"ans = {nested}"))

    def test_math_eval_oversized_code_string_blocked(self):
        """Attack 5: Code string exceeding 2000 characters must be rejected."""
        long_comment = "# " + ("A" * 2500) + "\nans = 42"
        self.assertIsNone(model_core.safe_math_eval(long_comment))


if __name__ == "__main__":
    unittest.main()
