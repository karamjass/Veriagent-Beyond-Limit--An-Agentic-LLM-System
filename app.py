"""Flask entry point for VeriAgent — Beyond Limits.

Provides:
  - Database persistence via SQLAlchemy:
      * Local dev: SQLite at instance/veriagent.db
      * Production: PostgreSQL via DATABASE_URL with pre-ping and connection pooling
  - Supabase Auth integration:
      * Production users authenticate via verified Supabase access tokens.
      * Local dev single-user mode bound to localhost when Supabase is not configured.
  - Endpoints:
      * POST /api/chat                  — runs reasoning pipeline & persists messages
      * GET  /api/health                — dependency, database, and model status
      * GET  /api/auth/config           — public auth configuration for client
      * POST /api/auth/session          — verifies token / establishes session
      * POST /api/auth/logout           — ends session
      * GET  /api/auth/user             — returns current authenticated user
      * GET  /api/conversations         — lists user's conversations
      * POST /api/conversations         — creates a new conversation
      * GET  /api/conversations/<id>    — retrieves conversation with messages
      * DELETE /api/conversations/<id> — deletes user's conversation
      * POST /api/conversations/import — imports browser-local conversations
      * GET  /api/benchmarks            — lists benchmark runs
      * POST /api/benchmarks/run        — executes/records a benchmark run
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, make_response, request, send_from_directory
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

BASE_DIR = Path(__file__).parent.resolve()

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

@app.after_request
def add_security_headers(response: Response) -> Response:
    """Apply low-risk browser protections and prevent caching of account/chat API data."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Environment & Production Configuration ────────────────────────────────────

IS_PRODUCTION = (
    os.environ.get("VERIAGENT_ENV", "").strip().lower() == "production"
    or os.environ.get("APP_ENV", "").strip().lower() == "production"
    or os.environ.get("FLASK_ENV", "").strip().lower() == "production"
    or os.environ.get("PRODUCTION", "").strip().lower() in ("1", "true", "yes")
)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "").strip()

# Production must fail closed if required environment configuration is missing.
PRODUCTION_CONFIG_ERROR: str | None = None
if IS_PRODUCTION:
    missing: list[str] = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL (must point to PostgreSQL)")
    elif DATABASE_URL.startswith("sqlite"):
        missing.append("DATABASE_URL (SQLite is not permitted in production)")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing.append("SUPABASE_ANON_KEY")
    if not FLASK_SECRET_KEY:
        missing.append("FLASK_SECRET_KEY")
    if missing:
        PRODUCTION_CONFIG_ERROR = (
            "Production configuration error: missing required environment variables: "
            + ", ".join(missing)
            + ". Single-user mode and unauthenticated fallbacks are strictly disabled in production."
        )


def _get_secret_key() -> str:
    """Return the secret key for session signing."""
    if FLASK_SECRET_KEY:
        return FLASK_SECRET_KEY
    if IS_PRODUCTION:
        raise RuntimeError("FLASK_SECRET_KEY is required in production.")
    # Local development persistent key
    instance_dir = BASE_DIR / "instance"
    instance_dir.mkdir(parents=True, exist_ok=True)
    secret_path = instance_dir / ".secret_key"
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    generated = secrets.token_hex(32)
    secret_path.write_text(generated, encoding="utf-8")
    return generated


# ── SQLAlchemy Models & Database Layer ────────────────────────────────────────

Base = declarative_base()


class User(Base):
    """User profile keyed by the verified Supabase Auth user ID (or local-dev-user)."""

    __tablename__ = "users"

    id = Column(String(128), primary_key=True)
    email = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=True)
    avatar_url = Column(String(1024), nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    benchmark_runs = relationship(
        "BenchmarkRun",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self) -> dict[str, Any]:
        first_name = _extract_first_name(self.display_name or "", self.email or "")
        return {
            "id": self.id,
            "email": self.email or "",
            "display_name": first_name,
            "first_name": first_name,
            "avatar_url": self.avatar_url or "",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def _extract_first_name(display_name: str, email: str) -> str:
    """Extract clean first name without digits, suffixes, or username numbers."""
    if display_name:
        parts = display_name.strip().split()
        if parts:
            cleaned = re.sub(r"[0-9_\-\.]+", "", parts[0]).strip()
            if cleaned:
                return cleaned.capitalize()
    if email and "@" in email:
        username = email.split("@")[0]
        cleaned = re.sub(r"[0-9_\-\.]+", "", username).strip()
        if cleaned:
            return cleaned.capitalize()
        return username.capitalize()
    return "User"


class Conversation(Base):
    """Conversation owned by one user."""

    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    def to_dict(self, include_messages: bool = False) -> dict[str, Any]:
        created_ts = (
            int(self.created_at.timestamp() * 1000)
            if self.created_at
            else int(time.time() * 1000)
        )
        updated_ts = (
            int(self.updated_at.timestamp() * 1000)
            if self.updated_at
            else int(time.time() * 1000)
        )
        data: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "createdAt": created_ts,
            "updatedAt": updated_ts,
            "messageCount": len(self.messages) if self.messages is not None else 0,
        }
        if include_messages:
            data["messages"] = [msg.to_dict() for msg in self.messages]
        return data


class Message(Base):
    """Message linked to a conversation, tracking verification status and metadata."""

    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    verification_label = Column(String(128), nullable=True)
    verified = Column(Boolean, default=False, nullable=False)
    elapsed = Column(Float, default=0.0, nullable=False)
    model = Column(String(128), nullable=True)
    review_rounds = Column(Integer, default=0, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self) -> dict[str, Any]:
        ts = (
            int(self.created_at.timestamp() * 1000)
            if self.created_at
            else int(time.time() * 1000)
        )
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "verification": self.verification_label or "Internal review only",
            "verified": bool(self.verified),
            "elapsed": float(self.elapsed or 0.0),
            "model": self.model or "",
            "review_rounds": int(self.review_rounds or 0),
            "timestamp": ts,
        }


class BenchmarkRun(Base):
    """Benchmark evaluation summary records."""

    __tablename__ = "benchmark_runs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    model = Column(String(128), nullable=False)
    model_revision = Column(String(128), nullable=True)
    seed = Column(Integer, nullable=False, default=42)
    gsm8k_accuracy = Column(Float, nullable=True)
    gsm8k_sample_size = Column(Integer, nullable=True)
    gsm8k_calculator_rate = Column(Float, nullable=True)
    truthfulqa_mc1_accuracy = Column(Float, nullable=True)
    truthfulqa_mc2_mass = Column(Float, nullable=True)
    truthfulqa_sample_size = Column(Integer, nullable=True)
    summary_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    user = relationship("User", back_populates="benchmark_runs")

    def to_dict(self) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        if self.summary_json:
            try:
                parsed = json.loads(self.summary_json)
            except Exception:
                pass
        return {
            "id": self.id,
            "model": self.model,
            "model_revision": self.model_revision,
            "seed": self.seed,
            "gsm8k": (
                {
                    "accuracy": self.gsm8k_accuracy,
                    "sample_size": self.gsm8k_sample_size,
                    "calculator_rate": self.gsm8k_calculator_rate,
                }
                if self.gsm8k_sample_size
                else None
            ),
            "truthfulqa": (
                {
                    "generation_choice_accuracy": self.truthfulqa_mc1_accuracy,
                    "sample_size": self.truthfulqa_sample_size,
                    "metric_note": "Generation-choice pilot, not the official MC1/MC2 likelihood metric.",
                }
                if self.truthfulqa_sample_size
                else None
            ),
            "summary": parsed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Database Engine Initialization ───────────────────────────────────────────

LOCAL_DEV_USER_ID = "local-dev-user"
_db_engine = None
_db_session_factory = None


def get_engine():
    """Create or return SQLAlchemy engine with SQLite-safe or PostgreSQL pooling settings."""
    global _db_engine, _db_session_factory
    if _db_engine is not None:
        return _db_engine

    if DATABASE_URL:
        db_uri = DATABASE_URL
        if db_uri.startswith("postgres://"):
            db_uri = "postgresql+psycopg://" + db_uri[len("postgres://") :]
        elif db_uri.startswith("postgresql://"):
            db_uri = "postgresql+psycopg://" + db_uri[len("postgresql://") :]
        engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
        )
    else:
        # Local SQLite
        instance_dir = BASE_DIR / "instance"
        instance_dir.mkdir(parents=True, exist_ok=True)
        sqlite_file = instance_dir / "veriagent.db"
        engine = create_engine(
            f"sqlite:///{sqlite_file.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    # Create tables if not present.
    # NOTE: Base.metadata.create_all initializes schema for development.
    # In production, schema changes should be managed using dedicated database migrations
    # (e.g. Alembic), not create_all alone.
    Base.metadata.create_all(bind=engine)

    _db_engine = engine
    _db_session_factory = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    return _db_engine


def get_db_session():
    """Return a scoped database session."""
    get_engine()
    return _db_session_factory()


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Remove scoped session at request end to avoid leaked connections."""
    global _db_session_factory
    if _db_session_factory is not None:
        _db_session_factory.remove()


def _ensure_local_dev_user(db) -> User:
    """Ensure the local-dev-user profile exists in local development mode."""
    user = db.query(User).filter_by(id=LOCAL_DEV_USER_ID).first()
    if not user:
        user = User(
            id=LOCAL_DEV_USER_ID,
            email="developer@localhost",
            display_name="Karamjass Kaur",
            avatar_url="",
        )
        db.add(user)
        db.commit()
    elif user.display_name != "Karamjass Kaur":
        user.display_name = "Karamjass Kaur"
        db.commit()
    return user


# ── Authentication & Security Helpers ─────────────────────────────────────────

SESSION_COOKIE = "veriagent_session"
CSRF_HEADER = "X-CSRF-Token"


def _verify_supabase_token(access_token: str) -> dict[str, Any] | None:
    """Verify Supabase JWT token using the official Supabase SDK.

    Calls Supabase auth server to validate signature, expiration, and status.
    Never relies on an unverified decode.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY or not access_token:
        return None
    try:
        from supabase import create_client

        sb_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        user_res = sb_client.auth.get_user(access_token)
        if user_res and getattr(user_res, "user", None):
            u = user_res.user
            meta = getattr(u, "user_metadata", {}) or {}
            display_name = (
                meta.get("full_name")
                or meta.get("name")
                or (u.email.split("@")[0] if u.email else "User")
            )
            return {
                "id": str(u.id),
                "email": u.email or "",
                "display_name": display_name,
                "avatar_url": meta.get("avatar_url") or "",
            }
    except Exception as exc:
        print(f"Supabase token validation error: {exc}", file=sys.stderr)
        return None
    return None


def _csrf_token_for(session_id: str) -> str:
    """Derive deterministic CSRF token bound to a session ID."""
    key = _get_secret_key().encode("utf-8")
    return hmac.new(key, session_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _create_session_cookie(user_id: str) -> tuple[str, str]:
    """Generate signed session token and its corresponding CSRF token."""
    session_id = secrets.token_hex(16)
    csrf_token = _csrf_token_for(session_id)
    ts = int(time.time())
    msg = f"{user_id}:{session_id}:{ts}"
    sig = hmac.new(_get_secret_key().encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{msg}:{sig}", csrf_token


def _verify_session_cookie(cookie_val: str) -> tuple[str, str] | None:
    """Validate signed session cookie and return (user_id, expected_csrf)."""
    if not cookie_val or cookie_val.count(":") != 3:
        return None
    user_id, session_id, ts_str, sig = cookie_val.split(":", 3)
    try:
        ts = int(ts_str)
    except ValueError:
        return None
    # 7-day session lifetime
    if time.time() - ts > 7 * 86400:
        return None
    expected_msg = f"{user_id}:{session_id}:{ts}"
    expected_sig = hmac.new(
        _get_secret_key().encode("utf-8"), expected_msg.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    return user_id, _csrf_token_for(session_id)


def authenticate_request(
    db, require_csrf: bool = False
) -> tuple[User | None, str | None, int]:
    """Authenticate request via Bearer token, session cookie, or local dev mode.

    Returns:
        (user, error_message, http_status_code)
    """
    if PRODUCTION_CONFIG_ERROR:
        return None, PRODUCTION_CONFIG_ERROR, 503

    # 1. Bearer Token (Supabase Access Token)
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        supa_user = _verify_supabase_token(token)
        if not supa_user:
            return None, "Invalid or expired Supabase authentication token.", 401

        # Sync profile to database
        user = db.query(User).filter_by(id=supa_user["id"]).first()
        if not user:
            user = User(
                id=supa_user["id"],
                email=supa_user["email"],
                display_name=supa_user["display_name"],
                avatar_url=supa_user["avatar_url"],
            )
            db.add(user)
            db.commit()
        else:
            changed = False
            if user.email != supa_user["email"]:
                user.email = supa_user["email"]
                changed = True
            if user.display_name != supa_user["display_name"]:
                user.display_name = supa_user["display_name"]
                changed = True
            if changed:
                db.commit()
        # Bearer tokens do not rely on browser ambient credentials and are immune to browser CSRF.
        return user, None, 200

    # 2. Session Cookie Authentication
    cookie_val = request.cookies.get(SESSION_COOKIE)
    if cookie_val:
        verified = _verify_session_cookie(cookie_val)
        if verified:
            user_id, expected_csrf = verified
            if require_csrf and request.method in ("POST", "PUT", "DELETE", "PATCH"):
                client_csrf = (
                    request.headers.get(CSRF_HEADER)
                    or request.headers.get("X-Csrf-Token")
                    or ""
                )
                if not client_csrf or not hmac.compare_digest(client_csrf, expected_csrf):
                    return None, "CSRF validation failed for cookie session.", 403

            user = db.query(User).filter_by(id=user_id).first()
            if user:
                return user, None, 200

    # Require an authenticated session (session cookie or Bearer token) before chatting
    return None, "Authentication required. Please sign in to start chatting.", 401


# ── Static Asset Helpers ──────────────────────────────────────────────────────

@app.route("/")
def index() -> Response:
    """Serve the main SPA page."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename: str) -> Response:
    """Serve CSS, JS, images, and other static assets from the project folder.

    Protected: blocks database files, scripts, and private directories.
    """
    clean_name = filename.replace("\\", "/").strip("/")
    path_parts = [part for part in clean_name.split("/") if part]
    blocked_dirs = {"instance", "uploads", ".git", "__pycache__"}
    blocked_suffixes = (".py", ".db", ".sqlite", ".sqlite3", ".env", ".log")
    if (
        not path_parts
        or any(part.startswith(".") or part in blocked_dirs for part in path_parts)
        or any(clean_name.lower().endswith(suffix) for suffix in blocked_suffixes)
        or ".." in path_parts
    ):
        return jsonify({"error": "Access denied"}), 403
    return send_from_directory(BASE_DIR, filename)


# ── Health Endpoint ───────────────────────────────────────────────────────────

def _deps_ready() -> bool:
    """Compatibility helper: model service is Ollama, not Transformers."""
    import model_core as mc
    return bool(mc.model_status().get("connected"))


@app.route("/api/health", methods=["GET"])
def health() -> Response:
    """Return Ollama model, database, and authentication readiness."""
    import model_core as mc

    model_state = mc.model_status()
    db_ok = False
    db_type = "postgresql" if DATABASE_URL else "sqlite"
    if not PRODUCTION_CONFIG_ERROR:
        try:
            db = get_db_session()
            db.execute(select(1))
            db_ok = True
        except Exception as exc:
            print(f"Database health check error: {exc}", file=sys.stderr)

    auth_mode = "supabase" if (SUPABASE_URL and SUPABASE_ANON_KEY) else "local_dev"
    if IS_PRODUCTION and auth_mode != "supabase":
        auth_mode = "unconfigured_production"

    return jsonify(
        {
            "status": "ok" if (db_ok and model_state.get("connected") and not PRODUCTION_CONFIG_ERROR) else "setup_needed",
            "dependencies_ready": bool(model_state.get("connected")),
            "loaded": bool(model_state.get("loaded")),
            "model": mc.MODEL,
            "model_revision": None,
            "ollama_connected": bool(model_state.get("connected")),
            "model_available": bool(model_state.get("model_available")),
            "model_error": model_state.get("error"),
            "database": {
                "ready": db_ok,
                "engine": db_type,
                "sqlite_path": "instance/veriagent.db" if not DATABASE_URL else None,
            },
            "database_type": db_type,
            "auth": {
                "mode": auth_mode,
                "is_production": IS_PRODUCTION,
                "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
                "local_dev_allowed": (not IS_PRODUCTION and not SUPABASE_URL),
            },
            "production_error": PRODUCTION_CONFIG_ERROR,
        }
    )


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.route("/api/auth/config", methods=["GET"])
def auth_config() -> Response:
    """Return public client authentication configuration.

    Never exposes service-role keys or database credentials.
    """
    if PRODUCTION_CONFIG_ERROR:
        return (
            jsonify(
                {
                    "error": PRODUCTION_CONFIG_ERROR,
                    "is_production": True,
                    "auth_mode": "unconfigured_production",
                    "mode": "production",
                    "supabase_configured": False,
                    "database_type": "postgresql",
                }
            ),
            503,
        )

    auth_mode = "supabase" if (SUPABASE_URL and SUPABASE_ANON_KEY) else "local_dev"
    return jsonify(
        {
            "auth_mode": auth_mode,
            "mode": "production" if IS_PRODUCTION else "development",
            "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
            "database_type": "postgresql" if DATABASE_URL else "sqlite",
            "supabase_url": SUPABASE_URL if SUPABASE_URL else None,
            "supabase_anon_key": SUPABASE_ANON_KEY if SUPABASE_ANON_KEY else None,
            "is_production": IS_PRODUCTION,
            "local_dev_allowed": (not IS_PRODUCTION and not SUPABASE_URL),
        }
    )


@app.route("/api/auth/session", methods=["POST"])
def auth_session() -> Response:
    """Establish or verify user session and return CSRF token with profile."""
    if PRODUCTION_CONFIG_ERROR:
        return jsonify({"error": PRODUCTION_CONFIG_ERROR}), 503

    db = get_db_session()
    payload = request.get_json(silent=True) or {}
    token = payload.get("access_token", "").strip()
    email = str(payload.get("email", "")).strip().lower()
    display_name_input = str(payload.get("display_name", "")).strip()

    user: User | None = None
    if token:
        supa_user = _verify_supabase_token(token)
        if not supa_user:
            return jsonify({"error": "Invalid or expired Supabase token."}), 401
        user = db.query(User).filter_by(id=supa_user["id"]).first()
        if not user:
            user = User(
                id=supa_user["id"],
                email=supa_user["email"],
                display_name=supa_user["display_name"],
                avatar_url=supa_user["avatar_url"],
            )
            db.add(user)
            db.commit()
    elif not IS_PRODUCTION and not SUPABASE_URL:
        # Local development authentication (bound to localhost)
        if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
            return jsonify({"error": "Single-user mode restricted to localhost."}), 401

        if email:
            if "@" not in email:
                return jsonify({"error": "Please enter a valid email address."}), 400
            user_id = f"local-{hashlib.sha256(email.encode('utf-8')).hexdigest()[:16]}"
            user = db.query(User).filter_by(id=user_id).first()
            derived_name = _extract_first_name(display_name_input, email)
            if not user:
                user = User(
                    id=user_id,
                    email=email,
                    display_name=derived_name,
                    avatar_url="",
                )
                db.add(user)
                db.commit()
            else:
                if user.display_name != derived_name or re.search(r"\d", user.display_name or ""):
                    user.display_name = derived_name
                    db.commit()
        else:
            user = _ensure_local_dev_user(db)
    else:
        return jsonify({"error": "Authentication required. Please sign in."}), 401

    cookie_val, csrf_token = _create_session_cookie(user.id)
    resp = make_response(
        jsonify(
            {
                "user": user.to_dict(),
                "csrf_token": csrf_token,
                "auth_mode": "supabase" if SUPABASE_URL else "local_dev",
                "authenticated": True,
            }
        )
    )
    resp.set_cookie(
        SESSION_COOKIE,
        cookie_val,
        httponly=True,
        samesite="Lax",
        secure=(IS_PRODUCTION or request.is_secure),
        max_age=7 * 86400,
    )
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout() -> Response:
    """Clear session cookie."""
    resp = make_response(jsonify({"status": "logged_out"}))
    resp.delete_cookie(SESSION_COOKIE, samesite="Lax")
    return resp


@app.route("/api/auth/user", methods=["GET"])
def auth_user() -> Response:
    """Return currently authenticated user profile."""
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=False)
    if not user:
        return jsonify({"error": err, "authenticated": False, "user": None}), status
    # Also generate a fresh CSRF token if cookie authenticated
    csrf_token = None
    cookie_val = request.cookies.get(SESSION_COOKIE)
    if cookie_val:
        verified = _verify_session_cookie(cookie_val)
        if verified:
            _, csrf_token = verified
    return jsonify(
        {
            "user": user.to_dict(),
            "csrf_token": csrf_token,
            "auth_mode": "supabase" if SUPABASE_URL else "local_dev",
            "authenticated": True,
        }
    )


# ── Conversation Database Endpoints ───────────────────────────────────────────

@app.route("/api/conversations", methods=["GET"])
def list_conversations() -> Response:
    """List conversations belonging to the authenticated user."""
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=False)
    if not user:
        return jsonify({"error": err}), status

    items = (
        db.query(Conversation)
        .filter_by(user_id=user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
        .all()
    )
    return jsonify({"conversations": [conv.to_dict(include_messages=False) for conv in items]})


@app.route("/api/conversations", methods=["POST"])
def create_conversation() -> Response:
    """Create a new conversation owned by the authenticated user."""
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=True)
    if not user:
        return jsonify({"error": err}), status

    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "New Conversation")).strip()[:90] or "New Conversation"
    conv = Conversation(user_id=user.id, title=title)
    db.add(conv)
    db.commit()
    return jsonify({"conversation": conv.to_dict(include_messages=True)}), 201


@app.route("/api/conversations/<string:conv_id>", methods=["GET"])
def get_conversation(conv_id: str) -> Response:
    """Retrieve a single conversation and its messages.

    Scopes query to authenticated user; returns 404 if not owned by caller.
    """
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=False)
    if not user:
        return jsonify({"error": err}), status

    conv = db.query(Conversation).filter_by(id=conv_id, user_id=user.id).first()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    return jsonify({"conversation": conv.to_dict(include_messages=True)})


@app.route("/api/conversations/<string:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id: str) -> Response:
    """Delete a conversation owned by the authenticated user.

    Returns 404 if not found or not owned by caller.
    """
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=True)
    if not user:
        return jsonify({"error": err}), status

    conv = db.query(Conversation).filter_by(id=conv_id, user_id=user.id).first()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    db.delete(conv)
    db.commit()
    return jsonify({"status": "deleted", "id": conv_id})


@app.route("/api/conversations/import", methods=["POST"])
def import_conversations() -> Response:
    """Deliberately import browser-local conversations into user's database records.

    Keeps browser-local storage safe until the user requests this import.
    """
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=True)
    if not user:
        return jsonify({"error": err}), status

    payload = request.get_json(silent=True) or {}
    items = payload.get("conversations", [])
    if not isinstance(items, list):
        return jsonify({"error": "Invalid payload; expected list of conversations"}), 400

    imported_count = 0
    for item in items[:20]:  # bounded import batch
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "Imported Conversation")).strip()[:90] or "Imported Conversation"
        raw_msgs = item.get("messages", [])
        if not isinstance(raw_msgs, list):
            raw_msgs = []

        conv = Conversation(user_id=user.id, title=title)
        db.add(conv)
        db.flush()  # assign conv.id

        for m in raw_msgs[-20:]:
            if not isinstance(m, dict):
                continue
            role = "assistant" if m.get("role") == "assistant" else "user"
            content = str(m.get("content", "")).strip()[:6000]
            if not content:
                continue
            msg = Message(
                conversation_id=conv.id,
                role=role,
                content=content,
                verification_label=str(m.get("verification", "")).strip()[:128] or None,
                verified=bool(m.get("verified")),
                elapsed=float(m.get("elapsed", 0.0) or 0.0),
            )
            db.add(msg)

        imported_count += 1

    db.commit()
    return jsonify({"status": "imported", "count": imported_count, "imported_count": imported_count})


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat_endpoint() -> Response:
    """Run reasoning pipeline and persist user and assistant messages."""
    import model_core as mc

    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=True)
    if not user:
        return jsonify({"error": err}), status

    payload = request.get_json(silent=True) or {}
    question: str = str(payload.get("question", "")).strip()
    conversation_id = payload.get("conversation_id")

    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 4000:
        return jsonify({"error": "question exceeds 4 000-character limit"}), 400

    # Retrieve or create conversation owned by caller
    conv: Conversation | None = None
    if conversation_id:
        conv = db.query(Conversation).filter_by(id=conversation_id, user_id=user.id).first()
        if not conv:
            return jsonify({"error": "Conversation not found"}), 404
    else:
        # Title from question snippet
        clean_prompt = " ".join(question.split())
        title = (clean_prompt[:44].rstrip() + "…") if len(clean_prompt) > 44 else clean_prompt
        conv = Conversation(user_id=user.id, title=title)
        db.add(conv)
        db.flush()

    # Build context from this user's persisted conversation only. Never trust a client-supplied history.
    prior_rows = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc())
        .limit(12)
        .all()
    )
    history = [
        {"role": row.role, "content": row.content}
        for row in reversed(prior_rows)
    ]

    # Save user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()

    try:
        result = mc.answer_question(question, history)
    except mc.ModelUnavailable as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                    "hint": (
                        f"Start Ollama and install the model with: ollama pull {mc.MODEL}"
                    ),
                    "conversation_id": conv.id,
                }
            ),
            503,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": "An internal inference error occurred.", "conversation_id": conv.id}), 500

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=result.get("answer", ""),
        verification_label=result.get("verification_label", "Internal review only"),
        verified=bool(result.get("verified")),
        elapsed=float(result.get("elapsed", 0)),
        model=result.get("model", ""),
        review_rounds=int(result.get("review_rounds", 0)),
    )
    db.add(assistant_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()

    return jsonify(
        {
            "answer": result.get("answer", ""),
            "verification_label": result.get("verification_label", "Internal review only"),
            "verified": bool(result.get("verified")),
            "elapsed": float(result.get("elapsed", 0)),
            "model": result.get("model", ""),
            "trace": result.get("trace", []),
            "review_rounds": int(result.get("review_rounds", 0)),
            "conversation_id": conv.id,
            "conversation_title": conv.title,
            "message_id": assistant_msg.id,
        }
    )


# ── Benchmarks Endpoints ──────────────────────────────────────────────────────

@app.route("/api/benchmarks", methods=["GET"])
def list_benchmarks() -> Response:
    """List recorded benchmark evaluation summaries."""
    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=False)
    if not user:
        return jsonify({"error": err}), status
    runs = (
        db.query(BenchmarkRun)
        .filter_by(user_id=user.id)
        .order_by(BenchmarkRun.created_at.desc())
        .limit(25)
        .all()
    )
    return jsonify([b.to_dict() for b in runs])


@app.route("/api/benchmarks/run", methods=["POST"])
def run_benchmark_endpoint() -> Response:
    """Execute a benchmark pilot evaluation and record its summary in the database."""
    import model_core as mc

    db = get_db_session()
    user, err, status = authenticate_request(db, require_csrf=True)
    if not user:
        return jsonify({"error": err}), status

    payload = request.get_json(silent=True) or {}
    gsm8k_limit = int(payload.get("gsm8k_limit", 5))
    truthfulqa_limit = int(payload.get("truthfulqa_limit", 5))
    seed = int(payload.get("seed", 42))

    # Safe bounds for API requests
    gsm8k_limit = max(0, min(gsm8k_limit, 40))
    truthfulqa_limit = max(0, min(truthfulqa_limit, 50))

    try:
        summary = mc.run_benchmarks(
            gsm8k_limit=gsm8k_limit,
            truthfulqa_limit=truthfulqa_limit,
            seed=seed,
            progress=False,
        )
    except mc.ModelUnavailable as exc:
        return jsonify({"error": str(exc), "hint": f"Start Ollama and install the model with: ollama pull {mc.MODEL}"}), 503
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "hint": "Install optional benchmark data support with: pip install datasets"}), 503
    except Exception as exc:
        return jsonify({"error": f"Benchmark evaluation error: {exc}"}), 500

    gsm_data = summary.get("gsm8k", {})
    tqa_data = summary.get("truthfulqa", {})

    b_record = BenchmarkRun(
        user_id=user.id,
        model=str(summary.get("model", mc.MODEL)),
        model_revision=str(summary.get("model_revision", "") or ""),
        seed=seed,
        gsm8k_accuracy=gsm_data.get("exact_numeric_accuracy"),
        gsm8k_sample_size=gsm_data.get("sample_size"),
        gsm8k_calculator_rate=gsm_data.get("calculator_crosscheck_rate"),
        truthfulqa_mc1_accuracy=tqa_data.get("generation_choice_accuracy"),
        truthfulqa_mc2_mass=None,
        truthfulqa_sample_size=tqa_data.get("sample_size"),
        summary_json=json.dumps(summary),
    )
    db.add(b_record)
    db.commit()

    return jsonify(b_record.to_dict()), 201


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_root = str(BASE_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Initialize database engine and schema
    if PRODUCTION_CONFIG_ERROR:
        print(f"CRITICAL: {PRODUCTION_CONFIG_ERROR}", file=sys.stderr)
        sys.exit(1)

    get_engine()

    port = int(os.environ.get("PORT", 5000))
    print(f"VeriAgent starting on http://127.0.0.1:{port}/")
    print(f"Database: {'PostgreSQL via DATABASE_URL' if DATABASE_URL else 'SQLite at instance/veriagent.db'}")
    print(f"Auth: {'Supabase Auth configured' if SUPABASE_URL and SUPABASE_ANON_KEY else 'Local single-user mode (localhost only)'}")
    import model_core as mc
    print(f"Model: {mc.MODEL} via Ollama at {mc.OLLAMA_BASE_URL}")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
