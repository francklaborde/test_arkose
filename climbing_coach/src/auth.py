"""
Minimal magic-link authentication for Arkose Coach.

Three tables live in the same SQLite DB as the climbing data:
- users          : one row per person (account_id, email, sboulder_user_id, profile_path)
- magic_links    : short-lived one-time tokens sent by email (15 min TTL)
- auth_sessions  : long-lived session tokens stored in the browser cookie
                   (90 days, renewed on activity so regular users never see it expire)

No passwords are ever stored — email possession IS the credential.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("climbing_coach")

MAGIC_LINK_TTL = timedelta(minutes=15)
SESSION_TTL = timedelta(days=90)
SESSION_RENEW_THRESHOLD = timedelta(days=7)  # renew if less than this remains

COOKIE_NAME = "arkose_session"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def ensure_auth_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                account_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                sboulder_user_id TEXT,
                profile_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS magic_links (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


class AuthStore:
    """Thin wrapper around the auth tables. One instance per server process."""

    def __init__(self, db_path: str, profiles_dir: str):
        self.db_path = db_path
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        ensure_auth_tables(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------

    def get_or_create_account(self, email: str) -> sqlite3.Row:
        email = email.strip().lower()
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                return row
            account_id = str(uuid.uuid4())
            profile_path = str(self.profiles_dir / f"{account_id}.json")
            conn.execute(
                "INSERT INTO users (account_id, email, sboulder_user_id, profile_path, created_at) "
                "VALUES (?, ?, NULL, ?, ?)",
                (account_id, email, profile_path, _iso(_now())),
            )
            conn.commit()
            log.info("New account created for %s", email)
            return conn.execute(
                "SELECT * FROM users WHERE account_id = ?", (account_id,)
            ).fetchone()
        finally:
            conn.close()

    def get_account(self, account_id: str) -> Optional[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM users WHERE account_id = ?", (account_id,)
            ).fetchone()
        finally:
            conn.close()

    def set_sboulder_user_id(self, account_id: str, sboulder_user_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET sboulder_user_id = ? WHERE account_id = ?",
                (sboulder_user_id.strip(), account_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Magic links
    # ------------------------------------------------------------------

    def create_magic_link(self, email: str) -> str:
        token = secrets.token_urlsafe(32)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO magic_links (token, email, expires_at, used) VALUES (?, ?, ?, 0)",
                (token, email.strip().lower(), _iso(_now() + MAGIC_LINK_TTL)),
            )
            conn.commit()
        finally:
            conn.close()
        return token

    def consume_magic_link(self, token: str) -> Optional[str]:
        """Validate + mark used. Returns the associated email, or None if
        the token is missing, expired, or already used."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM magic_links WHERE token = ?", (token,)
            ).fetchone()
            if not row or row["used"] or _parse(row["expires_at"]) < _now():
                return None
            conn.execute("UPDATE magic_links SET used = 1 WHERE token = ?", (token,))
            conn.commit()
            return row["email"]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Sessions (browser cookie)
    # ------------------------------------------------------------------

    def create_session(self, account_id: str) -> str:
        token = secrets.token_urlsafe(32)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO auth_sessions (token, account_id, expires_at) VALUES (?, ?, ?)",
                (token, account_id, _iso(_now() + SESSION_TTL)),
            )
            conn.commit()
        finally:
            conn.close()
        return token

    def resolve_session(self, token: str) -> Optional[sqlite3.Row]:
        """Returns the account row for a valid, non-expired session, silently
        renewing its expiry if it's getting close to running out — so anyone
        using the app regularly never sees it expire."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM auth_sessions WHERE token = ?", (token,)
            ).fetchone()
            if not row:
                return None
            expires_at = _parse(row["expires_at"])
            if expires_at < _now():
                conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
                conn.commit()
                return None
            if expires_at - _now() < SESSION_RENEW_THRESHOLD:
                conn.execute(
                    "UPDATE auth_sessions SET expires_at = ? WHERE token = ?",
                    (_iso(_now() + SESSION_TTL), token),
                )
                conn.commit()
            return conn.execute(
                "SELECT * FROM users WHERE account_id = ?", (row["account_id"],)
            ).fetchone()
        finally:
            conn.close()

    def delete_session(self, token: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Email delivery (Resend)
# ---------------------------------------------------------------------------

def send_magic_link_email(
    resend_api_key: str, from_email: str, to_email: str, verify_url: str
) -> None:
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {resend_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": from_email,
            "to": [to_email],
            "subject": "Ton lien de connexion Arkose Coach",
            "html": (
                "<p>Clique sur ce lien pour te connecter à Arkose Coach "
                f"(valide 15 minutes) :</p><p><a href='{verify_url}'>{verify_url}</a></p>"
            ),
        },
        timeout=10,
    )
    resp.raise_for_status()
