from hashlib import sha256
from typing import Optional

import bcrypt

from app.db import get_connection
from app.time_utils import now_kst_iso


def _safe_pin(pin: str) -> str:
    raw = pin or ""
    # bcrypt는 72 bytes 제한이 있으므로 초과 시 1차 해시로 축약
    if len(raw.encode("utf-8")) > 72:
        return sha256(raw.encode("utf-8")).hexdigest()
    return raw


def hash_pin(pin: str) -> str:
    # bcrypt는 기본적으로 바이트(bytes)를 요구합니다.
    safe_bytes = _safe_pin(pin).encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(safe_bytes, salt).decode("utf-8")


def verify_pin(pin: str, pin_hash: str) -> bool:
    safe_bytes = _safe_pin(pin).encode("utf-8")
    hash_bytes = pin_hash.encode("utf-8")
    return bcrypt.checkpw(safe_bytes, hash_bytes)


def create_user(username: str, role: str, pin: str, phone: Optional[str] = None) -> int:
    pin_hash = hash_pin(pin)
    now = now_kst_iso()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, role, phone, pin_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, role, phone, pin_hash, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_phone(phone: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def has_any_user() -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(1) AS cnt FROM users").fetchone()
        return bool(row["cnt"])


def list_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id, username, role, phone, created_at FROM users ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
