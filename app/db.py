import sqlite3
from pathlib import Path
from typing import Iterator

from app.settings import settings
from app.time_utils import now_kst_iso


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()

    # Lightweight migration for existing DBs
    with get_connection() as conn:
        user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "phone" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_phone ON users(phone)")

        emotion_cols = {row["name"] for row in conn.execute("PRAGMA table_info(emotion_scores)").fetchall()}
        energy_columns = [
            ("energy_vx", "REAL"),
            ("energy_vy", "REAL"),
            ("energy_tau", "REAL"),
            ("energy_angle", "REAL"),
            ("energy_magnitude", "REAL"),
            ("energy_coherence", "REAL"),
            ("energy_angle_reliable", "INTEGER NOT NULL DEFAULT 0"),
            ("energy_active_count", "INTEGER"),
            ("emotion_source", "TEXT"),
            ("emotion_error", "TEXT"),
            ("energy_status", "TEXT"),
        ]
        for col_name, col_type in energy_columns:
            if col_name not in emotion_cols:
                conn.execute(f"ALTER TABLE emotion_scores ADD COLUMN {col_name} {col_type}")

        app_config_cols = {row["name"] for row in conn.execute("PRAGMA table_info(app_config)").fetchall()}
        if app_config_cols and "updated_by" not in app_config_cols:
            conn.execute("ALTER TABLE app_config ADD COLUMN updated_by INTEGER")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_message_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assistant_message_id INTEGER NOT NULL UNIQUE,
                response_json TEXT NOT NULL DEFAULT '{}',
                evidence_json TEXT NOT NULL DEFAULT '[]',
                schema_valid INTEGER NOT NULL DEFAULT 0,
                generation_source TEXT NOT NULL,
                sanitizer_hit INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (assistant_message_id) REFERENCES messages(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_message_meta_message_id ON assistant_message_meta(assistant_message_id)"
        )
        assistant_meta_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(assistant_message_meta)").fetchall()
        }
        if assistant_meta_cols and "sanitizer_hit" not in assistant_meta_cols:
            conn.execute("ALTER TABLE assistant_message_meta ADD COLUMN sanitizer_hit INTEGER NOT NULL DEFAULT 0")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL CHECK (gender IN ('male', 'female', 'other', 'unknown')),
                residence TEXT,
                assistant_guidance TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
            """
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_profiles_user_id ON patient_profiles(user_id)")
        profile_cols = {row["name"] for row in conn.execute("PRAGMA table_info(patient_profiles)").fetchall()}
        if profile_cols and "assistant_guidance" not in profile_cols:
            conn.execute("ALTER TABLE patient_profiles ADD COLUMN assistant_guidance TEXT NOT NULL DEFAULT ''")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS counselor_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_user_id INTEGER NOT NULL,
                counselor_user_id INTEGER NOT NULL,
                session_summary TEXT NOT NULL,
                intervention_note TEXT,
                followup_plan TEXT,
                counselor_risk_level REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (patient_user_id) REFERENCES users(id),
                FOREIGN KEY (counselor_user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_counselor_notes_patient_created ON counselor_notes(patient_user_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_counselor_notes_counselor_created ON counselor_notes(counselor_user_id, created_at DESC)"
        )

        # Backfill consent rows for existing patient users
        now = now_kst_iso()
        conn.execute(
            """
            INSERT INTO patient_ai_consents (
                user_id, consent_given, consent_version, consent_text, created_at, updated_at
            )
            SELECT users.id, 0, 'v1', 'AI 상담 보조 기능 사용에 동의합니다.', ?, ?
            FROM users
            LEFT JOIN patient_ai_consents ON patient_ai_consents.user_id = users.id
            WHERE users.role = 'patient' AND patient_ai_consents.user_id IS NULL
            """,
            (now, now),
        )
        conn.commit()


def stream_rows(query: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
    conn = get_connection()
    cur = conn.execute(query, params)
    try:
        for row in cur:
            yield row
    finally:
        cur.close()
        conn.close()
