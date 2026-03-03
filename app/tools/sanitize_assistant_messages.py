from __future__ import annotations

import argparse
import sqlite3

from app.patient_response import FORBIDDEN_PATTERNS, sanitize_patient_display_text
from app.settings import settings
from app.time_utils import now_kst_iso


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _build_filter_clause() -> tuple[str, list[str]]:
    clauses = []
    params: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        clauses.append("LOWER(content) LIKE ?")
        params.append(f"%{pattern.lower()}%")
    return " OR ".join(clauses), params


def _ensure_backup_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assistant_message_legacy_backup (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL UNIQUE,
            original_content TEXT NOT NULL,
            backed_up_at TEXT NOT NULL
        )
        """
    )


def dry_run(db_path: str) -> int:
    where_clause, params = _build_filter_clause()
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, content
            FROM messages
            WHERE role = 'assistant' AND ({where_clause})
            ORDER BY id ASC
            """,
            params,
        ).fetchall()

    print(f"[dry-run] target_count={len(rows)}")
    for row in rows[:20]:
        preview = (row["content"] or "").replace("\n", " ")[:80]
        print(f"- message_id={row['id']} preview={preview}")
    if len(rows) > 20:
        print(f"... and {len(rows) - 20} more")
    return len(rows)


def apply(db_path: str) -> tuple[int, int]:
    where_clause, params = _build_filter_clause()
    changed = 0
    scanned = 0
    now = now_kst_iso()

    with _connect(db_path) as conn:
        _ensure_backup_table(conn)
        rows = conn.execute(
            f"""
            SELECT id, content
            FROM messages
            WHERE role = 'assistant' AND ({where_clause})
            ORDER BY id ASC
            """,
            params,
        ).fetchall()

        for row in rows:
            scanned += 1
            message_id = int(row["id"])
            original = row["content"] or ""
            sanitized, hit = sanitize_patient_display_text(original)
            if not hit:
                continue

            conn.execute(
                """
                INSERT OR IGNORE INTO assistant_message_legacy_backup (message_id, original_content, backed_up_at)
                VALUES (?, ?, ?)
                """,
                (message_id, original, now),
            )
            conn.execute("UPDATE messages SET content = ? WHERE id = ?", (sanitized, message_id))
            conn.execute(
                """
                UPDATE assistant_message_meta
                SET sanitizer_hit = 1
                WHERE assistant_message_id = ?
                """,
                (message_id,),
            )
            changed += 1

        conn.commit()

    return scanned, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize leaked assistant metadata from patient-visible messages.")
    parser.add_argument("--db-path", default=settings.db_path, help="SQLite DB path")
    parser.add_argument("--apply", action="store_true", help="Apply sanitize updates")
    args = parser.parse_args()

    if not args.apply:
        dry_run(args.db_path)
        return

    scanned, changed = apply(args.db_path)
    print(f"[apply] scanned={scanned} changed={changed}")


if __name__ == "__main__":
    main()
