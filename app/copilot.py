from __future__ import annotations

import json
from typing import Optional

from app.db import get_connection
from app.openai_client import chat_text
from app.rag import search_rag
from app.settings import settings
from app.time_utils import now_kst_iso


COPILOT_SYSTEM_PROMPT = """너는 심리상담 전문가를 보조하는 AI 코파일럿이다.
- 진단을 단정하지 말고 근거 중심으로 가설/권고를 제시한다.
- 치료를 대체하지 않으며, 임상적 판단은 전문가가 수행한다.
- 자해/자살 위험 신호가 보이면 즉시 안전계획, 전문기관 연계, 응급 대응 우선순위를 제시한다.
- 한국어로 간결하고 구조화된 답변을 제공한다.
"""

DEFAULT_COPILOT_CONTEXT_MODE = "global"


def validate_copilot_context(context_mode: str, selected_patient_id: Optional[int]) -> tuple[bool, str]:
    if context_mode not in {"patient", "global"}:
        return False, "컨텍스트 모드가 올바르지 않습니다."
    if context_mode == "patient" and not selected_patient_id:
        return False, "환자 컨텍스트 모드에서는 환자를 선택해야 합니다."
    return True, ""


def create_copilot_thread(counselor_user_id: int, title: str = "새 코파일럿 스레드") -> int:
    now = now_kst_iso()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO copilot_threads (
                counselor_user_id,
                selected_patient_id,
                context_mode,
                title,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (counselor_user_id, None, DEFAULT_COPILOT_CONTEXT_MODE, title, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_copilot_threads(counselor_user_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, counselor_user_id, selected_patient_id, context_mode, title, created_at, updated_at
            FROM copilot_threads
            WHERE counselor_user_id = ?
            ORDER BY updated_at DESC
            """,
            (counselor_user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_copilot_thread(thread_id: int, counselor_user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, counselor_user_id, selected_patient_id, context_mode, title, created_at, updated_at
            FROM copilot_threads
            WHERE id = ? AND counselor_user_id = ?
            """,
            (thread_id, counselor_user_id),
        ).fetchone()
        return dict(row) if row else None


def update_copilot_context(
    thread_id: int,
    counselor_user_id: int,
    context_mode: str,
    selected_patient_id: Optional[int],
) -> None:
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE copilot_threads
            SET context_mode = ?, selected_patient_id = ?, updated_at = ?
            WHERE id = ? AND counselor_user_id = ?
            """,
            (context_mode, selected_patient_id, now, thread_id, counselor_user_id),
        )
        conn.commit()


def add_copilot_message(
    thread_id: int,
    role: str,
    content: str,
    model: str = "",
    meta: Optional[dict] = None,
) -> int:
    now = now_kst_iso()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO copilot_messages (thread_id, role, content, created_at, model, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (thread_id, role, content, now, model, meta_json),
        )
        conn.commit()

    with get_connection() as conn:
        conn.execute(
            "UPDATE copilot_threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
        conn.commit()

    return int(cur.lastrowid)


def list_copilot_messages(thread_id: int, counselor_user_id: int) -> list[dict]:
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM copilot_threads WHERE id = ? AND counselor_user_id = ?",
            (thread_id, counselor_user_id),
        ).fetchone()
        if not exists:
            return []

        rows = conn.execute(
            """
            SELECT id, thread_id, role, content, created_at, model, meta_json
            FROM copilot_messages
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()
        data = []
        for row in rows:
            record = dict(row)
            try:
                record["meta"] = json.loads(record.get("meta_json") or "{}")
            except json.JSONDecodeError:
                record["meta"] = {}
            data.append(record)
        return data


def list_patient_users() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username FROM users WHERE role = 'patient' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def _fetch_patient_summary(patient_id: int, limit: int = 6) -> str:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT messages.role, messages.content, messages.created_at
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            WHERE sessions.user_id = ?
            ORDER BY messages.id DESC
            LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()
        if not rows:
            return "선택 환자의 최근 대화 기록이 없습니다."

        recent = []
        for row in reversed(rows):
            role = "환자" if row["role"] == "user" else "AI"
            recent.append(f"- {role}: {row['content']}")
        return "\n".join(recent)


def _fetch_global_summary() -> str:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(1) AS cnt FROM users WHERE role = 'patient'"
        ).fetchone()
        patient_count = int(row["cnt"]) if row else 0
    return f"등록 환자 수: {patient_count}"


def build_copilot_prompt(
    counselor_input: str,
    context_mode: str,
    selected_patient_id: Optional[int],
    rag_chunks: list[dict],
    patient_summary: str,
) -> str:
    rag_text = "\n".join([f"- ({item['title']}) {item['chunk_text']}" for item in rag_chunks])
    patient_text = str(selected_patient_id) if selected_patient_id else "none"

    return f"""
[컨텍스트 모드]
{context_mode}

[선택 환자 ID]
{patient_text}

[환자/전역 요약]
{patient_summary}

[RAG 근거]
{rag_text}

[상담자 입력]
{counselor_input}
""".strip()


def generate_copilot_response(
    counselor_input: str,
    context_mode: str,
    selected_patient_id: Optional[int],
) -> tuple[str, dict]:
    rag_result = search_rag(
        counselor_input,
        top_k=settings.rag_top_k,
        user_id=selected_patient_id if context_mode == "patient" else None,
    )

    if context_mode == "patient" and selected_patient_id:
        summary = _fetch_patient_summary(selected_patient_id)
    else:
        summary = _fetch_global_summary()

    prompt = build_copilot_prompt(
        counselor_input=counselor_input,
        context_mode=context_mode,
        selected_patient_id=selected_patient_id,
        rag_chunks=rag_result.get("doc_chunks", []),
        patient_summary=summary,
    )

    text = chat_text(
        system_prompt=COPILOT_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=700,
    )

    meta = {
        "context_mode": context_mode,
        "selected_patient_id": selected_patient_id,
        "doc_chunk_count": len(rag_result.get("doc_chunks", [])),
        "message_chunk_count": len(rag_result.get("messages", [])),
    }
    return text, meta
