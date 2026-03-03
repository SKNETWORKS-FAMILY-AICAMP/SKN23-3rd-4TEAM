from __future__ import annotations

from typing import Dict

from app.db import get_connection
from app.settings import settings
from app.time_utils import now_kst_iso


DEFAULT_SYSTEM_PROMPT = """[ROLE]
너는 로컬 심리상담 보조 AI다. 목표는 안전하고 근거 기반으로 상담 대화를 보조하는 것이다.

[GLOBAL RULES]
1) 전문 진단/처방을 단정하지 않는다.
2) 제공된 context 밖 사실을 단정하지 않는다.
3) 위험 신호가 감지되면 안전 안내를 최우선으로 한다.
4) 내부 추론(CoT) 장문을 노출하지 않는다.
5) 한국어로 간결하고 존중하는 톤을 유지한다.

[FAILURE POLICY]
근거가 부족하면 추측하지 말고 `insufficient_data`를 명시한 뒤 확인 질문을 제시한다.
"""

DEFAULT_USER_PROMPT_TEMPLATE = """
[INSTRUCTION]
1) 입력 감정과 핵심 문제를 요약하라.
2) 제공된 context를 우선 사용하고, 없으면 `insufficient_data`를 명시하라.
3) 위험/응급 가능성이 보이면 즉시 안전 행동 지침을 포함하라.
4) 아래 OUTPUT CONTRACT를 지켜 답변하라.

[CONTEXT_DOCS]
{doc_context}

[CONTEXT_HISTORY]
{msg_context}

[HELPER_GUIDANCE]
{assistant_guidance}

[INPUT]
{user_text}

[OUTPUT CONTRACT]
- 공감 1~2문장
- 상황 요약 1~2문장
- 실행 가능한 다음 단계 2~3개
- [근거] 섹션에 사용한 문맥 1~3개
- 정보 부족 시 `insufficient_data: true`를 명시

[QUALITY CHECK]
- schema_valid=true
- constraint_violations=[]
- evidence_coverage>=1
""".strip()

DEFAULT_PATIENT_SYSTEM_PROMPT = """[ROLE]
너는 심리상담 환자용 답변 생성기다.
안전하고 공감적인 문장을 만들되, 내부 품질 메타는 절대 노출하지 않는다.

[POLICY PRIORITY]
P0 Safety: risk_score >= risk_threshold 이거나 safety_urgent=true로 판단되면, 포맷 선호와 무관하게 안전 확보/주변 연락/1393 또는 112 안내를 반드시 포함한다.
P1 Respect: 환자가 리스트/번호/매뉴얼형 표현을 거부하면 목록형 문장 생성을 피한다.
P2 Validation: 해결책 제시 전에 감정 타당화 문장을 1~2문장 먼저 제시한다.
P3 Guidance: 비고위험일 때만 필요한 경우 next_steps를 제시한다(0~3개).
P4 Privacy/Ethics: internal_evidence, insufficient_data는 내부 평가용이며 환자 화면 문장으로 직접 노출하지 않는다.

[RULES]
1) 출력은 반드시 JSON 객체 1개만 반환한다.
2) 마크다운/코드블록/설명문을 추가하지 않는다.
3) 한국어로 간결하고 존중하는 톤을 유지한다.
"""

DEFAULT_PATIENT_USER_PROMPT_TEMPLATE = """
[INSTRUCTION]
아래 context와 입력을 바탕으로 환자 응답용 JSON을 작성하라.
추측하지 말고 모르면 insufficient_data=true로 표시하라.
환자가 번호/리스트/매뉴얼형 표현을 거부하는 포맷 거부 상황이면 next_steps는 빈 배열([])로 두어도 된다.
단, 고위험 상황에서는 포맷 선호와 무관하게 safety_urgent=true, safety_actions를 반드시 채워라.

[CONTEXT_DOCS]
{doc_context}

[CONTEXT_HISTORY]
{msg_context}

[HELPER_GUIDANCE]
{assistant_guidance}

[SESSION_META]
{session_meta}

[INPUT]
{user_text}

[OUTPUT CONTRACT]
반드시 JSON 객체로만 출력하고 키는 아래만 사용:
- empathy: string
- summary: string
- next_steps: string array (0~3)
- safety_urgent: boolean
- safety_actions: string array
- insufficient_data: boolean
- internal_evidence: string array (0~3)
""".strip()

RUNTIME_KEY_ORDER = (
    "openai_model",
    "openai_api_key",
    "system_prompt",
    "user_prompt_template",
    "patient_system_prompt",
    "patient_user_prompt_template",
)
RUNTIME_KEYS = set(RUNTIME_KEY_ORDER)
RUNTIME_DB_KEYS_SQL = ", ".join(f"'{key}'" for key in RUNTIME_KEY_ORDER)
REQUIRED_PROMPT_PLACEHOLDERS = ("{doc_context}", "{msg_context}", "{user_text}")
PATIENT_REQUIRED_SCHEMA_TOKENS = (
    "empathy",
    "summary",
    "next_steps",
    "safety_urgent",
    "safety_actions",
    "insufficient_data",
    "internal_evidence",
)


def _missing_required_tokens(text: str, required_tokens: tuple[str, ...]) -> list[str]:
    return [token for token in required_tokens if token not in text]


def _missing_placeholder_names(text: str) -> str:
    missing = _missing_required_tokens(text, REQUIRED_PROMPT_PLACEHOLDERS)
    return ", ".join(item.strip("{}") for item in missing)


def validate_prompt_template(template: str) -> tuple[bool, str]:
    cleaned = (template or "").strip()
    if not cleaned:
        return False, "유저 프롬프트 템플릿은 비워둘 수 없습니다."
    missing_names = _missing_placeholder_names(cleaned)
    if missing_names:
        return False, f"필수 플레이스홀더가 누락되었습니다: {missing_names}"
    ok, message = _validate_template_format(cleaned)
    if not ok:
        return False, message
    return True, ""


def validate_patient_prompt_template(template: str) -> tuple[bool, str]:
    cleaned = (template or "").strip()
    if not cleaned:
        return False, "환자 프롬프트 템플릿은 비워둘 수 없습니다."

    missing_names = _missing_placeholder_names(cleaned)
    if missing_names:
        return False, f"필수 플레이스홀더가 누락되었습니다: {missing_names}"

    missing_tokens = _missing_required_tokens(cleaned, PATIENT_REQUIRED_SCHEMA_TOKENS)
    if missing_tokens:
        return False, f"JSON 계약 키가 누락되었습니다: {', '.join(missing_tokens)}"

    ok, message = _validate_template_format(cleaned)
    if not ok:
        return False, message

    return True, ""


class _TemplateProbeDict(dict):
    def __missing__(self, key: str) -> str:
        return f"<{key}>"


def _validate_template_format(template: str) -> tuple[bool, str]:
    probe = _TemplateProbeDict(
        doc_context="DOC",
        msg_context="MSG",
        user_text="USER",
        assistant_guidance="GUIDANCE",
        session_meta="META",
    )
    try:
        template.format_map(probe)
    except (ValueError, KeyError) as exc:
        return False, f"프롬프트 템플릿 포맷 문법이 올바르지 않습니다: {exc}"
    return True, ""


def mask_api_key(value: str) -> str:
    key = (value or "").strip()
    if not key:
        return "-"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _runtime_defaults() -> Dict[str, str]:
    return {
        "openai_model": settings.openai_model,
        "openai_api_key": settings.openai_api_key,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt_template": DEFAULT_USER_PROMPT_TEMPLATE,
        "patient_system_prompt": DEFAULT_PATIENT_SYSTEM_PROMPT,
        "patient_user_prompt_template": DEFAULT_PATIENT_USER_PROMPT_TEMPLATE,
    }


def _merge_runtime_rows(rows: list[dict]) -> Dict[str, str]:
    values = _runtime_defaults()
    for row in rows:
        values[row["key"]] = row["value"]
    return values


def _latest_runtime_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda item: item.get("updated_at") or "")


def _load_runtime_rows() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT app_config.key, app_config.value, app_config.updated_at, app_config.updated_by, users.username AS updated_by_username
            FROM app_config
            LEFT JOIN users ON users.id = app_config.updated_by
            WHERE app_config.key IN ({runtime_keys_sql})
            """
            .format(runtime_keys_sql=RUNTIME_DB_KEYS_SQL)
        ).fetchall()
        return [dict(row) for row in rows]


def get_runtime_config() -> Dict[str, str]:
    return _merge_runtime_rows(_load_runtime_rows())


def get_runtime_config_view() -> Dict[str, str]:
    rows = _load_runtime_rows()
    values = _merge_runtime_rows(rows)
    latest_row = _latest_runtime_row(rows)
    return {
        **values,
        "openai_api_key_masked": mask_api_key(values.get("openai_api_key", "")),
        "last_updated_at": latest_row.get("updated_at", "-") if latest_row else "-",
        "last_updated_by": latest_row.get("updated_by_username", "-") if latest_row else "-",
    }


def set_runtime_config(updates: Dict[str, str], updated_by: int | None = None) -> None:
    if not updates:
        return
    now = now_kst_iso()
    with get_connection() as conn:
        actor_id = updated_by
        if actor_id is not None:
            row = conn.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (actor_id,)).fetchone()
            if row is None:
                actor_id = None
        for key, value in updates.items():
            if key not in RUNTIME_KEYS:
                continue
            conn.execute(
                """
                INSERT INTO app_config (key, value, updated_at, updated_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by
                """,
                (key, value, now, actor_id),
            )
        conn.commit()
