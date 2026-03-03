from __future__ import annotations

import re
from typing import Any, TypedDict


class PatientResponsePayload(TypedDict):
    empathy: str
    summary: str
    next_steps: list[str]
    safety_urgent: bool
    safety_actions: list[str]
    insufficient_data: bool
    internal_evidence: list[str]


ValidatedPatientResponse = PatientResponsePayload

REQUIRED_KEYS = {
    "empathy",
    "summary",
    "next_steps",
    "safety_urgent",
    "safety_actions",
    "insufficient_data",
    "internal_evidence",
}

FORBIDDEN_PATTERNS = (
    "[근거]",
    "insufficient_data",
    "schema_valid",
    "constraint_violations",
    "[QUALITY CHECK]",
    "[OUTPUT CONTRACT]",
    "output contract",
    "quality check",
    "데이터 라벨링",
    "데이터 라벨",
    "라벨링 대상",
    "데이터상",
    "스키마 유효",
    "계약상",
)

FORBIDDEN_REGEX = [re.compile(re.escape(pattern), re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS]
SANITIZER_WHITELIST_MARKERS = (
    "1393",
    "112",
    "자살예방상담전화",
    "도움",
    "안전",
)

_DEFAULT_FALLBACK_MESSAGE = "현재는 답변을 안정적으로 생성하지 못했습니다. 잠시 숨을 고르고, 필요한 내용을 한 문장으로 다시 알려주세요."
RESPONSE_STYLES = {"guided_steps", "validation_only", "urgent_plain"}
RESISTANCE_PATTERNS = (
    "리스트",
    "번호",
    "데이터 취급",
    "데이터취급",
    "매뉴얼",
    "포기",
    "하지 마",
    "하지마",
)


def _clean_text(value: Any, field_name: str, errors: list[str], max_len: int) -> str:
    if not isinstance(value, str):
        errors.append(f"{field_name}는 문자열이어야 합니다.")
        return ""
    cleaned = value.strip()
    if not cleaned:
        errors.append(f"{field_name}는 비워둘 수 없습니다.")
        return ""
    if len(cleaned) > max_len:
        errors.append(f"{field_name} 길이가 너무 깁니다.")
        return ""
    return cleaned


def _clean_string_list(
    value: Any,
    field_name: str,
    errors: list[str],
    min_items: int,
    max_items: int,
    item_max_len: int = 120,
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{field_name}는 문자열 리스트여야 합니다.")
        return []
    if not (min_items <= len(value) <= max_items):
        errors.append(f"{field_name} 개수는 {min_items}~{max_items}개여야 합니다.")
        return []

    cleaned_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            errors.append(f"{field_name} 항목은 문자열이어야 합니다.")
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        if len(cleaned) > item_max_len:
            errors.append(f"{field_name} 항목 길이가 너무 깁니다.")
            continue
        cleaned_items.append(cleaned)

    if len(cleaned_items) < min_items:
        errors.append(f"{field_name} 유효 항목 수가 부족합니다.")
    if len(cleaned_items) > max_items:
        cleaned_items = cleaned_items[:max_items]
    return cleaned_items


def validate_patient_payload(payload: object) -> tuple[ValidatedPatientResponse | None, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["payload는 JSON 객체여야 합니다."]

    payload_keys = set(payload.keys())
    missing = sorted(REQUIRED_KEYS - payload_keys)
    extra = sorted(payload_keys - REQUIRED_KEYS)
    if missing:
        errors.extend([f"필수 키 누락: {key}" for key in missing])
    if extra:
        errors.extend([f"허용되지 않은 키: {key}" for key in extra])
    if errors:
        return None, errors

    empathy = _clean_text(payload.get("empathy"), "empathy", errors, 280)
    summary = _clean_text(payload.get("summary"), "summary", errors, 360)
    next_steps = _clean_string_list(payload.get("next_steps"), "next_steps", errors, min_items=0, max_items=3)
    safety_actions = _clean_string_list(payload.get("safety_actions"), "safety_actions", errors, min_items=0, max_items=3)
    internal_evidence = _clean_string_list(payload.get("internal_evidence"), "internal_evidence", errors, min_items=0, max_items=3)

    safety_urgent = payload.get("safety_urgent")
    if not isinstance(safety_urgent, bool):
        errors.append("safety_urgent는 boolean이어야 합니다.")
        safety_urgent = False

    insufficient_data = payload.get("insufficient_data")
    if not isinstance(insufficient_data, bool):
        errors.append("insufficient_data는 boolean이어야 합니다.")
        insufficient_data = False

    if safety_urgent and len(safety_actions) == 0:
        errors.append("safety_urgent=true인 경우 safety_actions는 최소 1개 이상이어야 합니다.")

    if errors:
        return None, errors

    validated: ValidatedPatientResponse = {
        "empathy": empathy,
        "summary": summary,
        "next_steps": next_steps,
        "safety_urgent": safety_urgent,
        "safety_actions": safety_actions,
        "insufficient_data": insufficient_data,
        "internal_evidence": internal_evidence,
    }
    return validated, []


def _contains_resistance_pattern(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in RESISTANCE_PATTERNS)


def detect_response_resistance(latest_user_text: str, recent_user_texts: list[str]) -> int:
    recent = [item.strip() for item in (recent_user_texts or []) if isinstance(item, str) and item.strip()]
    latest = (latest_user_text or "").strip()
    samples = recent[-3:]
    if latest and (not samples or samples[-1] != latest):
        samples.append(latest)
    samples = samples[-3:]

    matched_count = sum(1 for text in samples if _contains_resistance_pattern(text))
    if matched_count >= 2:
        return 2
    if matched_count == 1:
        return 1
    return 0


def select_response_style(risk_score: float, risk_threshold: float, resistance_level: int) -> str:
    if float(risk_score) >= float(risk_threshold):
        return "urgent_plain"
    if int(resistance_level) >= 2:
        return "validation_only"
    return "guided_steps"


def _render_guided_steps(payload: ValidatedPatientResponse) -> list[str]:
    lines = [payload["empathy"], payload["summary"]]
    if payload["next_steps"]:
        lines.append("실행 가능한 다음 단계:")
        for step in payload["next_steps"]:
            lines.append(f"- {step}")
    else:
        lines.append("지금은 정답을 급히 찾기보다, 마음 상태를 한 문장으로 천천히 적어봐도 괜찮아요.")

    if payload["safety_urgent"] and payload["safety_actions"]:
        lines.append("지금 바로 안전을 위해:")
        for action in payload["safety_actions"]:
            lines.append(f"- {action}")

    if payload["insufficient_data"]:
        lines.append("추가 정보를 알려주시면 더 정확히 도와드릴 수 있어요.")
    return lines


def _render_validation_only(payload: ValidatedPatientResponse) -> list[str]:
    lines = [
        payload["empathy"],
        payload["summary"],
        "지금은 해결책을 밀어붙이기보다, 당신 마음을 먼저 함께 붙들고 싶어요.",
        "괜찮다면 지금 가장 크게 올라오는 감정을 한 단어로만 말해줄 수 있을까요?",
    ]
    return lines


def _render_urgent_plain(payload: ValidatedPatientResponse) -> list[str]:
    immediate_action = (
        payload["safety_actions"][0]
        if payload["safety_actions"]
        else "주변의 위험 물건을 잠시 멀리 두고 혼자 있지 않아 주세요."
    )
    lines = [
        payload["empathy"],
        "지금은 설명보다 안전을 먼저 지키는 것이 가장 중요해요.",
        f"가능하면 지금 바로 {immediate_action}",
        "신뢰하는 사람에게 현재 상태를 바로 알리고 곁에 있어 달라고 요청해 주세요.",
        "즉시 도움이 필요하면 1393(자살예방상담전화) 또는 112에 바로 연락해 주세요.",
    ]
    return lines


def render_patient_message(payload: ValidatedPatientResponse | None, style: str, is_urgent: bool) -> str:
    if payload is None:
        return build_patient_fallback_message(is_urgent=is_urgent)

    selected_style = style if style in RESPONSE_STYLES else "guided_steps"
    if is_urgent:
        selected_style = "urgent_plain"

    if selected_style == "urgent_plain":
        lines = _render_urgent_plain(payload)
    elif selected_style == "validation_only":
        lines = _render_validation_only(payload)
    else:
        lines = _render_guided_steps(payload)

    rendered = "\n".join(line.strip() for line in lines if line and line.strip())
    sanitized, _ = sanitize_patient_display_text(rendered)
    return sanitized or _DEFAULT_FALLBACK_MESSAGE


def sanitize_patient_display_text(text: str) -> tuple[str, bool]:
    if not text:
        return "", False

    original = text
    kept_lines: list[str] = []
    for line in original.splitlines():
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in SANITIZER_WHITELIST_MARKERS):
            kept_lines.append(line)
            continue
        if any(marker.lower() in lowered for marker in FORBIDDEN_PATTERNS):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    for pattern in FORBIDDEN_REGEX:
        cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        cleaned = _DEFAULT_FALLBACK_MESSAGE

    return cleaned, cleaned != original.strip()


def build_patient_fallback_message(is_urgent: bool = False) -> str:
    if is_urgent:
        return (
            "지금 많이 힘든 상태로 보여 걱정됩니다.\n"
            "지금은 안전을 우선으로 지켜 주세요.\n"
            "가능하면 주변의 위험 물건을 잠시 멀리 두고 혼자 있지 않아 주세요.\n"
            "신뢰하는 사람에게 지금 상태를 바로 알리고 곁에 있어 달라고 요청해 주세요.\n"
            "즉시 도움이 필요하면 1393(자살예방상담전화) 또는 112에 연락하세요."
        )
    return _DEFAULT_FALLBACK_MESSAGE
