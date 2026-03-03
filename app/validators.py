from datetime import date
import re
from typing import Optional, Tuple


GENDER_CHOICES = {"male", "female", "other", "unknown"}


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    return re.sub(r"\D", "", phone)


def validate_phone(phone: str) -> Tuple[bool, str, str]:
    normalized = normalize_phone(phone)
    if not normalized:
        return False, "휴대폰 번호를 입력하세요.", normalized
    if len(normalized) < 10 or len(normalized) > 11:
        return False, "휴대폰 번호 형식이 올바르지 않습니다.", normalized
    return True, "", normalized


def validate_pin(pin: str) -> Tuple[bool, str]:
    if not pin:
        return False, "PIN을 입력하세요."
    if not pin.isdigit():
        return False, "PIN은 숫자만 가능합니다."
    if len(pin) < 4 or len(pin) > 12:
        return False, "PIN은 4~12자리여야 합니다."
    return True, ""


def validate_username(username: str) -> Tuple[bool, str, str]:
    cleaned = (username or "").strip()
    if not cleaned:
        return False, "아이디를 입력하세요.", cleaned
    if any(ch.isspace() for ch in cleaned):
        return False, "아이디에는 공백을 사용할 수 없습니다.", cleaned
    if len(cleaned) < 3 or len(cleaned) > 32:
        return False, "아이디는 3~32자여야 합니다.", cleaned
    return True, "", cleaned


def validate_full_name(full_name: str) -> Tuple[bool, str, str]:
    cleaned = (full_name or "").strip()
    if not cleaned:
        return False, "이름을 입력하세요.", cleaned
    if "\n" in cleaned or "\r" in cleaned:
        return False, "이름에 줄바꿈을 사용할 수 없습니다.", cleaned
    if len(cleaned) > 50:
        return False, "이름은 50자 이하여야 합니다.", cleaned
    return True, "", cleaned


def validate_age(age: str) -> Tuple[bool, str, Optional[int]]:
    cleaned = (age or "").strip()
    if not cleaned:
        return False, "나이를 입력하세요.", None
    if not cleaned.isdigit():
        return False, "나이는 숫자만 입력하세요.", None
    age_value = int(cleaned)
    if age_value < 1 or age_value > 120:
        return False, "나이는 1~120 범위여야 합니다.", None
    return True, "", age_value


def validate_gender(gender: str) -> Tuple[bool, str, str]:
    cleaned = (gender or "").strip().lower()
    if cleaned not in GENDER_CHOICES:
        return False, "성별 값이 올바르지 않습니다.", "unknown"
    return True, "", cleaned


def validate_residence(residence: str) -> Tuple[bool, str, str]:
    cleaned = (residence or "").strip()
    if len(cleaned) > 120:
        return False, "거주지는 120자 이하여야 합니다.", cleaned[:120]
    return True, "", cleaned


def validate_assistant_guidance(guidance: str) -> Tuple[bool, str, str]:
    cleaned = (guidance or "").strip()
    if len(cleaned) > 1000:
        return False, "도우미 유의 사항은 1000자 이하여야 합니다.", cleaned[:1000]
    return True, "", cleaned


def validate_note_date(note_date: object) -> Tuple[bool, str, str]:
    cleaned = note_date.strip() if isinstance(note_date, str) else ""
    if not cleaned:
        return True, "", ""
    try:
        date.fromisoformat(cleaned)
    except ValueError:
        return False, "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)", ""
    return True, "", cleaned


def validate_counselor_risk_level(level: str) -> Tuple[bool, str, Optional[float]]:
    cleaned = (level or "").strip()
    if not cleaned:
        return True, "", None
    try:
        value = float(cleaned)
    except ValueError:
        return False, "상담 위험수치는 숫자여야 합니다.", None
    clamped = max(0.0, min(1.0, value))
    return True, "", round(clamped, 4)
