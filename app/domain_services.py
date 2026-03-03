from __future__ import annotations

import csv
import io
import json
import logging
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.middleware.sessions import SessionMiddleware

from app.analysis import analyze_emotions, analyze_risk
from app.auth import (
    create_user,
    get_user_by_id,
    get_user_by_phone,
    get_user_by_username,
    has_any_user,
    list_users,
    verify_pin,
)
from app.emotions import EMOTIONS, EMOTION_ANGLES_DEG
from app.db import get_connection, init_db
from app.openai_client import chat_json, list_available_models
from app.patient_response import (
    FORBIDDEN_PATTERNS,
    build_patient_fallback_message,
    detect_response_resistance,
    render_patient_message,
    sanitize_patient_display_text,
    select_response_style,
    validate_patient_payload,
)
from app.rag import (
    PDF_NAMES,
    ensure_upload_source_dir,
    index_message_embedding,
    reindex_documents,
    search_rag,
)
from app.runtime_config import (
    get_runtime_config,
    get_runtime_config_view,
    set_runtime_config,
    validate_patient_prompt_template,
    validate_prompt_template,
)
from app.settings import settings
from app.time_utils import KST, now_kst_iso
from app.validators import (
    normalize_phone,
    validate_age,
    validate_counselor_risk_level,
    validate_full_name,
    validate_gender,
    validate_phone,
    validate_pin,
    validate_residence,
    validate_assistant_guidance,
    validate_username,
    validate_note_date,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield

templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

CSRF_SESSION_KEY = "_csrf_token"
CSRF_SALT = "therapy-assist-csrf-v1"


def _csrf_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt=CSRF_SALT)


def issue_csrf_token(request: Request) -> str:
    raw_token = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(raw_token, str) or not raw_token:
        raw_token = secrets.token_urlsafe(32)
        request.session[CSRF_SESSION_KEY] = raw_token
    return _csrf_serializer().dumps(raw_token)


def verify_csrf_token(request: Request, token: str | None) -> bool:
    if not settings.csrf_enforce:
        return True
    if not token:
        return False
    try:
        payload = _csrf_serializer().loads(token, max_age=settings.csrf_ttl_seconds)
    except (BadSignature, SignatureExpired):
        return False
    session_raw = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(session_raw, str) or not session_raw:
        return False
    return secrets.compare_digest(str(payload), session_raw)


def _csrf_forbidden_response(request: Request):
    message = "보안 토큰이 없거나 만료되었습니다. 페이지를 새로고침 후 다시 시도하세요."
    if request.headers.get("HX-Request"):
        return HTMLResponse(f'<div class="notice">{message}</div>', status_code=403)
    return PlainTextResponse(message, status_code=403)


def enforce_csrf(request: Request, csrf_token: str | None = None):
    # Unit tests directly call route functions with DummyRequest(session/headers only).
    if not hasattr(request, "scope"):
        return None

    token = (request.headers.get("X-CSRF-Token") or "").strip()
    if not token:
        token = (csrf_token or "").strip()

    if verify_csrf_token(request, token):
        return None
    return _csrf_forbidden_response(request)


templates.env.globals["csrf_token_for"] = issue_csrf_token


def get_current_user(request: Request) -> Optional[dict]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(int(user_id))


def require_role(user: Optional[dict], role: str) -> bool:
    return user is not None and user.get("role") == role


AI_CONSENT_VERSION = "v1"
AI_CONSENT_TEXT = (
    "본인은 상담 보조를 위한 AI 시스템 사용에 대해 충분히 안내받았으며, "
    "대화 내용이 로컬 시스템에 기록되고 AI 분석에 사용될 수 있음을 이해하고 동의합니다."
)

GENDER_LABELS = {
    "male": "남성",
    "female": "여성",
    "other": "기타",
    "unknown": "미입력",
}

OPPOSITE_EMOTION_PAIRS = [
    ("joy", "sadness"),
    ("trust", "disgust"),
    ("fear", "anger"),
    ("surprise", "anticipation"),
]

EMOTION_LABELS_KO = {
    "joy": "기쁨",
    "trust": "신뢰",
    "fear": "두려움",
    "surprise": "놀람",
    "sadness": "슬픔",
    "disgust": "혐오",
    "anger": "분노",
    "anticipation": "기대",
}

NEGATIVE_EMOTIONS = ("anger", "fear", "sadness")
EWMA_ALPHA = 0.35
VOLATILITY_WINDOW = 5
DOMINANT_SWITCH_MARGIN = 0.15
DOMINANT_SWITCH_STREAK = 2
ACUTE_NEGATIVE_THRESHOLD = 0.65
ACUTE_NEGATIVE_DELTA = 0.15


def _parse_iso_kst(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _normalize_emotion_profile(profile: dict[str, float] | None) -> dict[str, float]:
    normalized = {emotion: 0.0 for emotion in EMOTIONS}
    source = profile or {}
    for emotion in EMOTIONS:
        normalized[emotion] = round(max(0.0, min(1.0, _safe_float(source.get(emotion)) or 0.0)), 4)
    return normalized


def _top_emotion_from_profile(profile: dict[str, float] | None) -> tuple[str | None, float]:
    normalized = _normalize_emotion_profile(profile)
    top_name = None
    top_value = -1.0
    for emotion in EMOTIONS:
        score = normalized[emotion]
        if score > top_value:
            top_name = emotion
            top_value = score
    if top_name is None or top_value <= 0.0:
        return None, 0.0
    return top_name, round(top_value, 4)


def emotion_label_ko(emotion_key: str | None) -> str:
    if not emotion_key:
        return "-"
    return EMOTION_LABELS_KO.get(str(emotion_key), str(emotion_key))


def _emotion_label_ko_with_profile(emotion_key: str | None, profile: dict[str, float] | None) -> str:
    if not emotion_key:
        return "-"
    normalized = _normalize_emotion_profile(profile)
    key = str(emotion_key)
    if key == "anticipation":
        negative_load = max(normalized.get("anger", 0.0), normalized.get("fear", 0.0), normalized.get("sadness", 0.0))
        positive_load = max(normalized.get("joy", 0.0), normalized.get("trust", 0.0))
        if negative_load >= max(0.2, positive_load):
            return "긴장/예상"
    return emotion_label_ko(key)


def _build_emotion_label_map(profile: dict[str, float]) -> dict[str, str]:
    normalized = _normalize_emotion_profile(profile)
    return {emotion: _emotion_label_ko_with_profile(emotion, normalized) for emotion in EMOTIONS}


def _format_decimal(value: object, digits: int = 2) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{max(0, int(digits))}f}"


def _format_datetime_kst(value: str | None) -> str:
    parsed = _parse_iso_kst(value)
    if parsed is None:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M")


def _build_emotion_profile_ko(profile: dict[str, float]) -> list[dict[str, str]]:
    normalized = _normalize_emotion_profile(profile)
    labels = _build_emotion_label_map(normalized)
    items: list[dict[str, str]] = []
    for emotion in EMOTIONS:
        value = normalized.get(emotion) or 0.0
        items.append(
            {
                "key": emotion,
                "label": labels.get(emotion, emotion_label_ko(emotion)),
                "value_display": _format_decimal(value, digits=2),
            }
        )
    return items


def _extract_top_emotion(emotions_json: str | None) -> str | None:
    if not emotions_json:
        return None
    try:
        payload = json.loads(emotions_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload:
        return None

    top_name = None
    top_value = -1.0
    for key, value in payload.items():
        score = _safe_float(value)
        if score is None:
            continue
        if score > top_value:
            top_name = str(key)
            top_value = score
    if top_name is None or top_value <= 0.0:
        return None
    return top_name


def _extract_emotion_profile(emotions_json: str | None) -> dict[str, float]:
    profile = {emotion: 0.0 for emotion in EMOTIONS}
    if not emotions_json:
        return profile
    try:
        payload = json.loads(emotions_json)
    except json.JSONDecodeError:
        return profile
    if not isinstance(payload, dict):
        return profile
    for emotion in EMOTIONS:
        value = _safe_float(payload.get(emotion))
        if value is None:
            continue
        profile[emotion] = round(max(0.0, min(1.0, value)), 4)
    return profile


def _preview_text(content: str, max_len: int = 120) -> str:
    cleaned = (content or "").replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def _risk_band(score: float | None) -> str:
    if score is None:
        return "데이터 없음"
    if score < 0.4:
        return "안정 구간"
    if score < 0.7:
        return "주의 구간"
    return "고위험 구간"


def _energy_band(value: float | None) -> str:
    if value is None:
        return "데이터 없음"
    if value < 0.25:
        return "낮은 활성"
    if value < 0.6:
        return "중간 활성"
    return "높은 활성"


def _circular_gap_deg(source_deg: float, target_deg: float) -> float:
    diff = abs((source_deg - target_deg) % 360.0)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def compute_emotion_alignment(
    profile: dict[str, float],
    energy_angle: float | None,
    angle_reliable: bool,
    coherence: float | None,
    dominant_emotion_override: str | None = None,
) -> dict[str, object]:
    normalized = _normalize_emotion_profile(profile)

    dominant_emotion = EMOTIONS[0] if EMOTIONS else ""
    dominant_score = -1.0
    if dominant_emotion_override in EMOTIONS:
        dominant_emotion = str(dominant_emotion_override)
        dominant_score = normalized.get(dominant_emotion, 0.0)
    else:
        for emotion in EMOTIONS:
            score = normalized[emotion]
            if score > dominant_score:
                dominant_score = score
                dominant_emotion = emotion
    dominant_angle = float(EMOTION_ANGLES_DEG.get(dominant_emotion, 0.0))

    angle_component = 0.0
    angle_gap_deg = None
    if angle_reliable and energy_angle is not None:
        energy_angle_value = float(energy_angle) % 360.0
        angle_gap = _circular_gap_deg(energy_angle_value, dominant_angle)
        angle_gap_deg = round(angle_gap, 2)
        angle_component = max(0.0, min(1.0, 1.0 - (angle_gap / 180.0)))

    coherence_value = max(0.0, min(1.0, _safe_float(coherence) or 0.0))
    opposite_tension = 0.0
    for left, right in OPPOSITE_EMOTION_PAIRS:
        opposite_tension += min(normalized[left], normalized[right])
    opposite_tension = opposite_tension / len(OPPOSITE_EMOTION_PAIRS)
    opposite_tension = round(max(0.0, min(1.0, opposite_tension)), 4)
    tension_component = max(0.0, min(1.0, 1.0 - opposite_tension))

    consistency_score = round(
        100.0 * (0.45 * angle_component + 0.35 * coherence_value + 0.20 * tension_component),
        1,
    )
    if not angle_reliable:
        consistency_band = "판단불가"
    elif consistency_score >= 70.0:
        consistency_band = "높음"
    elif consistency_score >= 40.0:
        consistency_band = "보통"
    else:
        consistency_band = "낮음"

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_label_ko": _emotion_label_ko_with_profile(dominant_emotion, normalized),
        "dominant_score": round(max(0.0, dominant_score), 4),
        "dominant_angle": round(dominant_angle, 2),
        "angle_gap_deg": angle_gap_deg,
        "opposite_tension": opposite_tension,
        "consistency_score": consistency_score,
        "consistency_band": consistency_band,
    }


def _compute_state_profiles_ewma(profiles: list[dict[str, float]], alpha: float = EWMA_ALPHA) -> list[dict[str, float]]:
    if not profiles:
        return []
    alpha_value = max(0.05, min(0.95, float(alpha)))
    state = _normalize_emotion_profile(profiles[0])
    states = [dict(state)]
    for raw_profile in profiles[1:]:
        current = _normalize_emotion_profile(raw_profile)
        for emotion in EMOTIONS:
            state[emotion] = round(
                alpha_value * current[emotion] + (1.0 - alpha_value) * state[emotion],
                4,
            )
        states.append(dict(state))
    return states


def _select_dominant_emotion_with_hysteresis(
    state_profiles: list[dict[str, float]],
    margin: float = DOMINANT_SWITCH_MARGIN,
    streak_required: int = DOMINANT_SWITCH_STREAK,
) -> str | None:
    if not state_profiles:
        return None
    current, _ = _top_emotion_from_profile(state_profiles[0])
    candidate = current
    streak = 0
    margin_value = max(0.0, float(margin))
    required = max(1, int(streak_required))

    for profile in state_profiles[1:]:
        top_name, top_score = _top_emotion_from_profile(profile)
        if not top_name:
            continue
        if current is None:
            current = top_name
            candidate = top_name
            streak = 0
            continue
        current_score = _safe_float(profile.get(current)) or 0.0
        if top_name != current and top_score >= (current_score + margin_value):
            if candidate == top_name:
                streak += 1
            else:
                candidate = top_name
                streak = 1
            if streak >= required:
                current = top_name
                candidate = top_name
                streak = 0
        else:
            candidate = current
            streak = 0
    return current


def _compute_emotion_volatility(profiles: list[dict[str, float]], window: int = VOLATILITY_WINDOW) -> float:
    if len(profiles) < 2:
        return 0.0
    normalized = [_normalize_emotion_profile(profile) for profile in profiles]
    recent = normalized[-max(2, int(window)) :]
    deltas: list[float] = []
    for previous, current in zip(recent, recent[1:]):
        mean_abs_delta = sum(abs(current[emotion] - previous[emotion]) for emotion in EMOTIONS) / max(1, len(EMOTIONS))
        deltas.append(mean_abs_delta)
    if not deltas:
        return 0.0
    return round(sum(deltas) / len(deltas), 4)


def _emotion_volatility_band(volatility: float) -> str:
    value = max(0.0, _safe_float(volatility) or 0.0)
    if value < 0.05:
        return "낮음"
    if value < 0.12:
        return "중간"
    return "높음"


def _detect_acute_negative_override(
    snapshot_profile: dict[str, float],
    state_profile: dict[str, float],
    latest_risk: float | None,
) -> dict[str, object]:
    snapshot = _normalize_emotion_profile(snapshot_profile)
    state = _normalize_emotion_profile(state_profile)
    negative_emotion = None
    negative_score = -1.0
    for emotion in NEGATIVE_EMOTIONS:
        score = snapshot.get(emotion, 0.0)
        if score > negative_score:
            negative_emotion = emotion
            negative_score = score

    if not negative_emotion or negative_score <= 0.0:
        return {"applied": False, "emotion": None, "score": 0.0}

    state_score = state.get(negative_emotion, 0.0)
    risk_trigger = latest_risk is not None and float(latest_risk) >= settings.risk_threshold
    spike_trigger = negative_score >= ACUTE_NEGATIVE_THRESHOLD and (negative_score - state_score) >= ACUTE_NEGATIVE_DELTA
    applied = bool(risk_trigger or spike_trigger)
    return {
        "applied": applied,
        "emotion": negative_emotion if applied else None,
        "score": round(negative_score, 4) if applied else 0.0,
    }


def group_notes_by_date(notes: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for note in notes:
        created = _parse_iso_kst(note.get("created_at"))
        date_key = created.date().isoformat() if created else "날짜 미상"
        if date_key not in groups:
            groups[date_key] = []
            order.append(date_key)
        item = dict(note)
        item["time_label"] = created.strftime("%H:%M") if created else "-"
        groups[date_key].append(item)
    return [{"date": key, "items": groups[key]} for key in order]


def _build_note_created_at_iso(note_date: str) -> str:
    target_date = datetime.fromisoformat(f"{note_date}T00:00:00").date()
    now = datetime.now(tz=KST)
    return datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
        microsecond=now.microsecond,
        tzinfo=KST,
    ).isoformat()


def get_patient_profile(user_id: int) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT full_name, age, gender, residence, assistant_guidance, created_at, updated_at
            FROM patient_profiles
            WHERE user_id = ?
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

    if not row:
        return {
            "full_name": "미입력",
            "age": None,
            "gender": "unknown",
            "gender_label": GENDER_LABELS["unknown"],
            "residence": "",
            "assistant_guidance": "",
            "created_at": None,
            "updated_at": None,
            "profile_missing": True,
        }
    profile = dict(row)
    profile["gender"] = profile.get("gender") or "unknown"
    profile["gender_label"] = GENDER_LABELS.get(profile["gender"], "미입력")
    profile["residence"] = profile.get("residence") or ""
    profile["assistant_guidance"] = profile.get("assistant_guidance") or ""
    profile["profile_missing"] = False
    return profile


def upsert_patient_profile(
    user_id: int,
    full_name: str,
    age: int,
    gender: str,
    residence: str,
    assistant_guidance: str,
    updated_by: int | None,
) -> None:
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_profiles (
                user_id, full_name, age, gender, residence, assistant_guidance, created_at, updated_at, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                full_name = excluded.full_name,
                age = excluded.age,
                gender = excluded.gender,
                residence = excluded.residence,
                assistant_guidance = excluded.assistant_guidance,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (user_id, full_name, age, gender, residence, assistant_guidance, now, now, updated_by),
        )
        conn.commit()


def add_counselor_note(
    patient_user_id: int,
    counselor_user_id: int,
    session_summary: str,
    intervention_note: str,
    followup_plan: str,
    counselor_risk_level: float | None,
    created_at: str | None = None,
) -> int:
    now = now_kst_iso()
    created_at_value = created_at or now
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO counselor_notes (
                patient_user_id, counselor_user_id, session_summary, intervention_note, followup_plan,
                counselor_risk_level, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_user_id,
                counselor_user_id,
                session_summary,
                intervention_note,
                followup_plan,
                counselor_risk_level,
                created_at_value,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_counselor_notes(patient_user_id: int, limit: int = 50, note_date: str | None = None) -> list[dict]:
    params: list[object] = [patient_user_id]
    date_clause = ""
    if note_date:
        date_clause = " AND substr(counselor_notes.created_at, 1, 10) = ?"
        params.append(note_date)
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT counselor_notes.id,
                   counselor_notes.patient_user_id,
                   counselor_notes.counselor_user_id,
                   users.username AS counselor_username,
                   counselor_notes.session_summary,
                   counselor_notes.intervention_note,
                   counselor_notes.followup_plan,
                   counselor_notes.counselor_risk_level,
                   counselor_notes.created_at,
                   counselor_notes.updated_at
            FROM counselor_notes
            JOIN users ON users.id = counselor_notes.counselor_user_id
            WHERE counselor_notes.patient_user_id = ?
              """ + date_clause + """
            ORDER BY counselor_notes.created_at DESC, counselor_notes.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def build_patient_dashboard(user_id: int, timeline_limit: int = 30, day_limit: int = 7) -> dict:
    safe_timeline_limit = max(1, min(200, int(timeline_limit)))
    safe_day_limit = max(1, min(30, int(day_limit)))

    with get_connection() as conn:
        timeline_rows = conn.execute(
            """
            SELECT messages.id, messages.content, messages.created_at,
                   risk_flags.score AS risk_score,
                   emotion_scores.energy_magnitude,
                   emotion_scores.energy_angle,
                   emotion_scores.energy_coherence,
                   emotion_scores.energy_angle_reliable,
                   emotion_scores.energy_active_count,
                   emotion_scores.energy_status,
                   emotion_scores.emotions_json
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            LEFT JOIN risk_flags ON risk_flags.message_id = messages.id
            LEFT JOIN emotion_scores ON emotion_scores.message_id = messages.id
            WHERE sessions.user_id = ? AND messages.role = 'user'
            ORDER BY messages.id DESC
            LIMIT ?
            """,
            (user_id, safe_timeline_limit),
        ).fetchall()

        daily_rows = conn.execute(
            """
            SELECT messages.id, messages.content, messages.created_at,
                   risk_flags.score AS risk_score,
                   emotion_scores.energy_magnitude
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            LEFT JOIN risk_flags ON risk_flags.message_id = messages.id
            LEFT JOIN emotion_scores ON emotion_scores.message_id = messages.id
            WHERE sessions.user_id = ? AND messages.role = 'user'
            ORDER BY messages.id DESC
            LIMIT 1000
            """,
            (user_id,),
        ).fetchall()

    timeline_points = []
    for row in reversed(timeline_rows):
        top_emotion = _extract_top_emotion(row["emotions_json"])
        emotion_profile = _extract_emotion_profile(row["emotions_json"])
        risk_score = _safe_float(row["risk_score"])
        energy_magnitude = _safe_float(row["energy_magnitude"])
        energy_angle = _safe_float(row["energy_angle"])
        timeline_points.append(
            {
                "message_id": row["id"],
                "created_at": row["created_at"],
                "created_at_display": _format_datetime_kst(row["created_at"]),
                "content_preview": _preview_text(row["content"], max_len=120),
                "risk_score": risk_score,
                "risk_score_display": _format_decimal(risk_score),
                "energy_magnitude": energy_magnitude,
                "energy_magnitude_display": _format_decimal(energy_magnitude),
                "energy_angle": energy_angle,
                "energy_angle_display": _format_decimal(energy_angle),
                "energy_coherence": _safe_float(row["energy_coherence"]),
                "energy_angle_reliable": _safe_bool(row["energy_angle_reliable"]),
                "energy_active_count": int(row["energy_active_count"] or 0),
                "energy_status": str(row["energy_status"] or "zero_input"),
                "top_emotion": top_emotion,
                "top_emotion_label_ko": _emotion_label_ko_with_profile(top_emotion, emotion_profile),
                "emotion_profile": emotion_profile,
            }
        )

    now_dt = datetime.now(tz=KST)
    date_keys = []
    daily_bucket: dict[str, dict[str, float | int]] = {}
    for offset in range(safe_day_limit - 1, -1, -1):
        day = (now_dt - timedelta(days=offset)).date()
        key = day.isoformat()
        date_keys.append(key)
        daily_bucket[key] = {
            "message_count": 0,
            "risk_sum": 0.0,
            "risk_count": 0,
            "energy_sum": 0.0,
            "energy_count": 0,
            "msg_len_sum": 0,
        }

    for row in daily_rows:
        created = _parse_iso_kst(row["created_at"])
        if created is None:
            continue
        key = created.date().isoformat()
        if key not in daily_bucket:
            continue
        item = daily_bucket[key]
        item["message_count"] = int(item["message_count"]) + 1
        item["msg_len_sum"] = int(item["msg_len_sum"]) + len((row["content"] or "").strip())

        risk_score = _safe_float(row["risk_score"])
        if risk_score is not None:
            item["risk_sum"] = float(item["risk_sum"]) + risk_score
            item["risk_count"] = int(item["risk_count"]) + 1

        energy_value = _safe_float(row["energy_magnitude"])
        if energy_value is not None:
            item["energy_sum"] = float(item["energy_sum"]) + energy_value
            item["energy_count"] = int(item["energy_count"]) + 1

    daily_points = []
    for key in date_keys:
        item = daily_bucket[key]
        msg_count = int(item["message_count"])
        risk_count = int(item["risk_count"])
        energy_count = int(item["energy_count"])
        avg_risk_score = round(float(item["risk_sum"]) / risk_count, 4) if risk_count > 0 else None
        avg_energy_magnitude = round(float(item["energy_sum"]) / energy_count, 4) if energy_count > 0 else None
        daily_points.append(
            {
                "date": key,
                "message_count": msg_count,
                "avg_risk_score": avg_risk_score,
                "avg_risk_score_display": _format_decimal(avg_risk_score),
                "avg_energy_magnitude": avg_energy_magnitude,
                "avg_energy_magnitude_display": _format_decimal(avg_energy_magnitude),
                "avg_msg_length": round(int(item["msg_len_sum"]) / msg_count, 2) if msg_count > 0 else 0.0,
            }
        )

    latest_risk = None
    latest_energy = {
        "magnitude": None,
        "angle": None,
        "coherence": 0.0,
        "angle_reliable": False,
        "status": "zero_input",
        "active_count": 0,
    }
    last_message_at = None
    latest_emotion_profile = {emotion: 0.0 for emotion in EMOTIONS}
    latest_snapshot_profile = {emotion: 0.0 for emotion in EMOTIONS}
    latest_emotion_alignment = compute_emotion_alignment(latest_emotion_profile, None, False, 0.0)
    latest_emotion_alignment["acute_override_applied"] = False
    latest_emotion_alignment["snapshot_negative_emotion"] = None
    latest_emotion_alignment["snapshot_negative_score"] = 0.0
    dominant_emotion_source = "state_top"
    emotion_volatility = 0.0
    emotion_volatility_band = "낮음"
    if timeline_points:
        last_item = timeline_points[-1]
        latest_risk = last_item.get("risk_score")
        latest_energy = {
            "magnitude": last_item.get("energy_magnitude"),
            "angle": last_item.get("energy_angle"),
            "coherence": round(max(0.0, min(1.0, _safe_float(last_item.get("energy_coherence")) or 0.0)), 4),
            "angle_reliable": bool(last_item.get("energy_angle_reliable")),
            "status": str(last_item.get("energy_status") or "zero_input"),
            "active_count": int(last_item.get("energy_active_count") or 0),
        }
        last_message_at = last_item.get("created_at")
        latest_snapshot_profile = _normalize_emotion_profile(last_item.get("emotion_profile"))

        raw_profiles = [point.get("emotion_profile") or {} for point in timeline_points]
        state_profiles = _compute_state_profiles_ewma(raw_profiles, alpha=EWMA_ALPHA)
        if state_profiles:
            latest_emotion_profile = dict(state_profiles[-1])

        emotion_volatility = _compute_emotion_volatility(raw_profiles, window=VOLATILITY_WINDOW)
        emotion_volatility_band = _emotion_volatility_band(emotion_volatility)

        dominant_override = _select_dominant_emotion_with_hysteresis(
            state_profiles,
            margin=DOMINANT_SWITCH_MARGIN,
            streak_required=DOMINANT_SWITCH_STREAK,
        )
        if dominant_override:
            dominant_emotion_source = "state_hysteresis"

        acute_override = _detect_acute_negative_override(
            latest_snapshot_profile,
            latest_emotion_profile,
            latest_risk,
        )
        if acute_override["applied"]:
            dominant_override = str(acute_override["emotion"])
            dominant_emotion_source = "acute_override"

        latest_emotion_alignment = compute_emotion_alignment(
            latest_emotion_profile,
            latest_energy.get("angle"),
            bool(latest_energy.get("angle_reliable")),
            _safe_float(latest_energy.get("coherence")),
            dominant_emotion_override=dominant_override,
        )
        latest_emotion_alignment["acute_override_applied"] = bool(acute_override["applied"])
        latest_emotion_alignment["snapshot_negative_emotion"] = acute_override["emotion"]
        latest_emotion_alignment["snapshot_negative_score"] = acute_override["score"]
        latest_emotion_alignment["dominant_emotion_source"] = dominant_emotion_source

    return {
        "timeline_points": timeline_points,
        "daily_points": daily_points,
        "latest_risk": latest_risk,
        "latest_risk_display": _format_decimal(latest_risk),
        "latest_energy": latest_energy,
        "latest_energy_magnitude": latest_energy.get("magnitude"),
        "latest_energy_magnitude_display": _format_decimal(latest_energy.get("magnitude")),
        "last_message_at": last_message_at,
        "last_message_at_display": _format_datetime_kst(last_message_at),
        "latest_emotion_profile": latest_emotion_profile,
        "latest_emotion_profile_ko": _build_emotion_profile_ko(latest_emotion_profile),
        "latest_emotion_labels_ko": _build_emotion_label_map(latest_emotion_profile),
        "latest_emotion_snapshot_profile": latest_snapshot_profile,
        "latest_emotion_snapshot_profile_ko": _build_emotion_profile_ko(latest_snapshot_profile),
        "latest_emotion_alignment": latest_emotion_alignment,
        "dominant_emotion_source": dominant_emotion_source,
        "emotion_volatility": emotion_volatility,
        "emotion_volatility_display": _format_decimal(emotion_volatility),
        "emotion_volatility_band": emotion_volatility_band,
        "risk_band": _risk_band(latest_risk),
        "energy_band": _energy_band(_safe_float(latest_energy.get("magnitude"))),
        "kpi_explanations": {
            "risk": "위험 점수는 0~1 범위입니다. 0.7 이상은 즉시 확인이 필요한 고위험 구간입니다.",
            "energy": "에너지 크기는 현재 정서 활성도의 크기입니다. 클수록 감정 반응이 강합니다.",
            "angle": "에너지 각도는 joy=0° 기준(CCW+)입니다. 방향 신뢰 불가 시 해석을 보류합니다.",
        },
    }


def ensure_patient_consent_row(user_id: int) -> None:
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO patient_ai_consents (
                user_id, consent_given, consent_version, consent_text, created_at, updated_at
            ) VALUES (?, 0, ?, ?, ?, ?)
            """,
            (user_id, AI_CONSENT_VERSION, AI_CONSENT_TEXT, now, now),
        )
        conn.commit()


def has_patient_ai_consent(user_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT consent_given
            FROM patient_ai_consents
            WHERE user_id = ?
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        return bool(row and int(row["consent_given"]) == 1)


def get_patient_ai_consent(user_id: int) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT consent_given, consent_version, consent_text, agreed_at, revoked_at, updated_at
            FROM patient_ai_consents
            WHERE user_id = ?
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if row:
            return dict(row)
    return {
        "consent_given": 0,
        "consent_version": AI_CONSENT_VERSION,
        "consent_text": AI_CONSENT_TEXT,
        "agreed_at": None,
        "revoked_at": None,
        "updated_at": None,
    }


def set_patient_ai_consent(user_id: int, consent_given: bool, updated_by: Optional[int]) -> None:
    now = now_kst_iso()
    ensure_patient_consent_row(user_id)
    with get_connection() as conn:
        if consent_given:
            conn.execute(
                """
                UPDATE patient_ai_consents
                SET consent_given = 1,
                    consent_version = ?,
                    consent_text = ?,
                    agreed_at = ?,
                    revoked_at = NULL,
                    updated_at = ?,
                    updated_by = ?
                WHERE user_id = ?
                """,
                (AI_CONSENT_VERSION, AI_CONSENT_TEXT, now, now, updated_by, user_id),
            )
        else:
            conn.execute(
                """
                UPDATE patient_ai_consents
                SET consent_given = 0,
                    revoked_at = ?,
                    updated_at = ?,
                    updated_by = ?
                WHERE user_id = ?
                """,
                (now, now, updated_by, user_id),
            )
        conn.commit()


def list_patient_ai_consents() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT users.id AS user_id,
                   users.username,
                   users.phone,
                   users.created_at,
                   COALESCE(patient_ai_consents.consent_given, 0) AS consent_given,
                   patient_ai_consents.consent_version,
                   patient_ai_consents.agreed_at,
                   patient_ai_consents.revoked_at,
                   patient_ai_consents.updated_at
            FROM users
            LEFT JOIN patient_ai_consents ON patient_ai_consents.user_id = users.id
            WHERE users.role = 'patient'
            ORDER BY users.created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def assign_patient_to_counselor(counselor_user_id: int, patient_user_id: int) -> tuple[bool, str]:
    now = now_kst_iso()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT counselor_user_id FROM counselor_patients WHERE patient_user_id = ?",
            (patient_user_id,),
        ).fetchone()
        if existing and int(existing["counselor_user_id"]) != counselor_user_id:
            return False, "이미 다른 상담자에게 배정된 환자입니다."

        conn.execute(
            """
            INSERT OR IGNORE INTO counselor_patients (counselor_user_id, patient_user_id, assigned_at)
            VALUES (?, ?, ?)
            """,
            (counselor_user_id, patient_user_id, now),
        )
        conn.commit()
    return True, ""


def list_counselor_patients(counselor_user_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT users.id,
                   users.username,
                   users.role,
                   users.phone,
                   users.created_at,
                   patient_profiles.full_name,
                   patient_profiles.age,
                   COALESCE(patient_profiles.gender, 'unknown') AS gender,
                   COALESCE(patient_profiles.residence, '') AS residence,
                   COUNT(messages.id) AS message_count,
                   MAX(messages.created_at) AS last_message_at,
                   (
                       SELECT risk_flags.score
                       FROM risk_flags
                       JOIN messages AS risk_messages ON risk_messages.id = risk_flags.message_id
                       JOIN sessions AS risk_sessions ON risk_sessions.id = risk_messages.session_id
                       WHERE risk_sessions.user_id = users.id
                       ORDER BY risk_flags.id DESC
                       LIMIT 1
                   ) AS latest_risk_score,
                   (
                       SELECT emotion_scores.energy_magnitude
                       FROM emotion_scores
                       JOIN messages AS energy_messages ON energy_messages.id = emotion_scores.message_id
                       JOIN sessions AS energy_sessions ON energy_sessions.id = energy_messages.session_id
                       WHERE energy_sessions.user_id = users.id
                       ORDER BY emotion_scores.id DESC
                       LIMIT 1
                   ) AS latest_energy_magnitude
            FROM counselor_patients
            JOIN users ON users.id = counselor_patients.patient_user_id
            LEFT JOIN patient_profiles ON patient_profiles.user_id = users.id
            LEFT JOIN sessions ON sessions.user_id = users.id
            LEFT JOIN messages ON messages.session_id = sessions.id
            WHERE counselor_patients.counselor_user_id = ? AND users.role = 'patient'
            GROUP BY users.id, users.username, users.role, users.phone, users.created_at,
                     patient_profiles.full_name, patient_profiles.age, patient_profiles.gender, patient_profiles.residence
            ORDER BY users.created_at DESC
            """,
            (counselor_user_id,),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["full_name"] = item.get("full_name") or "미입력"
            item["gender_label"] = GENDER_LABELS.get(item.get("gender") or "unknown", "미입력")
            item["latest_risk_score_display"] = _format_decimal(item.get("latest_risk_score"))
            item["latest_energy_magnitude_display"] = _format_decimal(item.get("latest_energy_magnitude"))
            item["last_message_at_display"] = _format_datetime_kst(item.get("last_message_at"))
            items.append(item)
        return items


def counselor_can_view_patient(counselor_user_id: int, patient_user_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM counselor_patients
            WHERE counselor_user_id = ? AND patient_user_id = ?
            LIMIT 1
            """,
            (counselor_user_id, patient_user_id),
        ).fetchone()
        return row is not None


def delete_user_account(user_id: int) -> bool:
    with get_connection() as conn:
        target = conn.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not target:
            return False

        conn.execute("UPDATE copilot_threads SET selected_patient_id = NULL WHERE selected_patient_id = ?", (user_id,))

        thread_rows = conn.execute(
            "SELECT id FROM copilot_threads WHERE counselor_user_id = ?",
            (user_id,),
        ).fetchall()
        thread_ids = [int(row["id"]) for row in thread_rows]
        if thread_ids:
            placeholders = ",".join("?" for _ in thread_ids)
            conn.execute(
                f"DELETE FROM copilot_messages WHERE thread_id IN ({placeholders})",
                thread_ids,
            )
            conn.execute(
                f"DELETE FROM copilot_threads WHERE id IN ({placeholders})",
                thread_ids,
            )

        conn.execute(
            "DELETE FROM counselor_patients WHERE counselor_user_id = ? OR patient_user_id = ?",
            (user_id, user_id),
        )
        conn.execute("DELETE FROM counselor_notes WHERE counselor_user_id = ? OR patient_user_id = ?", (user_id, user_id))
        conn.execute("DELETE FROM patient_profiles WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM patient_ai_consents WHERE user_id = ?", (user_id,))

        msg_rows = conn.execute(
            """
            SELECT messages.id
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            WHERE sessions.user_id = ?
            """,
            (user_id,),
        ).fetchall()
        message_ids = [int(row["id"]) for row in msg_rows]
        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            conn.execute(
                f"DELETE FROM assistant_message_meta WHERE assistant_message_id IN ({placeholders})",
                message_ids,
            )
            conn.execute(
                f"DELETE FROM emotion_scores WHERE message_id IN ({placeholders})",
                message_ids,
            )
            conn.execute(
                f"DELETE FROM risk_flags WHERE message_id IN ({placeholders})",
                message_ids,
            )
            conn.execute(
                f"DELETE FROM embeddings WHERE source_type = 'message' AND source_id IN ({placeholders})",
                message_ids,
            )
            conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                message_ids,
            )

        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM risk_flags WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

    return True


def get_or_create_session(user_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE user_id = ? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row:
            return int(row["id"])
        now = now_kst_iso()
        cur = conn.execute(
            "INSERT INTO sessions (user_id, started_at) VALUES (?, ?)",
            (user_id, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_message(session_id: int, role: str, content: str) -> int:
    now = now_kst_iso()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_assistant_message_meta(
    assistant_message_id: int,
    response_json: dict,
    evidence: list[str],
    schema_valid: bool,
    generation_source: str,
    sanitizer_hit: bool,
) -> None:
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO assistant_message_meta (
                assistant_message_id,
                response_json,
                evidence_json,
                schema_valid,
                generation_source,
                sanitizer_hit,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(assistant_message_id) DO UPDATE SET
                response_json = excluded.response_json,
                evidence_json = excluded.evidence_json,
                schema_valid = excluded.schema_valid,
                generation_source = excluded.generation_source,
                sanitizer_hit = excluded.sanitizer_hit,
                created_at = excluded.created_at
            """,
            (
                assistant_message_id,
                json.dumps(response_json, ensure_ascii=False),
                json.dumps(evidence, ensure_ascii=False),
                1 if schema_valid else 0,
                generation_source,
                1 if sanitizer_hit else 0,
                now,
            ),
        )
        conn.commit()


def list_user_messages(user_id: int, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT messages.id, messages.role, messages.content, messages.created_at
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            WHERE sessions.user_id = ?
            ORDER BY messages.id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        items = [dict(r) for r in rows]
        items.reverse()
        return items


def get_emotion_history(user_id: int, limit: int = 5) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT emotion_scores.emotions_json
            FROM emotion_scores
            JOIN messages ON messages.id = emotion_scores.message_id
            JOIN sessions ON sessions.id = messages.session_id
            WHERE sessions.user_id = ?
            ORDER BY emotion_scores.id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        history = []
        for row in rows:
            try:
                history.append(json.loads(row["emotions_json"]))
            except json.JSONDecodeError:
                continue
        return list(reversed(history))


def save_emotion_scores(
    message_id: int,
    emotions: dict,
    rolling: dict,
    trend: dict,
    energy: Optional[dict] = None,
    source: Optional[str] = None,
    error: Optional[str] = None,
    energy_status: Optional[str] = None,
) -> None:
    energy = energy or {}
    angle_reliable = 1 if _safe_bool(energy.get("angle_reliable")) else 0
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO emotion_scores (
                message_id,
                emotions_json,
                intensity_json,
                rolling_scores_json,
                trend_json,
                energy_vx,
                energy_vy,
                energy_tau,
                energy_angle,
                energy_magnitude,
                energy_coherence,
                energy_angle_reliable,
                energy_active_count,
                emotion_source,
                emotion_error,
                energy_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                json.dumps(emotions, ensure_ascii=False),
                json.dumps(emotions, ensure_ascii=False),
                json.dumps(rolling, ensure_ascii=False),
                json.dumps(trend, ensure_ascii=False),
                energy.get("v_x"),
                energy.get("v_y"),
                energy.get("tau"),
                energy.get("angle"),
                energy.get("magnitude"),
                energy.get("coherence"),
                angle_reliable,
                energy.get("active_count"),
                source,
                error,
                energy_status,
            ),
        )
        conn.commit()


def save_risk_flag(message_id: int, user_id: int, score: float, keywords: list[str]) -> None:
    threshold_hit = 1 if score >= settings.risk_threshold else 0
    status = "open" if threshold_hit else "none"
    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO risk_flags (message_id, user_id, score, keywords_json, threshold_hit, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (message_id, user_id, score, json.dumps(keywords, ensure_ascii=False), threshold_hit, status, now),
        )
        conn.commit()


def get_latest_emotion_summary(user_id: int) -> list[dict]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT rolling_scores_json, trend_json
            FROM emotion_scores
            JOIN messages ON messages.id = emotion_scores.message_id
            JOIN sessions ON sessions.id = messages.session_id
            WHERE sessions.user_id = ?
            ORDER BY emotion_scores.id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return []
        rolling = json.loads(row["rolling_scores_json"])
        trend = json.loads(row["trend_json"])
        summary = []
        for key, value in rolling.items():
            summary.append({"name": key, "value": value, "trend": trend.get(key, 0.0)})
        return summary


def build_prompt(
    user_text: str,
    rag: dict,
    user_prompt_template: str,
    assistant_guidance: str = "",
    session_meta: dict[str, Any] | None = None,
) -> str:
    doc_context = "\n".join([f"- ({d['title']}) {d['chunk_text']}" for d in rag["doc_chunks"]])
    msg_context = "\n".join([f"- {m['content']}" for m in rag["messages"]])
    guidance = (assistant_guidance or "").strip() or "없음"
    meta = _format_session_meta_for_prompt(session_meta)
    base = user_prompt_template.format(
        doc_context=doc_context,
        msg_context=msg_context,
        user_text=user_text,
        assistant_guidance=guidance,
        session_meta=meta,
    ).strip()
    if "{assistant_guidance}" not in user_prompt_template:
        base = f"{base}\n\n[도우미 유의 사항]\n{guidance}".strip()
    if "{session_meta}" not in user_prompt_template:
        base = f"{base}\n\n[SESSION_META]\n{meta}".strip()
    return base


def _format_session_meta_for_prompt(session_meta: dict[str, Any] | None = None) -> str:
    payload = session_meta if isinstance(session_meta, dict) else {}
    risk_score = _safe_float(payload.get("risk_score"))
    risk_threshold = _safe_float(payload.get("risk_threshold"))
    resistance_level = int(_safe_float(payload.get("resistance_level")) or 0)
    response_style = str(payload.get("response_style") or "guided_steps")
    is_urgent = _safe_bool(payload.get("is_urgent"))
    format_rejection_detected = _safe_bool(payload.get("format_rejection_detected"))

    lines = [
        f"- risk_score: {risk_score:.4f}" if risk_score is not None else "- risk_score: -",
        f"- risk_threshold: {risk_threshold:.4f}" if risk_threshold is not None else "- risk_threshold: -",
        f"- is_urgent: {'true' if is_urgent else 'false'}",
        f"- resistance_level: {resistance_level}",
        f"- response_style: {response_style}",
        f"- format_rejection_detected: {'true' if format_rejection_detected else 'false'}",
    ]
    return "\n".join(lines)
