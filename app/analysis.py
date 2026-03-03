from __future__ import annotations

import json
import logging
import time
from typing import Dict, List

from app.emotions import (
    EMOTIONS,
    DynamicEnergyCyclicEncoder,
    EmotionResult,
    apply_rule_boost,
    compute_rolling_scores,
    score_emotions_by_rules,
)
from app.openai_client import chat_json
from app.risk import has_self_harm_context, score_risk
from app.settings import settings


EMOTION_SYSTEM_PROMPT = """당신은 심리상담 대화의 감정 분류기입니다.
반드시 JSON 객체 한 개만 출력하세요. 마크다운/설명/코드블록은 금지합니다.
키는 정확히 다음 8개만 사용하세요:
joy, trust, fear, surprise, sadness, disgust, anger, anticipation
규칙:
1) 모든 키를 반드시 포함합니다.
2) 값은 0~1 사이 float입니다.
3) 근거가 부족한 감정은 0으로 둡니다.
"""

RISK_SYSTEM_PROMPT = """당신은 심리상담 대화의 위험도 분류기입니다.
자살/자해 위험도를 0~1 사이로 추정하세요.
반드시 JSON 객체 한 개만 출력하세요. 마크다운/설명/코드블록은 금지합니다.
키는 risk_score 하나만 사용하세요.
규칙:
1) risk_score는 0~1 사이 float입니다.
2) 정보가 불충분해도 null 대신 보수적 점수(예: 0.0~0.3)를 반환합니다.
"""

logger = logging.getLogger(__name__)


def _validate_emotion_payload(payload: object) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "llm_invalid_json"
    if "raw" in payload:
        return None, "llm_invalid_json"

    keys = set(payload.keys())
    if keys != set(EMOTIONS):
        return None, "llm_invalid_json"

    parsed: dict[str, float] = {}
    try:
        for key in EMOTIONS:
            value = float(payload[key])
            parsed[key] = max(0.0, min(1.0, value))
    except (TypeError, ValueError, KeyError):
        return None, "llm_invalid_json"

    return parsed, None


def analyze_emotions(
    text: str,
    history: List[Dict[str, float]],
    api_key: str | None = None,
    model: str | None = None,
    user_id: int | None = None,
    message_id: int | None = None,
) -> EmotionResult:
    started = time.perf_counter()
    source = "llm_json"
    error: str | None = None
    llm_result: dict | None = {}
    validated: dict | None = None

    try:
        llm_result = chat_json(
            EMOTION_SYSTEM_PROMPT,
            text,
            max_tokens=200,
            api_key=api_key,
            model=model,
        )
        validated, error = _validate_emotion_payload(llm_result)
        if validated is None:
            source = "llm_invalid_json"
    except Exception:
        llm_result = {}
        source = "llm_exception"
        error = "llm_exception"

    if validated is not None:
        boosted = apply_rule_boost(validated, text)
    else:
        boosted = score_emotions_by_rules(text)
        source = "rules_fallback"

    rolling = compute_rolling_scores(history + [boosted], window=settings.emotion_window)

    energy_metrics = None
    energy_status = "zero_input"
    if settings.energy_enabled:
        try:
            encoder = DynamicEnergyCyclicEncoder(exp_clip=settings.energy_exp_clip)
            energy_metrics = encoder.calculate_metrics(boosted, beta=settings.energy_beta)
            energy_status = str(energy_metrics.get("status", "normal"))
        except Exception:
            energy_metrics = None
            energy_status = "zero_input"

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    logger.info(
        "emotion_analysis %s",
        json.dumps(
            {
                "user_id": user_id,
                "message_id": message_id,
                "source": source,
                "error_type": error,
                "energy_status": energy_status,
                "latency_ms": latency_ms,
            },
            ensure_ascii=False,
        ),
    )

    return EmotionResult(
        emotions=boosted,
        rolling=rolling["rolling"],
        trend=rolling["trend"],
        energy=energy_metrics,
        source=source,
        error=error,
        energy_status=energy_status,
    )


def analyze_risk(text: str, api_key: str | None = None, model: str | None = None) -> dict:
    rule = score_risk(text)
    has_hint = has_self_harm_context(text)
    llm_score = None
    try:
        llm = chat_json(
            RISK_SYSTEM_PROMPT,
            text,
            max_tokens=60,
            api_key=api_key,
            model=model,
        )
        if isinstance(llm, dict) and "risk_score" in llm:
            llm_score = max(0.0, min(1.0, float(llm["risk_score"])))
    except Exception:
        llm_score = None

    final = rule["score"]
    if llm_score is not None:
        if final <= 0.0 and not has_hint:
            # 자/타해 맥락이 전혀 없는 문장에서 LLM 단독 과대평가를 제한한다.
            final = min(llm_score, 0.35)
        else:
            final = max(final, llm_score)

    return {"score": round(final, 2), "keywords": rule["keywords"]}
