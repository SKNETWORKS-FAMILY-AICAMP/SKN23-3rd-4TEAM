from __future__ import annotations

import re
from typing import Dict, List


RISK_KEYWORDS: Dict[str, float] = {
    "자살": 0.9,
    "죽고 싶어": 0.9,
    "죽고싶어": 0.9,
    "자해": 0.8,
    "해치고 싶어": 0.7,
    "끝내고 싶어": 0.6,
    "없어지고 싶어": 0.6,
    "살기 싫어": 0.7,
}

SELF_HARM_HINTS = [
    "자살",
    "자해",
    "죽",
    "해치",
    "끝내",
    "없어지",
    "사라지",
    "살기 싫",
    "목숨",
]

NEGATED_RISK_PATTERNS: Dict[str, List[str]] = {
    "자살": [
        r"자살\s*(?:생각|충동|계획)?\s*(?:은|는|이|가)?\s*(?:없|없고|없다|없어요|없습니다|아니|아닌|아님)",
    ],
    "자해": [
        r"자해\s*(?:생각|충동|계획)?\s*(?:은|는|이|가)?\s*(?:없|없고|없다|없어요|없습니다|아니|아닌|아님)",
    ],
    "해치고 싶어": [
        r"해치고\s*싶(?:지|진|지는)\s*않",
    ],
    "끝내고 싶어": [
        r"끝내고\s*싶(?:지|진|지는)\s*않",
    ],
    "없어지고 싶어": [
        r"없어지고\s*싶(?:지|진|지는)\s*않",
    ],
    "살기 싫어": [
        r"살기\s*싫(?:지|진|지는)\s*않",
    ],
    "죽고 싶어": [
        r"죽고\s*싶(?:지|진|지는)\s*않",
    ],
    "죽고싶어": [
        r"죽고\s*싶(?:지|진|지는)\s*않",
    ],
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_negated_keyword(normalized: str, keyword: str) -> bool:
    patterns = NEGATED_RISK_PATTERNS.get(keyword, [])
    return any(re.search(pattern, normalized) for pattern in patterns)


def has_self_harm_context(text: str) -> bool:
    normalized = normalize_text(text)
    return any(hint in normalized for hint in SELF_HARM_HINTS)


def score_risk(text: str) -> dict:
    normalized = normalize_text(text)
    matched: List[str] = []
    score = 0.0
    for keyword, weight in RISK_KEYWORDS.items():
        if keyword in normalized:
            if _is_negated_keyword(normalized, keyword):
                continue
            matched.append(keyword)
            score = max(score, weight)
    return {"score": round(score, 2), "keywords": matched}


def risk_hits_threshold(score: float, threshold: float) -> bool:
    return score >= threshold
