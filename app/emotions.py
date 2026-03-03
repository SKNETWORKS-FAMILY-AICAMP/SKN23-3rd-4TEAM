from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import numpy as np

from app.risk import normalize_text


EMOTIONS = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]


KEYWORD_BOOSTS: Dict[str, List[str]] = {
    "joy": ["행복", "기쁨", "즐거", "신나"],
    "trust": ["믿", "안심", "신뢰"],
    "fear": ["무서", "무섭", "두려", "겁", "불안", "막막"],
    "surprise": ["놀라", "깜짝"],
    "sadness": ["슬프", "우울", "눈물", "힘들", "엉망", "지치", "버겁", "괴로"],
    "disgust": ["역겨", "혐오"],
    "anger": ["화나", "화가", "분노", "짜증", "억까", "억울", "빡치", "분해"],
    "anticipation": ["기대", "설레", "걱정"],
}


EMOTION_ANGLES_DEG: Dict[str, float] = {
    "joy": 0.0,
    "anticipation": 45.0,
    "anger": 90.0,
    "disgust": 135.0,
    "sadness": 180.0,
    "surprise": 225.0,
    "fear": 270.0,
    "trust": 315.0,
}

ANGLE_EPS = 1e-6
FLAT_EPS = 1e-9


@dataclass
class EmotionResult:
    emotions: Dict[str, float]
    rolling: Dict[str, float]
    trend: Dict[str, float]
    energy: Dict[str, float] | None = None
    source: str = "llm_json"
    error: str | None = None
    energy_status: str = "zero_input"


class DynamicEnergyCyclicEncoder:
    """벡터화된 동적 에너지 사이클릭 인코더."""

    def __init__(
        self,
        sensitivity_profile: Dict[str, float] | None = None,
        exp_clip: float = 20.0,
    ):
        self.emotion_order = tuple(EMOTIONS)
        self.exp_clip = float(max(1.0, exp_clip))

        base_profile = {emotion: 1.0 for emotion in self.emotion_order}
        if sensitivity_profile:
            for emotion, value in sensitivity_profile.items():
                if emotion in base_profile:
                    base_profile[emotion] = float(value)
        self.sensitivity_profile = base_profile
        self.sensitivity_vector = np.array(
            [self.sensitivity_profile[emotion] for emotion in self.emotion_order],
            dtype=np.float64,
        )

        angles = np.array(
            [np.deg2rad(EMOTION_ANGLES_DEG[emotion]) for emotion in self.emotion_order],
            dtype=np.float64,
        )
        self.cos_angles = np.cos(angles)
        self.sin_angles = np.sin(angles)

    def _to_intensity_vector(self, intensities: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.emotion_order), dtype=np.float64)
        for idx, emotion in enumerate(self.emotion_order):
            raw = intensities.get(emotion, 0.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            vec[idx] = np.clip(value, 0.0, 1.0)
        return vec

    def _calculate_energy(self, intensity_vec: np.ndarray) -> np.ndarray:
        exp_input = np.clip(self.sensitivity_vector * intensity_vec, -self.exp_clip, self.exp_clip)
        return intensity_vec * np.exp(exp_input)

    def encode_with_dynamic_threshold(
        self,
        intensities: Dict[str, float],
        beta: float = 0.5,
    ) -> Tuple[float, float, float, int, str, np.ndarray]:
        if not intensities:
            return 0.0, 0.0, 0.0, 0, "zero_input", np.zeros(len(self.emotion_order), dtype=np.float64)

        intensity_vec = self._to_intensity_vector(intensities)
        energies = self._calculate_energy(intensity_vec)
        beta_value = max(0.0, float(beta))

        tau_dynamic = float(np.mean(energies) + beta_value * np.std(energies))
        active = np.maximum(energies - tau_dynamic, 0.0)
        active_count = int(np.count_nonzero(active > 0.0))
        status = "normal"

        if active_count == 0:
            max_energy = float(np.max(energies)) if energies.size > 0 else 0.0
            min_energy = float(np.min(energies)) if energies.size > 0 else 0.0
            if max_energy <= ANGLE_EPS:
                status = "zero_input"
            elif (max_energy - min_energy) <= FLAT_EPS:
                status = "flat_distribution"
            else:
                peak_idx = int(np.argmax(energies))
                active[peak_idx] = max_energy
                active_count = 1
                status = "fallback_single_peak"

        v_x = float(np.sum(active * self.cos_angles))
        v_y = float(np.sum(active * self.sin_angles))
        return v_x, v_y, tau_dynamic, active_count, status, active

    def analyze_state(self, v_x: float, v_y: float) -> Tuple[float, float, bool]:
        magnitude = float(np.hypot(v_x, v_y))
        if magnitude < ANGLE_EPS:
            return 0.0, 0.0, False

        angle_rad = float(np.arctan2(v_y, v_x))
        if angle_rad < 0.0:
            angle_rad += 2.0 * np.pi
        angle_deg = float(np.rad2deg(angle_rad))
        return angle_deg, magnitude, True

    def calculate_metrics(self, intensities: Dict[str, float], beta: float = 0.5) -> Dict[str, float]:
        v_x, v_y, tau_dynamic, active_count, status, active = self.encode_with_dynamic_threshold(intensities, beta=beta)
        angle_deg, magnitude, angle_reliable = self.analyze_state(v_x, v_y)
        active_sum = float(np.sum(active)) if active.size > 0 else 0.0
        if active_sum <= ANGLE_EPS:
            coherence = 0.0
        else:
            coherence = float(max(0.0, min(1.0, magnitude / active_sum)))
        return {
            "v_x": round(v_x, 6),
            "v_y": round(v_y, 6),
            "tau": round(tau_dynamic, 6),
            "angle": round(angle_deg, 6),
            "magnitude": round(magnitude, 6),
            "coherence": round(coherence, 6),
            "angle_reliable": bool(angle_reliable),
            "active_count": active_count,
            "status": status,
        }


def _fill_emotions(raw: Dict[str, float]) -> Dict[str, float]:
    result = {key: 0.0 for key in EMOTIONS}
    for key, value in raw.items():
        if key in result:
            result[key] = float(max(0.0, min(1.0, value)))
    return result


def apply_rule_boost(emotions: Dict[str, float], text: str, boost: float = 0.1) -> Dict[str, float]:
    normalized = normalize_text(text)
    updated = dict(emotions)
    for emotion, keywords in KEYWORD_BOOSTS.items():
        if sum(_count_non_negated_keyword_hits(normalized, keyword) for keyword in keywords) > 0:
            updated[emotion] = min(1.0, updated.get(emotion, 0.0) + boost)
    return updated


def score_emotions_by_rules(text: str, step: float = 0.25, cap: float = 0.8) -> Dict[str, float]:
    normalized = normalize_text(text)
    scores = {emotion: 0.0 for emotion in EMOTIONS}
    step_value = max(0.0, float(step))
    cap_value = min(1.0, max(0.0, float(cap)))

    if not normalized:
        return scores

    for emotion, keywords in KEYWORD_BOOSTS.items():
        hits = sum(_count_non_negated_keyword_hits(normalized, keyword) for keyword in keywords)
        if hits <= 0:
            continue
        scores[emotion] = min(cap_value, round(hits * step_value, 4))

    return scores


def _count_non_negated_keyword_hits(normalized: str, keyword: str) -> int:
    count = 0
    for match in re.finditer(re.escape(keyword), normalized):
        if _is_negated_context(normalized, match.start(), match.end()):
            continue
        count += 1
    return count


def _is_negated_context(normalized: str, start: int, end: int) -> bool:
    left = normalized[max(0, start - 8) : start]
    right = normalized[end : min(len(normalized), end + 10)]
    if re.search(r"(안\s*|못\s*|전혀\s*)$", left):
        return True
    if re.search(r"(않|아니|없)", right):
        return True
    return False


def compute_rolling_scores(history: List[Dict[str, float]], window: int = 5) -> Dict[str, Dict[str, float]]:
    if not history:
        empty = {k: 0.0 for k in EMOTIONS}
        return {"rolling": empty, "trend": empty}

    recent = history[-window:]
    rolling = {k: 0.0 for k in EMOTIONS}
    for item in recent:
        filled = _fill_emotions(item)
        for key in EMOTIONS:
            rolling[key] += filled[key]

    count = len(recent)
    for key in EMOTIONS:
        rolling[key] = round(rolling[key] / count, 2)

    if len(recent) > 1:
        prev = compute_rolling_scores(recent[:-1], window=len(recent) - 1)["rolling"]
    else:
        prev = {k: 0.0 for k in EMOTIONS}

    trend = {k: round(rolling[k] - prev.get(k, 0.0), 2) for k in EMOTIONS}
    return {"rolling": rolling, "trend": trend}
