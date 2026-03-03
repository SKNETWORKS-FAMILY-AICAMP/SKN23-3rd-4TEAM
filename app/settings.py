import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _parse_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _secret_has_min_complexity(secret: str) -> bool:
    classes = 0
    if any(ch.islower() for ch in secret):
        classes += 1
    if any(ch.isupper() for ch in secret):
        classes += 1
    if any(ch.isdigit() for ch in secret):
        classes += 1
    if any(not ch.isalnum() for ch in secret):
        classes += 1
    return classes >= 3


def _validate_secret_key(secret_key: str, min_len: int, required: bool) -> None:
    if not required:
        return

    normalized = (secret_key or "").strip()
    weak_defaults = {
        "",
        "change-me",
        "changeme",
        "default",
        "secret",
        "dev",
        "test",
    }
    if normalized.lower() in weak_defaults:
        raise ValueError("SECRET_KEY가 약한 기본값입니다. 강한 값으로 설정하세요.")
    if len(normalized) < max(16, min_len):
        raise ValueError(f"SECRET_KEY 길이가 너무 짧습니다. 최소 {max(16, min_len)}자 이상이어야 합니다.")
    if not _secret_has_min_complexity(normalized):
        raise ValueError("SECRET_KEY 복잡도가 낮습니다. 대/소문자, 숫자, 특수문자 조합을 사용하세요.")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_embed_model: str
    secret_key: str
    db_path: str
    pdf_dir: str
    rag_top_k: int
    risk_threshold: float
    emotion_window: int
    energy_enabled: bool
    energy_beta: float
    energy_exp_clip: float
    log_path: str
    require_strong_secret: bool
    min_secret_key_len: int
    csrf_ttl_seconds: int
    csrf_enforce: bool
    dynamic_response_style_enabled: bool


    @staticmethod
    def from_env() -> "Settings":
        require_strong_secret = _parse_bool(os.getenv("REQUIRE_STRONG_SECRET"), default=False)
        min_secret_key_len = int(os.getenv("MIN_SECRET_KEY_LEN", "16"))
        secret_key = os.getenv("SECRET_KEY", "change-me")
        _validate_secret_key(secret_key, min_secret_key_len, require_strong_secret)

        return Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            secret_key=secret_key,
            db_path=os.getenv("DB_PATH", "/home/user/Documents/01_Projects/therapy_assist_bot/data/app.db"),
            pdf_dir=os.getenv("PDF_DIR", "/home/user/Documents"),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            risk_threshold=float(os.getenv("RISK_THRESHOLD", "0.7")),
            emotion_window=int(os.getenv("EMOTION_WINDOW", "5")),
            energy_enabled=_parse_bool(os.getenv("ENERGY_ENABLED"), default=True),
            energy_beta=float(os.getenv("ENERGY_BETA", "0.5")),
            energy_exp_clip=float(os.getenv("ENERGY_EXP_CLIP", "20.0")),
            log_path=os.getenv("LOG_PATH", "/home/user/Documents/01_Projects/therapy_assist_bot/data/app.log"),
            require_strong_secret=require_strong_secret,
            min_secret_key_len=max(16, min_secret_key_len),
            csrf_ttl_seconds=max(60, int(os.getenv("CSRF_TTL_SECONDS", "7200"))),
            csrf_enforce=_parse_bool(os.getenv("CSRF_ENFORCE"), default=False),
            dynamic_response_style_enabled=_parse_bool(os.getenv("DYNAMIC_RESPONSE_STYLE_ENABLED"), default=True),
        )


settings = Settings.from_env()
