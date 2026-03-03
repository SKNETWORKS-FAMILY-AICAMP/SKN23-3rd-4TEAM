from __future__ import annotations

import json
from typing import List

from openai import OpenAI

from app.settings import settings


def _resolve_api_key(api_key: str | None = None) -> str:
    final_key = (api_key or "").strip() or settings.openai_api_key
    if not final_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return final_key


def _resolve_model(model: str | None = None) -> str:
    final_model = (model or "").strip() or settings.openai_model
    if not final_model:
        raise RuntimeError("OPENAI_MODEL이 설정되어 있지 않습니다.")
    return final_model


def chat_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 400,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    client = OpenAI(api_key=_resolve_api_key(api_key))
    response = client.responses.create(
        model=_resolve_model(model),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_tokens,
    )
    text = response.output_text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def embed_texts(
    texts: List[str],
    api_key: str | None = None,
    model: str | None = None,
) -> List[List[float]]:
    client = OpenAI(api_key=_resolve_api_key(api_key))
    response = client.embeddings.create(
        model=(model or "").strip() or settings.openai_embed_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def chat_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    client = OpenAI(api_key=_resolve_api_key(api_key))
    response = client.responses.create(
        model=_resolve_model(model),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_tokens,
    )
    return response.output_text


def list_available_models(api_key: str | None = None) -> List[str]:
    client = OpenAI(api_key=_resolve_api_key(api_key))
    response = client.models.list()
    model_ids = []
    for item in getattr(response, "data", []):
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())
    return sorted(set(model_ids))
