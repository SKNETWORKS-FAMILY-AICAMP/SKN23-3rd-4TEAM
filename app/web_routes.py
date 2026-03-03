from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from app.domain_services import *  # noqa: F401,F403
from app.domain_services import (
    _build_emotion_profile_ko,
    _build_note_created_at_iso,
    _extract_emotion_profile,
    _extract_top_emotion,
    _format_datetime_kst,
    _format_decimal,
    _safe_bool,
    _safe_float,
)
from app.runtime_config import DEFAULT_PATIENT_SYSTEM_PROMPT, DEFAULT_PATIENT_USER_PROMPT_TEMPLATE


router = APIRouter()


@router.get("/")
def index(request: Request):
    if not has_any_user():
        return RedirectResponse("/setup", status_code=302)

    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    if user["role"] == "admin":
        return RedirectResponse("/admin", status_code=302)
    if user["role"] == "counselor":
        return RedirectResponse("/counselor", status_code=302)
    if not has_patient_ai_consent(user["id"]):
        return RedirectResponse("/consent/ai", status_code=302)
    return RedirectResponse("/chat", status_code=302)


@router.get("/setup")
def setup_get(request: Request):
    if has_any_user():
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("setup.html", {"request": request, "current_user": None})


@router.post("/setup")
def setup_post(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    if has_any_user():
        return RedirectResponse("/login", status_code=302)

    ok, msg, cleaned_username = validate_username(username)
    if not ok:
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "current_user": None, "error": msg},
        )

    ok, msg = validate_pin(pin)
    if not ok:
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "current_user": None, "error": msg},
        )
    create_user(username=cleaned_username, role="admin", pin=pin)
    return RedirectResponse("/login", status_code=302)


@router.get("/login")
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "current_user": None})


@router.post("/auth/pin")
def login_post(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    login_id = username.strip()
    user = get_user_by_username(login_id)
    if not user:
        normalized = normalize_phone(login_id)
        if len(normalized) in (10, 11):
            user = get_user_by_phone(normalized) or get_user_by_username(normalized)
    if not user or not verify_pin(pin, user["pin_hash"]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "current_user": None, "error": "로그인 정보가 올바르지 않습니다."},
        )
    request.session["user_id"] = user["id"]
    return RedirectResponse("/", status_code=302)


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@router.get("/consent/ai")
def consent_ai_get(request: Request, error: str = ""):
    user = get_current_user(request)
    if not require_role(user, "patient"):
        return RedirectResponse("/", status_code=302)

    ensure_patient_consent_row(user["id"])
    consent = get_patient_ai_consent(user["id"])
    if int(consent.get("consent_given", 0)) == 1:
        return RedirectResponse("/chat", status_code=302)

    return templates.TemplateResponse(
        "consent_ai.html",
        {
            "request": request,
            "current_user": user,
            "error": error,
            "consent_text": consent.get("consent_text") or AI_CONSENT_TEXT,
            "consent_version": consent.get("consent_version") or AI_CONSENT_VERSION,
        },
    )


@router.post("/consent/ai")
def consent_ai_post(
    request: Request,
    agree: Optional[str] = Form(None),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "patient"):
        return RedirectResponse("/", status_code=302)

    if not agree:
        return templates.TemplateResponse(
            "consent_ai.html",
            {
                "request": request,
                "current_user": user,
                "error": "AI 상담 보조 사용 서약에 동의해야 채팅을 시작할 수 있습니다.",
                "consent_text": AI_CONSENT_TEXT,
                "consent_version": AI_CONSENT_VERSION,
            },
        )

    set_patient_ai_consent(user["id"], consent_given=True, updated_by=user["id"])
    return RedirectResponse("/chat", status_code=302)


@router.get("/chat")
def chat_get(request: Request):
    user = get_current_user(request)
    if not require_role(user, "patient"):
        return RedirectResponse("/", status_code=302)
    if not has_patient_ai_consent(user["id"]):
        return RedirectResponse("/consent/ai", status_code=302)

    messages = list_user_messages(user["id"])
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "current_user": user, "messages": messages},
    )


@router.post("/chat/send")
def chat_send(request: Request, content: str = Form(...), csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "patient"):
        return RedirectResponse("/", status_code=302)
    if not has_patient_ai_consent(user["id"]):
        return RedirectResponse("/consent/ai", status_code=302)

    recent_messages = list_user_messages(user["id"], limit=6)
    recent_user_texts = [item.get("content", "") for item in recent_messages if item.get("role") == "user"][-3:]

    session_id = get_or_create_session(user["id"])
    user_msg_id = add_message(session_id, "user", content)

    try:
        index_message_embedding(user_msg_id, content)
    except Exception as exc:
        logger.warning(
            "message_embedding_index_failed %s",
            json.dumps(
                {"user_id": user["id"], "message_id": user_msg_id, "error_type": exc.__class__.__name__},
                ensure_ascii=False,
            ),
        )

    runtime_config = get_runtime_config()
    history = get_emotion_history(user["id"], limit=settings.emotion_window)
    emotion_started = time.perf_counter()
    emotion_result = analyze_emotions(
        content,
        history,
        api_key=runtime_config["openai_api_key"],
        model=runtime_config["openai_model"],
        user_id=user["id"],
        message_id=user_msg_id,
    )
    emotion_latency_ms = round((time.perf_counter() - emotion_started) * 1000.0, 3)
    save_emotion_scores(
        user_msg_id,
        emotion_result.emotions,
        emotion_result.rolling,
        emotion_result.trend,
        emotion_result.energy,
        source=emotion_result.source,
        error=emotion_result.error,
        energy_status=emotion_result.energy_status,
    )
    logger.info(
        "emotion_score_saved %s",
        json.dumps(
            {
                "user_id": user["id"],
                "message_id": user_msg_id,
                "source": emotion_result.source,
                "error_type": emotion_result.error,
                "energy_status": emotion_result.energy_status,
                "latency_ms": emotion_latency_ms,
            },
            ensure_ascii=False,
        ),
    )

    risk = analyze_risk(
        content,
        api_key=runtime_config["openai_api_key"],
        model=runtime_config["openai_model"],
    )
    save_risk_flag(user_msg_id, user["id"], risk["score"], risk["keywords"])
    is_urgent = risk["score"] >= settings.risk_threshold
    resistance_level = detect_response_resistance(content, recent_user_texts)
    if settings.dynamic_response_style_enabled:
        response_style = select_response_style(risk["score"], settings.risk_threshold, resistance_level)
    else:
        response_style = "urgent_plain" if is_urgent else "guided_steps"
    format_rejection_detected = bool(resistance_level >= 2)
    safety_override_applied = bool(is_urgent and format_rejection_detected)
    session_meta = {
        "risk_score": risk["score"],
        "risk_threshold": settings.risk_threshold,
        "is_urgent": is_urgent,
        "resistance_level": resistance_level,
        "response_style": response_style,
        "format_rejection_detected": format_rejection_detected,
    }

    assistant_text = build_patient_fallback_message(is_urgent=is_urgent)
    assistant_response_json: dict = {}
    assistant_evidence: list[str] = []
    assistant_schema_valid = False
    assistant_generation_source = "llm_exception_fallback"
    sanitizer_hit = False
    try:
        rag = search_rag(content, top_k=settings.rag_top_k, user_id=user["id"])
        profile = get_patient_profile(user["id"])
        prompt = build_prompt(
            content,
            rag,
            runtime_config["patient_user_prompt_template"],
            assistant_guidance=profile.get("assistant_guidance", ""),
            session_meta=session_meta,
        )
        raw_payload = chat_json(
            runtime_config["patient_system_prompt"],
            prompt,
            max_tokens=600,
            api_key=runtime_config["openai_api_key"],
            model=runtime_config["openai_model"],
        )
        assistant_response_json = raw_payload if isinstance(raw_payload, dict) else {"raw": str(raw_payload)}
        validated_payload, validation_errors = validate_patient_payload(raw_payload)
        if validated_payload is None:
            assistant_generation_source = "json_invalid_fallback"
            assistant_response_json["validation_errors"] = validation_errors
            assistant_text = build_patient_fallback_message(is_urgent=is_urgent)
            logger.warning(
                "assistant_response_invalid_payload %s",
                json.dumps(
                    {
                        "user_id": user["id"],
                        "message_id": user_msg_id,
                        "errors": validation_errors,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            assistant_generation_source = "json_valid"
            assistant_schema_valid = True
            assistant_evidence = list(validated_payload.get("internal_evidence", []))
            assistant_text = render_patient_message(
                validated_payload,
                style=response_style,
                is_urgent=is_urgent,
            )
    except Exception as exc:
        logger.warning(
            "assistant_response_failed %s",
            json.dumps(
                {"user_id": user["id"], "message_id": user_msg_id, "error_type": exc.__class__.__name__},
                ensure_ascii=False,
            ),
        )
        assistant_response_json = {"error_type": exc.__class__.__name__}
        assistant_generation_source = "llm_exception_fallback"
        assistant_text = build_patient_fallback_message(is_urgent=is_urgent)

    assistant_response_json.setdefault("_meta", {})
    if not isinstance(assistant_response_json.get("_meta"), dict):
        assistant_response_json["_meta"] = {}
    assistant_response_json["_meta"].update(
        {
            "response_style": response_style,
            "resistance_level": resistance_level,
            "safety_override_applied": safety_override_applied,
            "prompt_policy_version": "patient_v2_resistance_aware",
            "session_meta_applied": True,
        }
    )
    logger.info(
        "assistant_response_style_applied %s",
        json.dumps(
            {
                "user_id": user["id"],
                "message_id": user_msg_id,
                "risk_score": risk["score"],
                "response_style": response_style,
                "resistance_level": resistance_level,
                "safety_override_applied": safety_override_applied,
            },
            ensure_ascii=False,
        ),
    )
    logger.info(
        "assistant_prompt_policy_applied %s",
        json.dumps(
            {
                "user_id": user["id"],
                "message_id": user_msg_id,
                "policy_version": "patient_v2_resistance_aware",
                "response_style": response_style,
                "risk_score": risk["score"],
                "resistance_level": resistance_level,
                "is_urgent": is_urgent,
            },
            ensure_ascii=False,
        ),
    )

    assistant_text, sanitizer_hit = sanitize_patient_display_text(assistant_text)
    if sanitizer_hit:
        logger.info(
            "assistant_response_sanitizer_hit %s",
            json.dumps(
                {
                    "user_id": user["id"],
                    "message_id": user_msg_id,
                    "stage": "pre_save",
                },
                ensure_ascii=False,
            ),
        )

    assistant_id = add_message(session_id, "assistant", assistant_text)
    save_assistant_message_meta(
        assistant_message_id=assistant_id,
        response_json=assistant_response_json,
        evidence=assistant_evidence,
        schema_valid=assistant_schema_valid,
        generation_source=assistant_generation_source,
        sanitizer_hit=sanitizer_hit,
    )
    try:
        index_message_embedding(assistant_id, assistant_text)
    except Exception as exc:
        logger.warning(
            "assistant_embedding_index_failed %s",
            json.dumps(
                {"user_id": user["id"], "message_id": assistant_id, "error_type": exc.__class__.__name__},
                ensure_ascii=False,
            ),
        )

    if request.headers.get("HX-Request"):
        assistant_text, render_hit = sanitize_patient_display_text(assistant_text)
        if render_hit:
            logger.info(
                "assistant_response_sanitizer_hit %s",
                json.dumps(
                    {
                        "user_id": user["id"],
                        "message_id": assistant_id,
                        "stage": "pre_render",
                    },
                    ensure_ascii=False,
                ),
            )
        return templates.TemplateResponse(
            "partials/message_pair.html",
            {"request": request, "user_text": content, "assistant_text": assistant_text},
        )

    return RedirectResponse("/chat", status_code=302)


@router.get("/admin")
def admin_get(request: Request, error: str = "", notice: str = ""):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    users = list_users()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT risk_flags.id, risk_flags.score, risk_flags.keywords_json, risk_flags.status,
                   users.username
            FROM risk_flags
            JOIN users ON users.id = risk_flags.user_id
            WHERE risk_flags.threshold_hit = 1
            ORDER BY risk_flags.id DESC
            LIMIT 20
            """
        ).fetchall()
        alerts = []
        for row in rows:
            alerts.append(
                {
                    "id": row["id"],
                    "score": row["score"],
                    "keywords": ",".join(json.loads(row["keywords_json"])),
                    "status": row["status"],
                    "username": row["username"],
                }
            )

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "current_user": user,
            "alerts": alerts,
            "users": users,
            "error": error,
            "notice": notice,
            "ai_config": get_runtime_config_view(),
        },
    )


@router.get("/admin/diagnostics/emotion")
def admin_emotion_diagnostics(request: Request, limit: int = 200):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    safe_limit = max(1, min(1000, int(limit)))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT emotion_source,
                   emotion_error,
                   energy_magnitude,
                   energy_status,
                   energy_coherence,
                   energy_angle,
                   energy_angle_reliable,
                   emotions_json
            FROM emotion_scores
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    source_distribution: dict[str, int] = {}
    error_distribution: dict[str, int] = {}
    status_distribution: dict[str, int] = {}
    zero_energy_count = 0
    angle_unreliable_count = 0
    coherence_sum = 0.0
    consistency_sum = 0.0
    low_consistency_count = 0

    for row in rows:
        source = (row["emotion_source"] or "").strip() or "unknown"
        source_distribution[source] = source_distribution.get(source, 0) + 1

        error = (row["emotion_error"] or "").strip()
        if error:
            error_distribution[error] = error_distribution.get(error, 0) + 1

        status = (row["energy_status"] or "").strip() or "unknown"
        status_distribution[status] = status_distribution.get(status, 0) + 1

        magnitude = row["energy_magnitude"]
        try:
            magnitude_value = float(magnitude) if magnitude is not None else 0.0
        except (TypeError, ValueError):
            magnitude_value = 0.0
        if magnitude_value <= 0.0:
            zero_energy_count += 1

        angle_reliable = _safe_bool(row["energy_angle_reliable"])
        if not angle_reliable:
            angle_unreliable_count += 1

        coherence_value = max(0.0, min(1.0, _safe_float(row["energy_coherence"]) or 0.0))
        coherence_sum += coherence_value

        profile = _extract_emotion_profile(row["emotions_json"])
        alignment = compute_emotion_alignment(profile, _safe_float(row["energy_angle"]), angle_reliable, coherence_value)
        score_value = _safe_float(alignment.get("consistency_score")) or 0.0
        consistency_sum += score_value
        if score_value < 40.0:
            low_consistency_count += 1

    total = len(rows)
    zero_energy_ratio = round((zero_energy_count / total), 4) if total else 0.0
    angle_unreliable_ratio = round((angle_unreliable_count / total), 4) if total else 0.0
    avg_coherence = round((coherence_sum / total), 4) if total else 0.0
    avg_consistency_score = round((consistency_sum / total), 1) if total else 0.0
    low_consistency_ratio = round((low_consistency_count / total), 4) if total else 0.0

    return JSONResponse(
        {
            "total": total,
            "source_distribution": source_distribution,
            "error_distribution": error_distribution,
            "zero_energy_ratio": zero_energy_ratio,
            "status_distribution": status_distribution,
            "angle_unreliable_ratio": angle_unreliable_ratio,
            "avg_coherence": avg_coherence,
            "avg_consistency_score": avg_consistency_score,
            "low_consistency_ratio": low_consistency_ratio,
        }
    )


@router.get("/admin/diagnostics/response-quality")
def admin_response_quality_diagnostics(request: Request, limit: int = 200):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    safe_limit = max(1, min(1000, int(limit)))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT messages.id, messages.content,
                   assistant_message_meta.schema_valid,
                   assistant_message_meta.sanitizer_hit,
                   assistant_message_meta.generation_source
            FROM messages
            LEFT JOIN assistant_message_meta ON assistant_message_meta.assistant_message_id = messages.id
            WHERE messages.role = 'assistant'
            ORDER BY messages.id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    schema_valid_count = 0
    sanitizer_hit_count = 0
    meta_leak_count = 0
    generation_source_distribution: dict[str, int] = {}
    lowered_forbidden = tuple(pattern.lower() for pattern in FORBIDDEN_PATTERNS)

    for row in rows:
        if int(row["schema_valid"] or 0) == 1:
            schema_valid_count += 1
        if int(row["sanitizer_hit"] or 0) == 1:
            sanitizer_hit_count += 1

        source = (row["generation_source"] or "").strip() or "unknown"
        generation_source_distribution[source] = generation_source_distribution.get(source, 0) + 1

        content = (row["content"] or "").lower()
        if any(marker in content for marker in lowered_forbidden):
            meta_leak_count += 1

    total = len(rows)
    schema_valid_rate = round((schema_valid_count / total), 4) if total else 0.0
    sanitizer_hit_rate = round((sanitizer_hit_count / total), 4) if total else 0.0

    return JSONResponse(
        {
            "total": total,
            "schema_valid_rate": schema_valid_rate,
            "sanitizer_hit_rate": sanitizer_hit_rate,
            "meta_leak_count": meta_leak_count,
            "generation_source_distribution": generation_source_distribution,
        }
    )


@router.get("/admin/consents")
def admin_consents_get(request: Request, error: str = "", notice: str = ""):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse(
        "admin_consents.html",
        {
            "request": request,
            "current_user": user,
            "error": error,
            "notice": notice,
            "consents": list_patient_ai_consents(),
        },
    )


@router.post("/admin/consents/{user_id}/grant")
def admin_consent_grant(request: Request, user_id: int, csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    target = get_user_by_id(user_id)
    if not target or target.get("role") != "patient":
        return RedirectResponse("/admin/consents?error=환자+계정을+찾을+수+없습니다.", status_code=302)

    set_patient_ai_consent(user_id, consent_given=True, updated_by=user["id"])
    return RedirectResponse("/admin/consents?notice=서약+동의+상태로+변경했습니다.", status_code=302)


@router.post("/admin/consents/{user_id}/revoke")
def admin_consent_revoke(request: Request, user_id: int, csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    target = get_user_by_id(user_id)
    if not target or target.get("role") != "patient":
        return RedirectResponse("/admin/consents?error=환자+계정을+찾을+수+없습니다.", status_code=302)

    set_patient_ai_consent(user_id, consent_given=False, updated_by=user["id"])
    return RedirectResponse("/admin/consents?notice=서약+철회+상태로+변경했습니다.", status_code=302)


@router.post("/admin/config/openai")
def admin_update_openai_config(
    request: Request,
    openai_model: str = Form(...),
    openai_api_key: str = Form(""),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    model = openai_model.strip()
    if not model:
        return RedirectResponse("/admin?error=모델명은+비워둘+수+없습니다.", status_code=302)

    updates = {"openai_model": model}
    new_api_key = openai_api_key.strip()
    if new_api_key:
        updates["openai_api_key"] = new_api_key

    set_runtime_config(updates, updated_by=user["id"])
    return RedirectResponse("/admin?notice=OpenAI+설정을+저장했습니다.", status_code=302)


@router.get("/admin/config/models")
def admin_list_openai_models(request: Request):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    runtime_config = get_runtime_config()
    current_model = runtime_config["openai_model"]
    error = ""
    models = []
    try:
        models = list_available_models(api_key=runtime_config["openai_api_key"])
    except Exception:
        error = "모델 목록을 불러오지 못했습니다."
        models = []

    merged = [current_model] if current_model else []
    merged.extend(models)
    deduped_models = list(dict.fromkeys(item for item in merged if item))
    return JSONResponse(
        {
            "models": deduped_models,
            "current_model": current_model,
            "error": error,
        }
    )


@router.post("/admin/config/prompts")
def admin_update_prompt_config(
    request: Request,
    system_prompt: str = Form(...),
    user_prompt_template: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    cleaned_system = system_prompt.strip()
    cleaned_template = user_prompt_template.strip()
    if not cleaned_system:
        return RedirectResponse("/admin?error=시스템+프롬프트는+비워둘+수+없습니다.", status_code=302)

    ok, message = validate_prompt_template(cleaned_template)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(message)}", status_code=302)

    set_runtime_config(
        {
            "system_prompt": cleaned_system,
            "user_prompt_template": cleaned_template,
        },
        updated_by=user["id"],
    )
    return RedirectResponse("/admin?notice=프롬프트+설정을+저장했습니다.", status_code=302)


@router.post("/admin/config/patient-prompts")
def admin_update_patient_prompt_config(
    request: Request,
    patient_system_prompt: str = Form(...),
    patient_user_prompt_template: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    cleaned_system = patient_system_prompt.strip()
    cleaned_template = patient_user_prompt_template.strip()
    if not cleaned_system:
        return RedirectResponse("/admin?error=환자+시스템+프롬프트는+비워둘+수+없습니다.", status_code=302)

    ok, message = validate_patient_prompt_template(cleaned_template)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(message)}", status_code=302)

    set_runtime_config(
        {
            "patient_system_prompt": cleaned_system,
            "patient_user_prompt_template": cleaned_template,
        },
        updated_by=user["id"],
    )
    return RedirectResponse("/admin?notice=환자+응답+프롬프트(v2)+설정을+저장했습니다.", status_code=302)


def _backup_patient_prompts(runtime_config: dict[str, str], actor_id: int | None) -> Path:
    backup_dir = Path(settings.db_path).resolve().parent / "prompt_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"patient_prompt_backup_{stamp}.json"
    idx = 1
    while backup_path.exists():
        backup_path = backup_dir / f"patient_prompt_backup_{stamp}_{idx}.json"
        idx += 1

    patient_system_prompt = runtime_config.get("patient_system_prompt", "")
    patient_user_template = runtime_config.get("patient_user_prompt_template", "")
    previous_policy_hint = (
        "patient_v2_resistance_aware"
        if "[POLICY PRIORITY]" in patient_system_prompt and "[SESSION_META]" in patient_user_template
        else "custom_or_legacy"
    )
    payload = {
        "timestamp": now_kst_iso(),
        "actor_id": actor_id,
        "previous_policy_hint": previous_policy_hint,
        "previous": {
            "patient_system_prompt": patient_system_prompt,
            "patient_user_prompt_template": patient_user_template,
        },
    }
    backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return backup_path


@router.post("/admin/config/patient-prompts/apply-recommended")
def admin_apply_recommended_patient_prompts(
    request: Request,
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    runtime_config = get_runtime_config()
    try:
        backup_path = _backup_patient_prompts(runtime_config, user["id"])
    except OSError as exc:
        logger.warning(
            "patient_prompt_backup_failed %s",
            json.dumps(
                {
                    "user_id": user["id"],
                    "error_type": exc.__class__.__name__,
                },
                ensure_ascii=False,
            ),
        )
        return RedirectResponse("/admin?error=환자+프롬프트+백업에+실패했습니다.", status_code=302)

    set_runtime_config(
        {
            "patient_system_prompt": DEFAULT_PATIENT_SYSTEM_PROMPT,
            "patient_user_prompt_template": DEFAULT_PATIENT_USER_PROMPT_TEMPLATE,
        },
        updated_by=user["id"],
    )
    notice = quote_plus(
        "권장 v2 프롬프트 적용 완료 "
        f"(backup: {backup_path.name}, version: patient_v2_resistance_aware)"
    )
    return RedirectResponse(f"/admin?notice={notice}", status_code=302)


@router.get("/counselor/login")
def counselor_login_get(request: Request):
    return templates.TemplateResponse("counselor_login.html", {"request": request, "current_user": None})


@router.post("/counselor/auth")
def counselor_login_post(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_user_by_username(username.strip())
    if not user or user["role"] != "counselor" or not verify_pin(pin, user["pin_hash"]):
        return templates.TemplateResponse(
            "counselor_login.html",
            {"request": request, "current_user": None, "error": "로그인 정보가 올바르지 않습니다."},
        )
    request.session["user_id"] = user["id"]
    return RedirectResponse("/counselor", status_code=302)


@router.get("/counselor")
def counselor_get(request: Request, error: str = "", notice: str = ""):
    user = get_current_user(request)
    if not require_role(user, "counselor"):
        return RedirectResponse("/", status_code=302)
    patients = list_counselor_patients(user["id"])
    return templates.TemplateResponse(
        "counselor.html",
        {"request": request, "current_user": user, "users": patients, "error": error, "notice": notice},
    )


@router.post("/counselor/patients")
def counselor_create_patient(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
    full_name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    residence: str = Form(""),
    assistant_guidance: str = Form(""),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "counselor"):
        return RedirectResponse("/", status_code=302)

    ok, message, cleaned_username = validate_username(username)
    if not ok:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)

    ok, _ = validate_pin(pin)
    if not ok:
        return RedirectResponse("/counselor?error=PIN+형식을+확인하세요.", status_code=302)

    ok, message, cleaned_name = validate_full_name(full_name)
    if not ok:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)
    ok, message, age_value = validate_age(age)
    if not ok or age_value is None:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)
    ok, message, gender_value = validate_gender(gender)
    if not ok:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)
    ok, message, residence_value = validate_residence(residence)
    if not ok:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)
    ok, message, guidance_value = validate_assistant_guidance(assistant_guidance)
    if not ok:
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)

    try:
        patient_id = create_user(username=cleaned_username, role="patient", pin=pin)
        ensure_patient_consent_row(patient_id)
    except sqlite3.IntegrityError:
        existing = get_user_by_username(cleaned_username)
        if not existing or existing.get("role") != "patient":
            return RedirectResponse("/counselor?error=이미+사용+중인+아이디입니다.", status_code=302)
        ensure_patient_consent_row(int(existing["id"]))
        existing_id = int(existing["id"])
        ok, message = assign_patient_to_counselor(user["id"], existing_id)
        if not ok:
            return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)
        upsert_patient_profile(
            existing_id,
            full_name=cleaned_name,
            age=age_value,
            gender=gender_value,
            residence=residence_value,
            assistant_guidance=guidance_value,
            updated_by=user["id"],
        )
        return RedirectResponse("/counselor?notice=기존+환자를+배정하고+프로필을+저장했습니다.", status_code=302)

    upsert_patient_profile(
        patient_id,
        full_name=cleaned_name,
        age=age_value,
        gender=gender_value,
        residence=residence_value,
        assistant_guidance=guidance_value,
        updated_by=user["id"],
    )

    ok, message = assign_patient_to_counselor(user["id"], patient_id)
    if not ok:
        delete_user_account(patient_id)
        return RedirectResponse(f"/counselor?error={quote_plus(message)}", status_code=302)

    return RedirectResponse("/counselor?notice=환자를+추가하고+프로필을+저장했습니다.", status_code=302)


@router.post("/admin/users")
def admin_create_user(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
    full_name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    residence: str = Form(""),
    assistant_guidance: str = Form(""),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    ok, message, cleaned_username = validate_username(username)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(message)}", status_code=302)

    ok, msg = validate_pin(pin)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)

    ok, msg, cleaned_name = validate_full_name(full_name)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)
    ok, msg, age_value = validate_age(age)
    if not ok or age_value is None:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)
    ok, msg, gender_value = validate_gender(gender)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)
    ok, msg, residence_value = validate_residence(residence)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)
    ok, msg, guidance_value = validate_assistant_guidance(assistant_guidance)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)

    try:
        patient_id = create_user(username=cleaned_username, role="patient", pin=pin)
    except sqlite3.IntegrityError:
        return RedirectResponse("/admin?error=이미+사용+중인+아이디입니다.", status_code=302)
    ensure_patient_consent_row(patient_id)
    upsert_patient_profile(
        patient_id,
        full_name=cleaned_name,
        age=age_value,
        gender=gender_value,
        residence=residence_value,
        assistant_guidance=guidance_value,
        updated_by=user["id"],
    )
    return RedirectResponse("/admin?notice=환자+계정과+프로필을+생성했습니다.", status_code=302)


@router.post("/admin/counselors")
def admin_create_counselor(
    request: Request,
    username: str = Form(...),
    phone: str = Form(...),
    pin: str = Form(...),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    ok, msg, cleaned_username = validate_username(username)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)

    ok, msg = validate_pin(pin)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)

    ok, msg, normalized = validate_phone(phone)
    if not ok:
        return RedirectResponse(f"/admin?error={quote_plus(msg)}", status_code=302)

    try:
        create_user(username=cleaned_username, role="counselor", pin=pin, phone=normalized)
    except sqlite3.IntegrityError:
        return RedirectResponse("/admin?error=이미+사용+중인+상담자+코드/휴대폰입니다.", status_code=302)
    return RedirectResponse("/admin?notice=상담자+계정을+생성했습니다.", status_code=302)


@router.post("/admin/users/{user_id}/delete")
def admin_delete_user(request: Request, user_id: int, csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    target = get_user_by_id(user_id)
    if not target:
        return RedirectResponse("/admin?error=삭제할+계정을+찾을+수+없습니다.", status_code=302)
    if target.get("role") == "admin":
        return RedirectResponse("/admin?error=관리자+계정은+삭제할+수+없습니다.", status_code=302)

    delete_user_account(user_id)
    return RedirectResponse("/admin", status_code=302)


@router.get("/admin/user/{user_id}")
def admin_user_view(request: Request, user_id: int, back: Optional[str] = None, note_date: str = ""):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    target = get_user_by_id(user_id)
    if not target:
        return RedirectResponse("/admin", status_code=302)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT messages.id, messages.role, messages.content, messages.created_at,
                   emotion_scores.emotions_json,
                   emotion_scores.energy_angle,
                   emotion_scores.energy_magnitude,
                   risk_flags.score
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            LEFT JOIN emotion_scores ON emotion_scores.message_id = messages.id
            LEFT JOIN risk_flags ON risk_flags.message_id = messages.id
            WHERE sessions.user_id = ?
            ORDER BY messages.id ASC
            """,
            (user_id,),
        ).fetchall()
        messages = []
        for row in rows:
            top_emotion_key = _extract_top_emotion(row["emotions_json"])
            messages.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "created_at_display": _format_datetime_kst(row["created_at"]),
                    "primary_emotion_label_ko": emotion_label_ko(top_emotion_key),
                    "risk_score": row["score"],
                    "risk_score_display": _format_decimal(row["score"]),
                    "energy_magnitude_display": _format_decimal(row["energy_magnitude"]),
                    "energy_angle_display": _format_decimal(row["energy_angle"]),
                }
            )

    selected_note_date = ""
    note_filter_error = ""
    if target.get("role") == "patient":
        ok, message, parsed_date = validate_note_date(note_date)
        if ok:
            selected_note_date = parsed_date
        else:
            note_filter_error = message

    patient_profile = get_patient_profile(user_id) if target.get("role") == "patient" else None
    counselor_notes = (
        list_counselor_notes(user_id, note_date=selected_note_date or None) if target.get("role") == "patient" else []
    )
    note_groups = group_notes_by_date(counselor_notes) if target.get("role") == "patient" else []
    note_item_total = sum(len(group.get("items", [])) for group in note_groups)
    dashboard = build_patient_dashboard(user_id) if target.get("role") == "patient" else {
        "timeline_points": [],
        "daily_points": [],
        "latest_risk": None,
        "latest_risk_display": "-",
        "latest_energy": {
            "magnitude": None,
            "angle": None,
            "coherence": 0.0,
            "angle_reliable": False,
            "status": "zero_input",
            "active_count": 0,
        },
        "latest_energy_magnitude": None,
        "latest_energy_magnitude_display": "-",
        "last_message_at": None,
        "last_message_at_display": "-",
        "latest_emotion_profile": {emotion: 0.0 for emotion in EMOTIONS},
        "latest_emotion_profile_ko": _build_emotion_profile_ko({emotion: 0.0 for emotion in EMOTIONS}),
        "latest_emotion_labels_ko": {emotion: emotion_label_ko(emotion) for emotion in EMOTIONS},
        "latest_emotion_snapshot_profile": {emotion: 0.0 for emotion in EMOTIONS},
        "latest_emotion_snapshot_profile_ko": _build_emotion_profile_ko({emotion: 0.0 for emotion in EMOTIONS}),
        "latest_emotion_alignment": {
            **compute_emotion_alignment({emotion: 0.0 for emotion in EMOTIONS}, None, False, 0.0),
            "acute_override_applied": False,
            "snapshot_negative_emotion": None,
            "snapshot_negative_score": 0.0,
            "dominant_emotion_source": "state_top",
        },
        "dominant_emotion_source": "state_top",
        "emotion_volatility": 0.0,
        "emotion_volatility_display": "0.00",
        "emotion_volatility_band": "낮음",
        "risk_band": "데이터 없음",
        "energy_band": "데이터 없음",
        "kpi_explanations": {"risk": "", "energy": "", "angle": ""},
    }

    return templates.TemplateResponse(
        "admin_user.html",
        {
            "request": request,
            "current_user": user,
            "user": target,
            "messages": messages,
            "patient_profile": patient_profile,
            "counselor_notes": counselor_notes,
            "note_groups": note_groups,
            "note_item_total": note_item_total,
            "selected_note_date": selected_note_date,
            "note_filter_error": note_filter_error,
            "dashboard": dashboard,
            "back_url": back or "/admin",
        },
    )


@router.get("/counselor/user/{user_id}")
def counselor_user_view(request: Request, user_id: int, error: str = "", notice: str = "", note_date: str = ""):
    user = get_current_user(request)
    if not require_role(user, "counselor"):
        return RedirectResponse("/", status_code=302)

    target = get_user_by_id(user_id)
    if not target or target.get("role") != "patient":
        return RedirectResponse("/counselor", status_code=302)
    if not counselor_can_view_patient(user["id"], user_id):
        return RedirectResponse("/counselor?error=배정된+환자만+열람할+수+있습니다.", status_code=302)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT messages.id, messages.role, messages.content,
                   emotion_scores.emotions_json,
                   emotion_scores.energy_vx,
                   emotion_scores.energy_vy,
                   emotion_scores.energy_tau,
                   emotion_scores.energy_angle,
                   emotion_scores.energy_magnitude,
                   emotion_scores.energy_coherence,
                   emotion_scores.energy_angle_reliable,
                   emotion_scores.energy_active_count,
                   emotion_scores.emotion_source,
                   emotion_scores.emotion_error,
                   emotion_scores.energy_status,
                   assistant_message_meta.schema_valid AS response_schema_valid,
                   assistant_message_meta.generation_source AS response_generation_source,
                   assistant_message_meta.sanitizer_hit AS response_sanitizer_hit,
                   risk_flags.score
            FROM messages
            JOIN sessions ON sessions.id = messages.session_id
            LEFT JOIN emotion_scores ON emotion_scores.message_id = messages.id
            LEFT JOIN assistant_message_meta ON assistant_message_meta.assistant_message_id = messages.id
            LEFT JOIN risk_flags ON risk_flags.message_id = messages.id
            WHERE sessions.user_id = ?
            ORDER BY messages.id ASC
            """,
            (user_id,),
        ).fetchall()
        messages = []
        for row in rows:
            emotions = row["emotions_json"]
            has_energy = any(
                row[key] is not None
                for key in (
                    "energy_vx",
                    "energy_vy",
                    "energy_tau",
                    "energy_angle",
                    "energy_magnitude",
                    "energy_coherence",
                    "energy_active_count",
                    "energy_status",
                )
            )
            energy = {
                "v_x": row["energy_vx"],
                "v_y": row["energy_vy"],
                "tau": row["energy_tau"],
                "angle": row["energy_angle"],
                "magnitude": row["energy_magnitude"],
                "coherence": row["energy_coherence"],
                "angle_reliable": bool(row["energy_angle_reliable"]) if row["energy_angle_reliable"] is not None else False,
                "active_count": row["energy_active_count"],
            }
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "emotions": emotions,
                    "energy": energy if has_energy else None,
                    "emotion_source": row["emotion_source"],
                    "emotion_error": row["emotion_error"],
                    "energy_status": row["energy_status"],
                    "response_schema_valid": row["response_schema_valid"],
                    "response_generation_source": row["response_generation_source"],
                    "response_sanitizer_hit": row["response_sanitizer_hit"],
                    "risk_score": row["score"],
                }
            )

    selected_note_date = ""
    note_filter_error = ""
    ok, message, parsed_note_date = validate_note_date(note_date)
    if ok:
        selected_note_date = parsed_note_date
    else:
        note_filter_error = message

    patient_profile = get_patient_profile(user_id)
    counselor_notes = list_counselor_notes(user_id, note_date=selected_note_date or None)
    note_groups = group_notes_by_date(counselor_notes)
    dashboard = build_patient_dashboard(user_id)

    return templates.TemplateResponse(
        "counselor_user.html",
        {
            "request": request,
            "current_user": user,
            "user": target,
            "messages": messages,
            "patient_profile": patient_profile,
            "counselor_notes": counselor_notes,
            "note_groups": note_groups,
            "dashboard": dashboard,
            "error": error or note_filter_error,
            "notice": notice,
            "selected_note_date": selected_note_date,
            "back_url": "/counselor",
            "gender_options": [
                {"value": "male", "label": GENDER_LABELS["male"]},
                {"value": "female", "label": GENDER_LABELS["female"]},
                {"value": "other", "label": GENDER_LABELS["other"]},
                {"value": "unknown", "label": GENDER_LABELS["unknown"]},
            ],
        },
    )


@router.post("/counselor/user/{user_id}/profile")
def counselor_update_patient_profile(
    request: Request,
    user_id: int,
    full_name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    residence: str = Form(""),
    assistant_guidance: str = Form(""),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "counselor"):
        return RedirectResponse("/", status_code=302)
    target = get_user_by_id(user_id)
    if not target or target.get("role") != "patient":
        return RedirectResponse("/counselor?error=환자+계정을+찾을+수+없습니다.", status_code=302)
    if not counselor_can_view_patient(user["id"], user_id):
        return RedirectResponse("/counselor?error=배정된+환자만+수정할+수+있습니다.", status_code=302)

    ok, message, cleaned_name = validate_full_name(full_name)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)
    ok, message, age_value = validate_age(age)
    if not ok or age_value is None:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)
    ok, message, gender_value = validate_gender(gender)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)
    ok, message, residence_value = validate_residence(residence)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)
    ok, message, guidance_value = validate_assistant_guidance(assistant_guidance)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)

    upsert_patient_profile(
        user_id,
        full_name=cleaned_name,
        age=age_value,
        gender=gender_value,
        residence=residence_value,
        assistant_guidance=guidance_value,
        updated_by=user["id"],
    )
    return RedirectResponse(f"/counselor/user/{user_id}?notice=환자+프로필을+저장했습니다.", status_code=302)


@router.post("/counselor/user/{user_id}/notes")
def counselor_add_note(
    request: Request,
    user_id: int,
    session_summary: str = Form(...),
    intervention_note: str = Form(""),
    followup_plan: str = Form(""),
    counselor_risk_level: str = Form(""),
    note_date: str = Form(""),
    csrf_token: str = Form(""),
):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "counselor"):
        return RedirectResponse("/", status_code=302)
    target = get_user_by_id(user_id)
    if not target or target.get("role") != "patient":
        return RedirectResponse("/counselor?error=환자+계정을+찾을+수+없습니다.", status_code=302)
    if not counselor_can_view_patient(user["id"], user_id):
        return RedirectResponse("/counselor?error=배정된+환자만+기록할+수+있습니다.", status_code=302)

    summary = (session_summary or "").strip()
    if not summary:
        return RedirectResponse(f"/counselor/user/{user_id}?error=상담+요약을+입력하세요.", status_code=302)
    if len(summary) > 1000:
        return RedirectResponse(f"/counselor/user/{user_id}?error=상담+요약은+1000자+이하여야+합니다.", status_code=302)

    intervention = (intervention_note or "").strip()
    followup = (followup_plan or "").strip()
    ok, message, risk_level = validate_counselor_risk_level(counselor_risk_level)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)
    ok, message, note_date_value = validate_note_date(note_date)
    if not ok:
        return RedirectResponse(f"/counselor/user/{user_id}?error={quote_plus(message)}", status_code=302)

    created_at = _build_note_created_at_iso(note_date_value) if note_date_value else None

    add_counselor_note(
        patient_user_id=user_id,
        counselor_user_id=user["id"],
        session_summary=summary,
        intervention_note=intervention,
        followup_plan=followup,
        counselor_risk_level=risk_level,
        created_at=created_at,
    )
    if note_date_value:
        return RedirectResponse(
            f"/counselor/user/{user_id}?note_date={note_date_value}&notice=상담+기록을+저장했습니다.",
            status_code=302,
        )
    return RedirectResponse(f"/counselor/user/{user_id}?notice=상담+기록을+저장했습니다.", status_code=302)


@router.post("/admin/alerts/{alert_id}/resolve")
def resolve_alert(request: Request, alert_id: int, csrf_token: str = Form("")):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    now = now_kst_iso()
    with get_connection() as conn:
        conn.execute(
            "UPDATE risk_flags SET status = 'resolved', resolved_at = ? WHERE id = ?",
            (now, alert_id),
        )
        conn.commit()

    return RedirectResponse("/admin", status_code=302)


@router.get("/rag/reindex")
def rag_reindex(request: Request):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    result = reindex_documents()
    total_documents = int(result.get("total_documents", 0))
    indexed_documents = int(result.get("indexed_documents", 0))
    total_chunks = int(result.get("count", 0))
    source_mode = result.get("source_mode", "fallback")
    skipped_documents = result.get("skipped_documents", [])

    source_label = "업로드 폴더 우선" if source_mode == "upload" else "Fallback 폴더"
    if total_documents == 0:
        notice_text = str(result.get("message", "문서 재색인을 완료했습니다."))
    else:
        notice_text = (
            f"문서 재색인 완료: 대상 {total_documents}건, 색인 {indexed_documents}건, "
            f"미색인 {len(skipped_documents)}건, 총 청크 {total_chunks}개 ({source_label})"
        )

    warning_text = ""
    if skipped_documents:
        preview = ", ".join(item.get("title", "-") for item in skipped_documents[:3])
        suffix = " 등" if len(skipped_documents) > 3 else ""
        warning_text = f"미색인 문서: {preview}{suffix} (빈 텍스트/추출 실패는 자동 제외)"

    redirect_url = f"/admin/rag/index?notice={quote_plus(notice_text)}"
    if warning_text:
        redirect_url += f"&warn={quote_plus(warning_text)}"
    return RedirectResponse(redirect_url, status_code=302)


@router.get("/admin/rag/index")
def admin_rag_index(
    request: Request,
    doc_id: Optional[int] = None,
    error: str = "",
    notice: str = "",
    warn: str = "",
):
    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    with get_connection() as conn:
        docs = conn.execute(
            """
            SELECT documents.id, documents.title, documents.source, documents.created_at,
                   COUNT(doc_chunks.id) AS chunk_count
            FROM documents
            LEFT JOIN doc_chunks ON doc_chunks.doc_id = documents.id
            GROUP BY documents.id, documents.title, documents.source, documents.created_at
            ORDER BY documents.id DESC
            """
        ).fetchall()
        doc_items = [dict(row) for row in docs]

        selected_doc = None
        preview_chunks = []
        if doc_id is not None:
            selected_doc = conn.execute(
                "SELECT id, title, source, created_at FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if selected_doc:
                chunk_rows = conn.execute(
                    """
                    SELECT chunk_idx, chunk_text
                    FROM doc_chunks
                    WHERE doc_id = ?
                    ORDER BY chunk_idx ASC
                    LIMIT 5
                    """,
                    (doc_id,),
                ).fetchall()
                preview_chunks = [dict(row) for row in chunk_rows]

    upload_dir = ensure_upload_source_dir()
    parse_info = {
        "source_dir": str(upload_dir),
        "fallback_dir": settings.pdf_dir,
        "target_docs": ["*.pdf (전체 포함)", "*.docx (전체 포함)"],
        "selection_rule": "업로드 폴더에 문서가 있으면 업로드 폴더만 사용, 없으면 fallback 폴더 사용",
        "preferred_examples": sorted(PDF_NAMES),
        "extractor": "pdf: pypdf.PdfReader / docx: word/document.xml 파싱",
        "preprocess_rule": "줄바꿈 비율 > 0.01일 때만 '-\\n' 제거 + 줄바꿈/다중공백 정리",
        "chunking": "chunk_size=600, overlap=120 (문자 기준)",
        "embed_model": settings.openai_embed_model,
    }
    return templates.TemplateResponse(
        "admin_rag_index.html",
        {
            "request": request,
            "current_user": user,
            "docs": doc_items,
            "selected_doc": dict(selected_doc) if selected_doc else None,
            "preview_chunks": preview_chunks,
            "parse_info": parse_info,
            "users": list_users(),
            "error": error,
            "notice": notice,
            "warn": warn,
        },
    )


@router.post("/admin/rag/upload")
def admin_rag_upload(request: Request, csrf_token: str = Form(""), document: UploadFile = File(...)):
    csrf_error = enforce_csrf(request, csrf_token)
    if csrf_error:
        return csrf_error

    user = get_current_user(request)
    if not require_role(user, "admin"):
        return RedirectResponse("/", status_code=302)

    original_name = Path((document.filename or "").strip()).name
    if not original_name:
        return RedirectResponse("/admin/rag/index?error=업로드할+파일명을+확인하세요.", status_code=302)

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        return RedirectResponse("/admin/rag/index?error=PDF+또는+DOCX만+업로드할+수+있습니다.", status_code=302)

    source_dir = ensure_upload_source_dir()
    target_path = source_dir / original_name
    if target_path.exists():
        stem = target_path.stem
        ext = target_path.suffix
        idx = 1
        while True:
            candidate = source_dir / f"{stem}_{idx}{ext}"
            if not candidate.exists():
                target_path = candidate
                break
            idx += 1

    with target_path.open("wb") as fp:
        fp.write(document.file.read())

    notice = quote_plus(f"업로드 완료: {target_path.name}")
    return RedirectResponse(f"/admin/rag/index?notice={notice}", status_code=302)


@router.get("/export/csv")
def export_csv(request: Request, scope: str = "messages", user_id: Optional[int] = None):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    if user["role"] != "admin":
        scope = "messages"
        user_id = user["id"]

    output = io.StringIO()
    writer = csv.writer(output)

    if scope == "metrics":
        writer.writerow(
            [
                "user_id",
                "role",
                "content",
                "created_at",
                "emotions",
                "risk_score",
                "keywords",
                "energy_angle",
                "energy_magnitude",
                "energy_tau",
                "energy_vx",
                "energy_vy",
                "energy_coherence",
                "energy_angle_reliable",
                "energy_active_count",
                "emotion_source",
                "emotion_error",
                "energy_status",
            ]
        )
        with get_connection() as conn:
            params = []
            clause = ""
            if user_id:
                clause = "WHERE sessions.user_id = ?"
                params.append(user_id)
            rows = conn.execute(
                f"""
                SELECT sessions.user_id, messages.role, messages.content, messages.created_at,
                       emotion_scores.emotions_json, risk_flags.score, risk_flags.keywords_json,
                       emotion_scores.energy_angle, emotion_scores.energy_magnitude, emotion_scores.energy_tau,
                       emotion_scores.energy_vx, emotion_scores.energy_vy, emotion_scores.energy_coherence,
                       emotion_scores.energy_angle_reliable, emotion_scores.energy_active_count,
                       emotion_scores.emotion_source, emotion_scores.emotion_error, emotion_scores.energy_status
                FROM messages
                JOIN sessions ON sessions.id = messages.session_id
                LEFT JOIN emotion_scores ON emotion_scores.message_id = messages.id
                LEFT JOIN risk_flags ON risk_flags.message_id = messages.id
                {clause}
                ORDER BY messages.id ASC
                """,
                params,
            ).fetchall()
            for row in rows:
                writer.writerow(
                    [
                        row["user_id"],
                        row["role"],
                        row["content"],
                        row["created_at"],
                        row["emotions_json"],
                        row["score"],
                        row["keywords_json"],
                        row["energy_angle"],
                        row["energy_magnitude"],
                        row["energy_tau"],
                        row["energy_vx"],
                        row["energy_vy"],
                        row["energy_coherence"],
                        row["energy_angle_reliable"],
                        row["energy_active_count"],
                        row["emotion_source"],
                        row["emotion_error"],
                        row["energy_status"],
                    ]
                )
    else:
        writer.writerow(["user_id", "role", "content", "created_at"])
        with get_connection() as conn:
            params = []
            clause = ""
            if user_id:
                clause = "WHERE sessions.user_id = ?"
                params.append(user_id)
            rows = conn.execute(
                f"""
                SELECT sessions.user_id, messages.role, messages.content, messages.created_at
                FROM messages
                JOIN sessions ON sessions.id = messages.session_id
                {clause}
                ORDER BY messages.id ASC
                """,
                params,
            ).fetchall()
            for row in rows:
                writer.writerow([row["user_id"], row["role"], row["content"], row["created_at"]])

    output.seek(0)
    filename = "export_metrics.csv" if scope == "metrics" else "export_messages.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
