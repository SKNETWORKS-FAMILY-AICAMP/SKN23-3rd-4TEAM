from __future__ import annotations

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import app.web_routes as web_routes_module
from app.settings import settings
from app.domain_services import issue_csrf_token, lifespan, templates
from app.domain_services import _format_datetime_kst, _format_decimal
from app.web_routes import router

# Re-export helpers and route handlers for existing tests/import paths.
from app.domain_services import *  # noqa: F401,F403,E402
from app.web_routes import *  # noqa: F401,F403,E402


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates.env.globals["csrf_token_for"] = issue_csrf_token
app.include_router(router)


def _sync_route_runtime_hooks() -> None:
    # Keep monkeypatch compatibility for tests that patch app.main symbols.
    web_routes_module.list_available_models = list_available_models
    web_routes_module.chat_json = chat_json
    web_routes_module.search_rag = search_rag
    web_routes_module.index_message_embedding = index_message_embedding
    web_routes_module.analyze_emotions = analyze_emotions
    web_routes_module.analyze_risk = analyze_risk


def admin_list_openai_models(request: Request):
    _sync_route_runtime_hooks()
    return web_routes_module.admin_list_openai_models(request)


def chat_send(request: Request, content: str = Form(...), csrf_token: str = Form("")):
    _sync_route_runtime_hooks()
    return web_routes_module.chat_send(request, content=content, csrf_token=csrf_token)
