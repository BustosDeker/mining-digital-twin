"""Aplicación FastAPI: gemelo digital de evacuación minera.

Expone en tiempo real (WebSocket) y bajo demanda (REST) todo lo construido
en las fases anteriores: gemelo digital + ABM (Fases 2-4), Motor IA
completo (Fases 5-8). El frontend Next.js consume exclusivamente esta API.

Arranque local:
    uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes_ml, routes_sessions, routes_simulation, websocket
from backend.utils.config import get_settings
from backend.utils.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("API iniciada", extra={"environment": settings.ENVIRONMENT})
    yield


app = FastAPI(
    title="Gemelo Digital de Evacuación Minera — API",
    description=(
        "Backend de producción: sirve el gemelo digital en tiempo real, la "
        "inferencia del clasificador de estrés, resultados de simulación ABM, "
        "historial de sesiones y descarga de reportes. El entrenamiento vive "
        "en el Motor IA (training/), nunca en el frontend."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_simulation.router)
app.include_router(routes_simulation.mc_router)
app.include_router(routes_ml.router)
app.include_router(routes_sessions.router)
app.include_router(websocket.router)


@app.get("/api/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok", "app_name": settings.APP_NAME, "environment": settings.ENVIRONMENT}
