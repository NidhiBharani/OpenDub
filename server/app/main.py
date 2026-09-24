"""OpenDub server — app factory. Serves the API under /api and the built web UI (web/dist) at /."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .api import capabilities, events, jobs, media, projects, providers, segments, versions
from .jobs import engine
from .providers.base import load_all

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    engine.start()
    yield
    await engine.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="OpenDub", version=VERSION, lifespan=lifespan)
    # The UI is served same-origin in production; CORS exists only for the Vite dev server.
    # Keep it scoped so arbitrary websites can't read a locally-hosted instance's data.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "version": VERSION}

    for router in (projects.router, segments.router, jobs.router, providers.router,
                   capabilities.router, versions.router,
                   media.router, events.router):
        app.include_router(router, prefix="/api")

    if config.WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=config.WEB_DIST, html=True), name="web")
    return app


app = create_app()
