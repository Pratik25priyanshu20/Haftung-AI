"""FastAPI application for Haftung_AI."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from haftung_ai.api.routes import analyze, health, report, stream

app = FastAPI(
    title="Haftung_AI",
    description="Automated accident causation analysis API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(report.router, prefix="/report", tags=["report"])
app.include_router(stream.router, prefix="/stream", tags=["stream"])
