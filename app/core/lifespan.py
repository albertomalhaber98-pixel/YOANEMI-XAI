from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.ml.registry import ModelRegistry
from app.services import (
    AnemiaAnalysisService,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    settings = get_settings()

    registry = ModelRegistry(
        settings=settings
    )

    app.state.model_registry = registry
    app.state.analysis_service = None
    app.state.models_ready = False

    try:
        registry.load_all()

        analysis_service = (
            AnemiaAnalysisService(
                registry=registry,
                settings=settings,
            )
        )

        app.state.analysis_service = (
            analysis_service
        )

        app.state.models_ready = (
            registry.is_ready
        )

        yield

    finally:
        app.state.analysis_service = None
        app.state.models_ready = False

        registry.unload_all()

        app.state.model_registry = None
