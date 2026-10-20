from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.middlewares.privacy import (
    PrivacyHeadersMiddleware,
)


APP_DIR = Path(
    __file__
).resolve().parent

STATIC_DIR = (
    APP_DIR
    / "static"
)


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    application.add_middleware(
        PrivacyHeadersMiddleware,
    )

    application.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static",
    )

    application.include_router(
        router
    )

    return application


app = create_app()