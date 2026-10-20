from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings


APP_DIRECTORY = Path(
    __file__
).resolve().parents[2]

TEMPLATES_DIRECTORY = (
    APP_DIRECTORY
    / "templates"
)

templates = Jinja2Templates(
    directory=str(
        TEMPLATES_DIRECTORY
    )
)


router = APIRouter(
    include_in_schema=False,
)


@router.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
) -> HTMLResponse:
    settings = get_settings()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "max_upload_mb": settings.max_upload_mb,
        },
    )
