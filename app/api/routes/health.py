from fastapi import APIRouter, Request

from app.ml.registry import ModelRegistry
from app.schemas.health import HealthResponse


router = APIRouter(
    tags=["Health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check(
    request: Request,
) -> HealthResponse:
    registry: ModelRegistry | None = getattr(
        request.app.state,
        "model_registry",
        None,
    )

    if registry is None:
        return HealthResponse(
            status="starting",
            models_ready=False,
            device=None,
            unet_loaded=False,
            yolov9_loaded=False,
            classes=[],
        )

    return HealthResponse(
        status=(
            "ok"
            if registry.is_ready
            else "unavailable"
        ),
        models_ready=registry.is_ready,
        device=str(
            registry.device
        ),
        unet_loaded=registry.unet.is_loaded,
        yolov9_loaded=registry.yolov9.is_loaded,
        classes=(
            registry.yolov9.class_names
            if registry.yolov9.is_loaded
            else []
        ),
    )