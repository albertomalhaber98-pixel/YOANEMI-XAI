from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    device: str | None
    unet_loaded: bool
    yolov9_loaded: bool
    classes: list[str]