from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(
    __file__
).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = (
        "YOANEMI-XAI"
    )

    app_version: str = "0.1.0"
    environment: str = "development"
    max_upload_mb: int = 5

    unet_model_path: Path = Path(
        "weights/unet/unet_pediatric_final_best.pth"
    )

    yolov9_model_path: Path = Path(
        "weights/yolov9/yolov9t_final_best.pt"
    )

    yolov9_repository_path: Path = Path(
        "external/yolov9"
    )

    unet_image_size: int = 320
    unet_threshold: float = 0.45

    unet_minimum_mask_percentage: float = 0.01

    unet_keep_largest_component: bool = True

    unet_roi_margin_ratio: float = 0.0

    datasetp_image_size: int = 224
    yolov9_image_size: int = 384

    gradcam_overlay_alpha: float = 0.45

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def absolute_unet_model_path(
        self,
    ) -> Path:
        return self._absolute_path(
            self.unet_model_path
        )

    @property
    def absolute_yolov9_model_path(
        self,
    ) -> Path:
        return self._absolute_path(
            self.yolov9_model_path
        )

    @property
    def absolute_yolov9_repository_path(
        self,
    ) -> Path:
        return self._absolute_path(
            self.yolov9_repository_path
        )

    @staticmethod
    def _absolute_path(
        path: Path,
    ) -> Path:
        if path.is_absolute():
            return path

        return (
            PROJECT_DIR
            / path
        ).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()