from pathlib import Path

from app.core.config import Settings


class ModelFileValidationError(
    RuntimeError
):
    pass


def validate_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise ModelFileValidationError(
            f"No se encontró {description}: {path}"
        )

    if not path.is_file():
        raise ModelFileValidationError(
            f"La ruta de {description} "
            f"no corresponde a un archivo: {path}"
        )

    if path.stat().st_size == 0:
        raise ModelFileValidationError(
            f"El archivo de {description} está vacío: {path}"
        )


def validate_directory(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise ModelFileValidationError(
            f"No se encontró {description}: {path}"
        )

    if not path.is_dir():
        raise ModelFileValidationError(
            f"La ruta de {description} "
            f"no corresponde a un directorio: {path}"
        )


def validate_model_files(
    settings: Settings,
) -> None:
    validate_file(
        settings.absolute_unet_model_path,
        "el checkpoint de U-Net",
    )

    validate_file(
        settings.absolute_yolov9_model_path,
        "el checkpoint de YOLOv9-T",
    )

    validate_directory(
        settings.absolute_yolov9_repository_path,
        "el repositorio de YOLOv9",
    )

    validate_file(
        settings.absolute_yolov9_repository_path
        / "models"
        / "yolo.py",
        "models/yolo.py de YOLOv9",
    )

    validate_file(
        settings.absolute_yolov9_repository_path
        / "models"
        / "common.py",
        "models/common.py de YOLOv9",
    )