import importlib
import sys
from pathlib import Path

import torch
from torch import nn


class YoloV9ModelLoadError(
    RuntimeError
):
    pass


class YoloV9Model:
    """
    Responsable de preparar el repositorio oficial
    y cargar el clasificador YOLOv9-T entrenado.
    """

    REQUIRED_MODULES = {
        "ELAN1",
        "AConv",
        "RepNCSPELAN4",
        "Classify",
    }

    def __init__(
        self,
        checkpoint_path: Path,
        repository_path: Path,
        device: torch.device,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._repository_path = repository_path
        self._device = device

        self._model: nn.Module | None = None
        self._class_names: list[str] = []

    @property
    def model(
        self,
    ) -> nn.Module:
        if self._model is None:
            raise YoloV9ModelLoadError(
                "YOLOv9-T todavía no ha sido cargado."
            )

        return self._model

    @property
    def class_names(
        self,
    ) -> list[str]:
        if not self._class_names:
            raise YoloV9ModelLoadError(
                "Las clases de YOLOv9-T "
                "todavía no están disponibles."
            )

        return self._class_names.copy()

    @property
    def is_loaded(
        self,
    ) -> bool:
        return self._model is not None

    def load(
        self,
    ) -> None:
        self._validate_paths()
        self._prepare_repository_imports()

        checkpoint = torch.load(
            self._checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise YoloV9ModelLoadError(
                "El checkpoint de YOLOv9-T "
                "no es un diccionario."
            )

        if "model" not in checkpoint:
            raise YoloV9ModelLoadError(
                "El checkpoint de YOLOv9-T "
                "no contiene la clave 'model'."
            )

        loaded_model = checkpoint[
            "model"
        ]

        if not isinstance(
            loaded_model,
            nn.Module,
        ):
            raise YoloV9ModelLoadError(
                "El objeto contenido en 'model' "
                "no es un modelo de PyTorch."
            )

        model = (
            loaded_model
            .float()
            .to(
                self._device
            )
        )

        model.eval()

        class_names = self._extract_class_names(
            checkpoint=checkpoint,
            model=model,
        )

        self._validate_architecture(
            model
        )

        self._model = model
        self._class_names = class_names

    def unload(
        self,
    ) -> None:
        self._model = None
        self._class_names = []

    def _validate_paths(
        self,
    ) -> None:
        if not self._checkpoint_path.is_file():
            raise YoloV9ModelLoadError(
                "No se encontró el checkpoint "
                "final de YOLOv9-T: "
                f"{self._checkpoint_path}"
            )

        if not self._repository_path.is_dir():
            raise YoloV9ModelLoadError(
                "No se encontró el repositorio "
                "oficial de YOLOv9: "
                f"{self._repository_path}"
            )

        required_files = [
            self._repository_path
            / "models"
            / "yolo.py",

            self._repository_path
            / "models"
            / "common.py",

            self._repository_path
            / "utils"
            / "general.py",
        ]

        for required_file in required_files:
            if not required_file.is_file():
                raise YoloV9ModelLoadError(
                    "Falta un archivo requerido "
                    "del repositorio YOLOv9: "
                    f"{required_file}"
                )

    def _prepare_repository_imports(
        self,
    ) -> None:
        repository_string = str(
            self._repository_path
        )

        if repository_string in sys.path:
            sys.path.remove(
                repository_string
            )

        sys.path.insert(
            0,
            repository_string,
        )

        # Evitar que otro paquete llamado models o utils
        # interfiera con el repositorio oficial.
        for module_name in list(
            sys.modules.keys()
        ):
            if (
                module_name == "models"
                or module_name.startswith(
                    "models."
                )
                or module_name == "utils"
                or module_name.startswith(
                    "utils."
                )
            ):
                del sys.modules[
                    module_name
                ]

        importlib.invalidate_caches()

        try:
            importlib.import_module(
                "models.yolo"
            )

        except Exception as error:
            raise YoloV9ModelLoadError(
                "No fue posible importar la arquitectura "
                "oficial de YOLOv9. "
                f"Causa original: {type(error).__name__}: {error}"
            ) from error

    def _validate_architecture(
        self,
        model: nn.Module,
    ) -> None:
        if not hasattr(
            model,
            "model",
        ):
            raise YoloV9ModelLoadError(
                "El checkpoint no contiene una "
                "arquitectura YOLOv9 válida."
            )

        top_level_modules = {
            module.__class__.__name__
            for module
            in model.model
        }

        missing_modules = (
            self.REQUIRED_MODULES
            - top_level_modules
        )

        if missing_modules:
            raise YoloV9ModelLoadError(
                "La arquitectura cargada no contiene "
                "todos los módulos de YOLOv9-T. "
                f"Faltan: {sorted(missing_modules)}"
            )

        if "C2f" in top_level_modules:
            raise YoloV9ModelLoadError(
                "Se encontró el bloque C2f. "
                "El checkpoint podría corresponder "
                "a YOLOv8 y no a YOLOv9-T."
            )

    @staticmethod
    def _extract_class_names(
        checkpoint: dict,
        model: nn.Module,
    ) -> list[str]:
        class_names = checkpoint.get(
            "class_names"
        )

        if class_names is None:
            class_names = getattr(
                model,
                "names",
                None,
            )

        if isinstance(
            class_names,
            dict,
        ):
            normalized_names = [
                str(
                    class_names[index]
                )
                for index in sorted(
                    class_names.keys()
                )
            ]

        elif isinstance(
            class_names,
            (
                list,
                tuple,
            ),
        ):
            normalized_names = [
                str(
                    class_name
                )
                for class_name
                in class_names
            ]

        else:
            raise YoloV9ModelLoadError(
                "No se encontraron los nombres "
                "de las clases en YOLOv9-T."
            )

        if len(
            normalized_names
        ) != 2:
            raise YoloV9ModelLoadError(
                "Se esperaban dos clases, pero "
                f"se encontraron: {normalized_names}"
            )

        return normalized_names