import torch

from app.core.config import Settings
from app.ml.device import (
    get_device_description,
    resolve_device,
)
from app.ml.unet import (
    UnetModel,
    UnetSegmenter,
)
from app.ml.yolov9 import (
    YoloV9Classifier,
    YoloV9InputPreparer,
    YoloV9Model,
)

from app.ml.gradcam import (
    GradCamPlusPlusExplainer,
)


class ModelRegistry:
    """
    Registro central de modelos.

    Los modelos se cargan una sola vez cuando FastAPI inicia
    y se reutilizan durante todas las peticiones.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings
        self.device = resolve_device()

        self.unet = UnetModel(
            checkpoint_path=(
                settings.absolute_unet_model_path
            ),
            device=self.device,
        )

        self.unet_segmenter = UnetSegmenter(
            unet_model=self.unet,
            device=self.device,
            image_size=settings.unet_image_size,
            threshold=settings.unet_threshold,
            minimum_mask_percentage=(
                settings.unet_minimum_mask_percentage
            ),
            keep_only_largest_component=(
                settings.unet_keep_largest_component
            ),
        )

        self.yolov9 = YoloV9Model(
            checkpoint_path=(
                settings.absolute_yolov9_model_path
            ),
            repository_path=(
                settings.absolute_yolov9_repository_path
            ),
            device=self.device,
        )

        self.yolov9_input_preparer = YoloV9InputPreparer(
            device=self.device,
            datasetp_image_size=settings.datasetp_image_size,
            model_image_size=settings.yolov9_image_size,
        )

        self.yolov9_classifier = YoloV9Classifier(
            yolo_model=self.yolov9,
        )

        self.gradcam_explainer = (
            GradCamPlusPlusExplainer(
                yolo_model=self.yolov9,
                overlay_alpha=(
                    settings.gradcam_overlay_alpha
                ),
            )
        )

        self._ready = False

    @property
    def is_ready(
        self,
    ) -> bool:
        return (
            self._ready
            and self.unet.is_loaded
            and self.yolov9.is_loaded
        )

    @property
    def device_description(
        self,
    ) -> str:
        return get_device_description(
            self.device
        )

    def load_all(
        self,
    ) -> None:
        print(
            "\nCargando modelos de machine learning..."
        )

        print(
            f"Dispositivo: "
            f"{self.device_description}"
        )

        self.unet.load()

        print(
            "U-Net cargada correctamente."
        )

        self.yolov9.load()

        print(
            "YOLOv9-T cargado correctamente."
        )

        print(
            f"Clases: "
            f"{self.yolov9.class_names}"
        )

        self._ready = True

    def unload_all(
        self,
    ) -> None:
        self.unet.unload()
        self.yolov9.unload()

        self._ready = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()