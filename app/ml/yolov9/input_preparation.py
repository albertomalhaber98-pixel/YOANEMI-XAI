import numpy as np
import torch

from app.ml.preprocessing.letterbox import (
    letterbox_mask,
    letterbox_rgb,
)
from app.ml.preprocessing.types import UInt8Image
from app.ml.yolov9.exceptions import (
    YoloV9InferenceError,
)
from app.ml.yolov9.types import (
    YoloV9PreparedInput,
)


class YoloV9InputPreparer:
    def __init__(
        self,
        device: torch.device,
        datasetp_image_size: int,
        model_image_size: int,
    ) -> None:
        if datasetp_image_size <= 0:
            raise ValueError(
                "El tamaño de DatasetP debe ser mayor que cero."
            )

        if model_image_size <= 0:
            raise ValueError(
                "El tamaño de YOLOv9-T debe ser mayor que cero."
            )

        self._device = device
        self._datasetp_image_size = (
            datasetp_image_size
        )

        self._model_image_size = (
            model_image_size
        )

    def prepare(
        self,
        segmented_rgb: UInt8Image,
        tissue_mask: UInt8Image,
    ) -> YoloV9PreparedInput:
        self._validate_image(
            segmented_rgb
        )

        self._validate_mask(
            tissue_mask=tissue_mask,
            expected_shape=segmented_rgb.shape[:2],
        )

        datasetp_image = letterbox_rgb(
            rgb_image=segmented_rgb,
            output_size=self._datasetp_image_size,
        )

        datasetp_tissue_mask = letterbox_mask(
            binary_mask=tissue_mask,
            output_size=self._datasetp_image_size,
        )

        model_image = letterbox_rgb(
            rgb_image=datasetp_image,
            output_size=self._model_image_size,
        )

        model_tissue_mask = letterbox_mask(
            binary_mask=datasetp_tissue_mask,
            output_size=self._model_image_size,
        )

        tensor = (
            torch.from_numpy(
                model_image.copy()
            )
            .permute(
                2,
                0,
                1,
            )
            .contiguous()
            .float()
            .div(
                255.0
            )
            .unsqueeze(
                0
            )
            .to(
                self._device
            )
        )

        return YoloV9PreparedInput(
            datasetp_image=np.ascontiguousarray(
                datasetp_image
            ),
            datasetp_tissue_mask=np.ascontiguousarray(
                datasetp_tissue_mask
            ),
            model_image=np.ascontiguousarray(
                model_image
            ),
            model_tissue_mask=np.ascontiguousarray(
                model_tissue_mask
            ),
            tensor=tensor,
        )

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> None:
        if not isinstance(
            image,
            np.ndarray,
        ):
            raise YoloV9InferenceError(
                "La entrada debe ser una matriz NumPy."
            )

        if image.size == 0:
            raise YoloV9InferenceError(
                "La imagen segmentada está vacía."
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):
            raise YoloV9InferenceError(
                "La imagen segmentada debe tener formato RGB."
            )

        if image.dtype != np.uint8:
            raise YoloV9InferenceError(
                "La imagen segmentada debe utilizar uint8."
            )

    @staticmethod
    def _validate_mask(
        tissue_mask: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> None:
        if not isinstance(
            tissue_mask,
            np.ndarray,
        ):
            raise YoloV9InferenceError(
                "La máscara debe ser una matriz NumPy."
            )

        if tissue_mask.ndim != 2:
            raise YoloV9InferenceError(
                "La máscara debe tener dos dimensiones."
            )

        if tissue_mask.shape != expected_shape:
            raise YoloV9InferenceError(
                "La máscara y el recorte segmentado "
                "tienen dimensiones diferentes."
            )

        if not np.any(
            tissue_mask > 0
        ):
            raise YoloV9InferenceError(
                "La máscara de tejido está vacía."
            )