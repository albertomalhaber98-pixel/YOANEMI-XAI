import cv2
import numpy as np
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

from app.ml.preprocessing.mask_processing import (
    calculate_mask_percentage,
    keep_largest_component,
    normalize_binary_mask,
)
from app.ml.preprocessing.types import (
    UInt8Image,
)
from app.ml.unet.exceptions import (
    UnetInferenceError,
)
from app.ml.unet.model import (
    UnetModel,
)
from app.ml.unet.types import (
    UnetSegmentationResult,
)


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class UnetSegmenter:
    """
    Ejecuta la segmentación de la conjuntiva utilizando U-Net.

    Esta clase no carga el checkpoint y no guarda resultados.
    Solamente utiliza el modelo previamente cargado por UnetModel.
    """

    def __init__(
        self,
        unet_model: UnetModel,
        device: torch.device,
        image_size: int,
        threshold: float,
        minimum_mask_percentage: float = 0.01,
        keep_only_largest_component: bool = True,
    ) -> None:
        if image_size <= 0:
            raise ValueError(
                "El tamaño de entrada de U-Net "
                "debe ser mayor que cero."
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "El threshold de U-Net debe estar "
                "entre 0 y 1."
            )

        if minimum_mask_percentage < 0.0:
            raise ValueError(
                "El porcentaje mínimo de máscara "
                "no puede ser negativo."
            )

        self._unet_model = unet_model
        self._device = device
        self._image_size = image_size
        self._threshold = threshold
        self._minimum_mask_percentage = (
            minimum_mask_percentage
        )
        self._keep_only_largest_component = (
            keep_only_largest_component
        )

        self._transform = self._build_transform()

    @property
    def threshold(
        self,
    ) -> float:
        return self._threshold

    @property
    def image_size(
        self,
    ) -> int:
        return self._image_size

    def segment(
        self,
        original_rgb: UInt8Image,
    ) -> UnetSegmentationResult:
        """
        Segmenta la conjuntiva sobre una imagen RGB.

        Retorna el mapa de probabilidad y la máscara con
        las dimensiones originales de la imagen.
        """

        self._validate_image(
            original_rgb
        )

        if not self._unet_model.is_loaded:
            raise UnetInferenceError(
                "No se puede ejecutar U-Net porque "
                "el modelo todavía no ha sido cargado."
            )

        original_height, original_width = (
            original_rgb.shape[:2]
        )

        transformed = self._transform(
            image=original_rgb
        )

        input_tensor = (
            transformed[
                "image"
            ]
            .float()
            .unsqueeze(
                0
            )
            .to(
                self._device
            )
        )

        try:
            with torch.inference_mode():
                logits = self._unet_model.model(
                    input_tensor
                )

        except Exception as error:
            raise UnetInferenceError(
                "U-Net no pudo procesar la imagen."
            ) from error

        logits = self._extract_logits(
            logits
        )

        self._validate_output(
            logits
        )

        probability_small = (
            torch.sigmoid(
                logits
            )[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        probability_map = cv2.resize(
            probability_small,
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        ).astype(
            np.float32
        )

        probability_map = np.clip(
            probability_map,
            0.0,
            1.0,
        )

        raw_binary_mask = (
            probability_map
            >= self._threshold
        ).astype(
            np.uint8
        )

        if (
            self._keep_only_largest_component
        ):
            try:
                binary_mask = keep_largest_component(
                    raw_binary_mask
                )

            except Exception as error:
                raise UnetInferenceError(
                    "U-Net no detectó una región "
                    "válida de la conjuntiva."
                ) from error

        else:
            binary_mask = normalize_binary_mask(
                raw_binary_mask
            )

        segmented_percentage = (
            calculate_mask_percentage(
                binary_mask
            )
        )

        if (
            segmented_percentage
            < self._minimum_mask_percentage
        ):
            raise UnetInferenceError(
                "La región detectada por U-Net es "
                "demasiado pequeña. "
                f"Área encontrada: "
                f"{segmented_percentage:.6f}%."
            )

        return UnetSegmentationResult(
            probability_map=np.ascontiguousarray(
                probability_map
            ),
            binary_mask=np.ascontiguousarray(
                binary_mask
            ),
            segmented_percentage=float(
                segmented_percentage
            ),
        )

    def _build_transform(
        self,
    ) -> A.Compose:
        """
        Transformación idéntica a la utilizada durante
        las pruebas del modelo U-Net.
        """

        return A.Compose(
            [
                A.Resize(
                    height=self._image_size,
                    width=self._image_size,
                    interpolation=cv2.INTER_LINEAR,
                ),

                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(
                        8,
                        8,
                    ),
                    p=1.0,
                ),

                A.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),

                ToTensorV2(),
            ]
        )

    @staticmethod
    def _extract_logits(
        output: object,
    ) -> torch.Tensor:
        """
        Algunos modelos pueden devolver directamente un tensor
        o envolverlo dentro de una tupla o lista.
        """

        if isinstance(
            output,
            (
                tuple,
                list,
            ),
        ):
            if not output:
                raise UnetInferenceError(
                    "U-Net devolvió una salida vacía."
                )

            output = output[0]

        if not torch.is_tensor(
            output
        ):
            raise UnetInferenceError(
                "U-Net no devolvió un tensor válido."
            )

        return output

    @staticmethod
    def _validate_output(
        logits: torch.Tensor,
    ) -> None:
        expected_dimensions = 4

        if logits.ndim != expected_dimensions:
            raise UnetInferenceError(
                "La salida de U-Net tiene una forma "
                "inesperada. "
                f"Recibido: {tuple(logits.shape)}."
            )

        if logits.shape[0] != 1:
            raise UnetInferenceError(
                "La inferencia individual esperaba "
                "un batch de tamaño 1."
            )

        if logits.shape[1] != 1:
            raise UnetInferenceError(
                "U-Net debe producir un único canal "
                "de segmentación."
            )

    @staticmethod
    def _validate_image(
        original_rgb: np.ndarray,
    ) -> None:
        if not isinstance(
            original_rgb,
            np.ndarray,
        ):
            raise UnetInferenceError(
                "La entrada de U-Net debe ser "
                "una matriz NumPy."
            )

        if original_rgb.size == 0:
            raise UnetInferenceError(
                "La imagen de entrada está vacía."
            )

        if (
            original_rgb.ndim != 3
            or original_rgb.shape[2] != 3
        ):
            raise UnetInferenceError(
                "La imagen de entrada debe tener "
                "formato RGB."
            )

        if original_rgb.dtype != np.uint8:
            raise UnetInferenceError(
                "La imagen de entrada debe usar "
                "el tipo uint8."
            )
