import cv2
import numpy as np

from app.ml.preprocessing.exceptions import (
    InvalidMaskError,
)
from app.ml.preprocessing.types import (
    BoundingBox,
    SegmentedRegion,
    UInt8Image,
)


def normalize_binary_mask(
    binary_mask: np.ndarray,
) -> UInt8Image:
    """
    Convierte cualquier máscara binaria o probabilística
    previamente umbralizada en una matriz uint8 con valores 0 y 1.
    """

    if binary_mask.ndim != 2:
        raise InvalidMaskError(
            "La máscara debe tener dos dimensiones."
        )

    normalized_mask = (
        binary_mask > 0
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        normalized_mask
    )


def keep_largest_component(
    binary_mask: np.ndarray,
) -> UInt8Image:
    """
    Conserva únicamente el componente conectado de mayor área.
    """

    normalized_mask = normalize_binary_mask(
        binary_mask
    )

    (
        number_of_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        normalized_mask,
        connectivity=8,
    )

    if number_of_labels <= 1:
        raise InvalidMaskError(
            "La máscara no contiene ninguna región segmentada."
        )

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_label = (
        int(
            np.argmax(
                component_areas
            )
        )
        + 1
    )

    largest_component = (
        labels
        == largest_label
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        largest_component
    )


def calculate_mask_percentage(
    binary_mask: np.ndarray,
) -> float:
    """
    Calcula el porcentaje de píxeles pertenecientes
    a la región segmentada.
    """

    normalized_mask = normalize_binary_mask(
        binary_mask
    )

    segmented_pixels = int(
        normalized_mask.sum()
    )

    total_pixels = int(
        normalized_mask.size
    )

    if total_pixels == 0:
        raise InvalidMaskError(
            "La máscara no contiene píxeles."
        )

    return (
        segmented_pixels
        / total_pixels
        * 100.0
    )


def calculate_roi_bounds(
    binary_mask: np.ndarray,
    margin_ratio: float = 0.0,
) -> BoundingBox:
    """
    Calcula el rectángulo mínimo que contiene la máscara.

    Retorna:
        x1, y1, x2, y2
    """

    if margin_ratio < 0:
        raise ValueError(
            "El margen no puede ser negativo."
        )

    normalized_mask = normalize_binary_mask(
        binary_mask
    )

    coordinates = cv2.findNonZero(
        normalized_mask
    )

    if coordinates is None:
        raise InvalidMaskError(
            "La máscara no contiene una región válida."
        )

    (
        x,
        y,
        width,
        height,
    ) = cv2.boundingRect(
        coordinates
    )

    image_height, image_width = (
        normalized_mask.shape
    )

    horizontal_margin = int(
        round(
            width
            * margin_ratio
        )
    )

    vertical_margin = int(
        round(
            height
            * margin_ratio
        )
    )

    x1 = max(
        0,
        x - horizontal_margin,
    )

    y1 = max(
        0,
        y - vertical_margin,
    )

    x2 = min(
        image_width,
        x + width + horizontal_margin,
    )

    y2 = min(
        image_height,
        y + height + vertical_margin,
    )

    if x2 <= x1 or y2 <= y1:
        raise InvalidMaskError(
            "El área calculada para la conjuntiva es inválida."
        )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def extract_segmented_region(
    original_rgb: UInt8Image,
    binary_mask: np.ndarray,
    margin_ratio: float = 0.0,
) -> SegmentedRegion:
    """
    Recorta la región de la conjuntiva y coloca en negro
    los píxeles externos a la máscara.
    """

    if (
        original_rgb.ndim != 3
        or original_rgb.shape[2] != 3
    ):
        raise ValueError(
            "La imagen original debe tener formato RGB."
        )

    normalized_mask = normalize_binary_mask(
        binary_mask
    )

    if (
        original_rgb.shape[:2]
        != normalized_mask.shape
    ):
        raise InvalidMaskError(
            "La máscara y la imagen original "
            "tienen dimensiones diferentes."
        )

    (
        x1,
        y1,
        x2,
        y2,
    ) = calculate_roi_bounds(
        normalized_mask,
        margin_ratio=margin_ratio,
    )

    original_crop = original_rgb[
        y1:y2,
        x1:x2,
    ].copy()

    mask_crop = normalized_mask[
        y1:y2,
        x1:x2,
    ].copy()

    if (
        original_crop.size == 0
        or mask_crop.size == 0
    ):
        raise InvalidMaskError(
            "El recorte de la conjuntiva está vacío."
        )

    segmented_crop = np.zeros_like(
        original_crop,
        dtype=np.uint8,
    )

    segmented_crop[
        mask_crop == 1
    ] = original_crop[
        mask_crop == 1
    ]

    return SegmentedRegion(
        bounding_box=(
            x1,
            y1,
            x2,
            y2,
        ),
        original_crop=np.ascontiguousarray(
            original_crop
        ),
        mask_crop=np.ascontiguousarray(
            mask_crop
        ),
        segmented_crop=np.ascontiguousarray(
            segmented_crop
        ),
    )