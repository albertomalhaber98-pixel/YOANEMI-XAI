from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.ml.preprocessing.types import UInt8Image


if hasattr(Image, "Resampling"):
    BILINEAR_RESAMPLING = Image.Resampling.BILINEAR
else:
    BILINEAR_RESAMPLING = Image.BILINEAR


@dataclass(slots=True, frozen=True)
class LetterboxGeometry:
    original_width: int
    original_height: int

    output_size: int

    resized_width: int
    resized_height: int

    left: int
    top: int


def calculate_letterbox_geometry(
    original_width: int,
    original_height: int,
    output_size: int,
) -> LetterboxGeometry:
    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            "Las dimensiones originales deben ser mayores que cero."
        )

    if output_size <= 0:
        raise ValueError(
            "El tamaño de salida debe ser mayor que cero."
        )

    scale = min(
        output_size / original_width,
        output_size / original_height,
    )

    resized_width = max(
        1,
        int(round(original_width * scale)),
    )

    resized_height = max(
        1,
        int(round(original_height * scale)),
    )

    left = (
        output_size
        - resized_width
    ) // 2

    top = (
        output_size
        - resized_height
    ) // 2

    return LetterboxGeometry(
        original_width=original_width,
        original_height=original_height,
        output_size=output_size,
        resized_width=resized_width,
        resized_height=resized_height,
        left=left,
        top=top,
    )


def letterbox_rgb(
    rgb_image: UInt8Image,
    output_size: int,
) -> UInt8Image:
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
    ):
        raise ValueError(
            "La imagen debe tener formato RGB."
        )

    if rgb_image.dtype != np.uint8:
        raise ValueError(
            "La imagen debe utilizar el tipo uint8."
        )

    original_height, original_width = (
        rgb_image.shape[:2]
    )

    geometry = calculate_letterbox_geometry(
        original_width=original_width,
        original_height=original_height,
        output_size=output_size,
    )

    image = Image.fromarray(
        rgb_image,
        mode="RGB",
    )

    resized_image = image.resize(
        (
            geometry.resized_width,
            geometry.resized_height,
        ),
        resample=BILINEAR_RESAMPLING,
    )

    canvas = Image.new(
        mode="RGB",
        size=(
            output_size,
            output_size,
        ),
        color=(
            0,
            0,
            0,
        ),
    )

    canvas.paste(
        resized_image,
        (
            geometry.left,
            geometry.top,
        ),
    )

    return np.ascontiguousarray(
        np.asarray(
            canvas,
            dtype=np.uint8,
        )
    )


def letterbox_mask(
    binary_mask: np.ndarray,
    output_size: int,
) -> UInt8Image:
    if binary_mask.ndim != 2:
        raise ValueError(
            "La máscara debe tener dos dimensiones."
        )

    original_height, original_width = (
        binary_mask.shape
    )

    geometry = calculate_letterbox_geometry(
        original_width=original_width,
        original_height=original_height,
        output_size=output_size,
    )

    normalized_mask = (
        binary_mask > 0
    ).astype(
        np.uint8
    )

    resized_mask = cv2.resize(
        normalized_mask,
        (
            geometry.resized_width,
            geometry.resized_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros(
        (
            output_size,
            output_size,
        ),
        dtype=np.uint8,
    )

    canvas[
        geometry.top:
        geometry.top + geometry.resized_height,

        geometry.left:
        geometry.left + geometry.resized_width,
    ] = resized_mask

    return np.ascontiguousarray(
        canvas
    )


def undo_letterbox_map(
    square_map: np.ndarray,
    original_width: int,
    original_height: int,
    canvas_size: int,
) -> np.ndarray:
    if square_map.ndim != 2:
        raise ValueError(
            "El mapa debe tener dos dimensiones."
        )

    if square_map.shape != (
        canvas_size,
        canvas_size,
    ):
        square_map = cv2.resize(
            square_map,
            (
                canvas_size,
                canvas_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    geometry = calculate_letterbox_geometry(
        original_width=original_width,
        original_height=original_height,
        output_size=canvas_size,
    )

    content_map = square_map[
        geometry.top:
        geometry.top + geometry.resized_height,

        geometry.left:
        geometry.left + geometry.resized_width,
    ]

    if content_map.size == 0:
        raise ValueError(
            "No se pudo revertir el letterbox."
        )

    restored_map = cv2.resize(
        content_map,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.ascontiguousarray(
        restored_map.astype(
            np.float32
        )
    )