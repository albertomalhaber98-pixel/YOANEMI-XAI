from io import BytesIO

import numpy as np
from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)

from app.ml.preprocessing.exceptions import (
    ImageDecodeError,
)
from app.ml.preprocessing.types import (
    UInt8Image,
)


DEFAULT_MAX_PIXELS = 20_000_000


def decode_image_bytes(
    image_bytes: bytes,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> UInt8Image:
    """
    Decodifica JPEG, PNG o WebP y devuelve una imagen RGB uint8.

    La imagen se mantiene únicamente en memoria.

    Las transparencias se componen sobre fondo negro para mantener
    consistencia con el procesamiento utilizado en el entrenamiento.
    """

    if not image_bytes:
        raise ImageDecodeError(
            "La imagen recibida está vacía."
        )

    try:
        with Image.open(
            BytesIO(
                image_bytes
            )
        ) as source_image:

            source_image = (
                ImageOps.exif_transpose(
                    source_image
                )
            )

            width, height = (
                source_image.size
            )

            if width <= 0 or height <= 0:
                raise ImageDecodeError(
                    "La imagen tiene dimensiones inválidas."
                )

            total_pixels = (
                width
                * height
            )

            if total_pixels > max_pixels:
                raise ImageDecodeError(
                    "La imagen supera la cantidad máxima "
                    f"permitida de píxeles: {max_pixels:,}."
                )

            has_transparency = (
                "A"
                in source_image.getbands()
                or (
                    source_image.mode == "P"
                    and "transparency"
                    in source_image.info
                )
            )

            if has_transparency:
                rgba_image = (
                    source_image.convert(
                        "RGBA"
                    )
                )

                black_background = Image.new(
                    mode="RGBA",
                    size=rgba_image.size,
                    color=(
                        0,
                        0,
                        0,
                        255,
                    ),
                )

                rgb_image = (
                    Image.alpha_composite(
                        black_background,
                        rgba_image,
                    )
                    .convert(
                        "RGB"
                    )
                )

            else:
                rgb_image = (
                    source_image.convert(
                        "RGB"
                    )
                )

            decoded_image = np.asarray(
                rgb_image,
                dtype=np.uint8,
            )

    except ImageDecodeError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ImageDecodeError(
            "El archivo recibido no contiene "
            "una imagen válida."
        ) from error

    if (
        decoded_image.ndim != 3
        or decoded_image.shape[2] != 3
    ):
        raise ImageDecodeError(
            "La imagen decodificada no tiene "
            "el formato RGB esperado."
        )

    return np.ascontiguousarray(
        decoded_image
    )