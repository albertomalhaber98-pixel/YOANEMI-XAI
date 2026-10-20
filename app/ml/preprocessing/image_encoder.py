import base64
from io import BytesIO

import numpy as np
from PIL import Image


def normalize_image_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convierte una imagen binaria, probabilística o RGB
    al formato uint8 requerido para su codificación.
    """

    image_array = np.asarray(
        image
    )

    if image_array.size == 0:
        raise ValueError(
            "No se puede codificar una imagen vacía."
        )

    if not np.isfinite(
        image_array
    ).all():
        raise ValueError(
            "La imagen contiene valores NaN o infinitos."
        )

    if image_array.dtype == np.bool_:
        return np.ascontiguousarray(
            image_array.astype(
                np.uint8
            )
            * 255
        )

    if image_array.dtype == np.uint8:
        maximum = int(
            image_array.max()
        )

        if maximum <= 1:
            return np.ascontiguousarray(
                image_array
                * 255
            )

        return np.ascontiguousarray(
            image_array
        )

    float_image = image_array.astype(
        np.float32
    )

    minimum = float(
        float_image.min()
    )

    maximum = float(
        float_image.max()
    )

    if (
        minimum >= 0.0
        and maximum <= 1.0
    ):
        float_image = (
            float_image
            * 255.0
        )

    normalized = np.clip(
        float_image,
        0,
        255,
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        normalized
    )


def encode_png_bytes(
    image: np.ndarray,
) -> bytes:
    """
    Codifica una matriz NumPy como PNG dentro de memoria.
    """

    normalized_image = normalize_image_to_uint8(
        image
    )

    if normalized_image.ndim == 2:
        pil_image = Image.fromarray(
            normalized_image,
            mode="L",
        )

    elif (
        normalized_image.ndim == 3
        and normalized_image.shape[2] == 3
    ):
        pil_image = Image.fromarray(
            normalized_image,
            mode="RGB",
        )

    elif (
        normalized_image.ndim == 3
        and normalized_image.shape[2] == 4
    ):
        pil_image = Image.fromarray(
            normalized_image,
            mode="RGBA",
        )

    else:
        raise ValueError(
            "La imagen debe ser grayscale, RGB o RGBA."
        )

    output_buffer = BytesIO()

    pil_image.save(
        output_buffer,
        format="PNG",
        optimize=True,
    )

    return output_buffer.getvalue()


def encode_png_base64(
    image: np.ndarray,
) -> str:
    """
    Devuelve únicamente el contenido Base64 del PNG.
    """

    png_bytes = encode_png_bytes(
        image
    )

    return base64.b64encode(
        png_bytes
    ).decode(
        "ascii"
    )


def encode_png_data_uri(
    image: np.ndarray,
) -> str:
    """
    Devuelve una URI que puede asignarse directamente
    al atributo src de una etiqueta img.
    """

    encoded_image = encode_png_base64(
        image
    )

    return (
        "data:image/png;base64,"
        f"{encoded_image}"
    )