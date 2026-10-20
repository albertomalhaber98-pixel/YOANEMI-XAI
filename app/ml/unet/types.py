from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.ml.preprocessing.types import (
    UInt8Image,
)


Float32Image = NDArray[
    np.float32
]


@dataclass(
    slots=True,
    frozen=True,
)
class UnetSegmentationResult:
    """
    Resultado de la segmentación de conjuntiva.

    probability_map:
        Probabilidad por píxel, con valores entre 0 y 1.

    binary_mask:
        Máscara binaria uint8 con valores 0 y 1.

    segmented_percentage:
        Porcentaje de píxeles de la imagen pertenecientes
        a la región segmentada.
    """

    probability_map: Float32Image
    binary_mask: UInt8Image
    segmented_percentage: float
