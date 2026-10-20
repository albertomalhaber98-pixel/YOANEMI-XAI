from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


UInt8Image = NDArray[
    np.uint8
]

FloatImage = NDArray[
    np.float32
]

BoundingBox = tuple[
    int,
    int,
    int,
    int,
]


@dataclass(
    slots=True,
    frozen=True,
)
class SegmentedRegion:
    """
    Resultado de aplicar la máscara de U-Net
    sobre la imagen original.
    """

    bounding_box: BoundingBox

    original_crop: UInt8Image

    mask_crop: UInt8Image

    segmented_crop: UInt8Image