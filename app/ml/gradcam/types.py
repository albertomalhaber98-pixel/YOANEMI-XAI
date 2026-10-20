from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.ml.preprocessing.types import UInt8Image


Float32Map = NDArray[
    np.float32
]


@dataclass(slots=True, frozen=True)
class GradCamPlusPlusResult:
    raw_cam: Float32Map
    tissue_cam: Float32Map

    raw_heatmap: UInt8Image
    raw_overlay: UInt8Image

    tissue_heatmap: UInt8Image
    tissue_overlay: UInt8Image

    predicted_index: int
    predicted_class: str
    confidence: float

    target_index: int
    target_class: str
    target_score: float

    probabilities: tuple[float, ...]
    activation_shape: tuple[int, ...]
