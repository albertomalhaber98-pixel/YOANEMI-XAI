from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.ml.preprocessing.types import (
    BoundingBox,
    UInt8Image,
)


Float32Map = NDArray[
    np.float32
]


@dataclass(
    slots=True,
    frozen=True,
)
class AnalysisProbability:
    class_index: int
    class_name: str
    probability: float


@dataclass(
    slots=True,
    frozen=True,
)
class AnalysisImages:
    """
    Imágenes generadas exclusivamente en memoria.
    """

    original: UInt8Image

    unet_probability: Float32Map
    unet_mask: UInt8Image

    original_crop: UInt8Image
    segmented_crop: UInt8Image

    datasetp_input: UInt8Image
    yolov9_input: UInt8Image

    gradcam_raw_heatmap: UInt8Image
    gradcam_raw_overlay: UInt8Image

    gradcam_tissue_heatmap: UInt8Image
    gradcam_tissue_overlay: UInt8Image


@dataclass(
    slots=True,
    frozen=True,
)
class AnemiaAnalysisResult:
    predicted_index: int
    predicted_class: str
    confidence: float

    probabilities: tuple[
        AnalysisProbability,
        ...,
    ]

    segmented_percentage: float
    bounding_box: BoundingBox

    gradcam_target_index: int
    gradcam_target_class: str
    gradcam_target_score: float
    gradcam_activation_shape: tuple[int, ...]

    processing_time_ms: float

    images: AnalysisImages
