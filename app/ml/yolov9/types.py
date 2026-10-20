from dataclasses import dataclass

import torch

from app.ml.preprocessing.types import UInt8Image


@dataclass(slots=True, frozen=True)
class YoloV9PreparedInput:
    datasetp_image: UInt8Image
    datasetp_tissue_mask: UInt8Image

    model_image: UInt8Image
    model_tissue_mask: UInt8Image

    tensor: torch.Tensor


@dataclass(slots=True, frozen=True)
class ClassProbability:
    class_index: int
    class_name: str
    probability: float


@dataclass(slots=True, frozen=True)
class YoloV9ClassificationResult:
    predicted_index: int
    predicted_class: str
    confidence: float
    probabilities: tuple[
        ClassProbability,
        ...,
    ]