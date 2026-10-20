from pydantic import (
    BaseModel,
    Field,
)


class BoundingBoxResponse(BaseModel):
    x1: int = Field(
        ge=0
    )

    y1: int = Field(
        ge=0
    )

    x2: int = Field(
        ge=0
    )

    y2: int = Field(
        ge=0
    )


class ClassProbabilityResponse(BaseModel):
    class_index: int = Field(
        ge=0
    )

    model_class: str
    display_label: str

    probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    percentage: float = Field(
        ge=0.0,
        le=100.0,
    )


class SegmentationResponse(BaseModel):
    area_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    bounding_box: BoundingBoxResponse


class GradCamResponse(BaseModel):
    target_index: int = Field(
        ge=0
    )

    target_class: str
    target_label: str
    target_score: float

    activation_shape: list[int]


class AnalysisImagesResponse(BaseModel):
    """
    Imágenes derivadas codificadas como Data URI.

    No se devuelve nuevamente la imagen original porque
    el navegador ya dispone de ella localmente.
    """

    unet_probability: str
    unet_mask: str

    original_crop: str
    segmented_crop: str

    datasetp_input: str
    yolov9_input: str

    gradcam_raw_heatmap: str
    gradcam_raw_overlay: str

    gradcam_tissue_heatmap: str
    gradcam_tissue_overlay: str


class AnalysisResponse(BaseModel):
    predicted_index: int = Field(
        ge=0
    )

    predicted_class: str
    predicted_label: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    probabilities: list[
        ClassProbabilityResponse
    ]

    segmentation: SegmentationResponse
    gradcam: GradCamResponse

    processing_time_ms: float = Field(
        ge=0.0
    )

    images: AnalysisImagesResponse

    disclaimer: str
