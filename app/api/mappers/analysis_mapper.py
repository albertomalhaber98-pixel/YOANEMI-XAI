from app.ml.preprocessing import (
    encode_png_data_uri,
)
from app.schemas.analysis import (
    AnalysisImagesResponse,
    AnalysisResponse,
    BoundingBoxResponse,
    ClassProbabilityResponse,
    GradCamResponse,
    SegmentationResponse,
)
from app.services.types import (
    AnemiaAnalysisResult,
)


CLASS_LABELS = {
    "Anemic": "Anemia",
    "Non-anemic": "Sin anemia",
}


def translate_class_name(
    model_class: str,
) -> str:
    return CLASS_LABELS.get(
        model_class,
        model_class,
    )


def map_analysis_response(
    result: AnemiaAnalysisResult,
) -> AnalysisResponse:
    (
        x1,
        y1,
        x2,
        y2,
    ) = result.bounding_box

    probabilities = [
        ClassProbabilityResponse(
            class_index=(
                item.class_index
            ),
            model_class=(
                item.class_name
            ),
            display_label=(
                translate_class_name(
                    item.class_name
                )
            ),
            probability=(
                item.probability
            ),
            percentage=(
                item.probability
                * 100.0
            ),
        )
        for item
        in result.probabilities
    ]

    images = AnalysisImagesResponse(
        unet_probability=(
            encode_png_data_uri(
                result
                .images
                .unet_probability
            )
        ),

        unet_mask=(
            encode_png_data_uri(
                result
                .images
                .unet_mask
            )
        ),

        original_crop=(
            encode_png_data_uri(
                result
                .images
                .original_crop
            )
        ),

        segmented_crop=(
            encode_png_data_uri(
                result
                .images
                .segmented_crop
            )
        ),

        datasetp_input=(
            encode_png_data_uri(
                result
                .images
                .datasetp_input
            )
        ),

        yolov9_input=(
            encode_png_data_uri(
                result
                .images
                .yolov9_input
            )
        ),

        gradcam_raw_heatmap=(
            encode_png_data_uri(
                result
                .images
                .gradcam_raw_heatmap
            )
        ),

        gradcam_raw_overlay=(
            encode_png_data_uri(
                result
                .images
                .gradcam_raw_overlay
            )
        ),

        gradcam_tissue_heatmap=(
            encode_png_data_uri(
                result
                .images
                .gradcam_tissue_heatmap
            )
        ),

        gradcam_tissue_overlay=(
            encode_png_data_uri(
                result
                .images
                .gradcam_tissue_overlay
            )
        ),
    )

    return AnalysisResponse(
        predicted_index=(
            result.predicted_index
        ),

        predicted_class=(
            result.predicted_class
        ),

        predicted_label=(
            translate_class_name(
                result.predicted_class
            )
        ),

        confidence=(
            result.confidence
        ),

        confidence_percentage=(
            result.confidence
            * 100.0
        ),

        probabilities=probabilities,

        segmentation=SegmentationResponse(
            area_percentage=(
                result
                .segmented_percentage
            ),
            bounding_box=BoundingBoxResponse(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            ),
        ),

        gradcam=GradCamResponse(
            target_index=(
                result
                .gradcam_target_index
            ),
            target_class=(
                result
                .gradcam_target_class
            ),
            target_label=(
                translate_class_name(
                    result
                    .gradcam_target_class
                )
            ),
            target_score=(
                result
                .gradcam_target_score
            ),
            activation_shape=list(
                result
                .gradcam_activation_shape
            ),
        ),

        processing_time_ms=round(
            result.processing_time_ms,
            2,
        ),

        images=images,

        disclaimer=(
            "Resultado experimental generado por un modelo "
            "de inteligencia artificial. No constituye un "
            "diagnóstico médico ni reemplaza una evaluación "
            "clínica o una prueba de hemoglobina."
        ),
    )
