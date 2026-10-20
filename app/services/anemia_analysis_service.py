import asyncio
from time import perf_counter

from starlette.concurrency import (
    run_in_threadpool,
)

from app.core.config import Settings
from app.ml.gradcam.exceptions import (
    GradCamPlusPlusError,
)
from app.ml.preprocessing import (
    decode_image_bytes,
    extract_segmented_region,
)
from app.ml.preprocessing.exceptions import (
    ImageDecodeError,
    InvalidMaskError,
)
from app.ml.registry import ModelRegistry
from app.ml.unet.exceptions import (
    UnetInferenceError,
)
from app.ml.yolov9.exceptions import (
    YoloV9InferenceError,
)
from app.services.exceptions import (
    AnalysisExecutionError,
    InvalidAnalysisInputError,
)
from app.services.types import (
    AnalysisImages,
    AnalysisProbability,
    AnemiaAnalysisResult,
)


class AnemiaAnalysisService:
    """
    Coordina el pipeline experimental completo.

    Responsabilidades:
        - validar el tamaño del archivo;
        - ejecutar U-Net;
        - extraer la región segmentada;
        - ejecutar YOLOv9-T;
        - generar Grad-CAM++;
        - devolver un resultado estructurado.

    No guarda archivos ni conoce detalles HTTP.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        settings: Settings,
    ) -> None:
        self._registry = registry

        self._maximum_upload_bytes = (
            settings.max_upload_mb
            * 1024
            * 1024
        )

        self._roi_margin_ratio = (
            settings.unet_roi_margin_ratio
        )

        # Grad-CAM++ utiliza hooks sobre el modelo compartido.
        # Por eso inicialmente se ejecutará un análisis a la vez.
        self._analysis_lock = asyncio.Lock()

    async def analyze(
        self,
        image_bytes: bytes,
    ) -> AnemiaAnalysisResult:
        """
        Ejecuta el análisis sin bloquear el event loop de FastAPI.

        El procesamiento pesado se envía a un threadpool,
        pero el lock impide inferencias simultáneas sobre
        el mismo modelo y sus hooks.
        """

        self._validate_uploaded_bytes(
            image_bytes
        )

        async with self._analysis_lock:
            return await run_in_threadpool(
                self._analyze_sync,
                image_bytes,
            )

    def _analyze_sync(
        self,
        image_bytes: bytes,
    ) -> AnemiaAnalysisResult:
        if not self._registry.is_ready:
            raise AnalysisExecutionError(
                "Los modelos todavía no están disponibles."
            )

        started_at = perf_counter()

        try:
            original_rgb = decode_image_bytes(
                image_bytes
            )

            segmentation = (
                self._registry
                .unet_segmenter
                .segment(
                    original_rgb
                )
            )

            segmented_region = (
                extract_segmented_region(
                    original_rgb=original_rgb,
                    binary_mask=(
                        segmentation.binary_mask
                    ),
                    margin_ratio=(
                        self._roi_margin_ratio
                    ),
                )
            )

            prepared_input = (
                self._registry
                .yolov9_input_preparer
                .prepare(
                    segmented_rgb=(
                        segmented_region
                        .segmented_crop
                    ),
                    tissue_mask=(
                        segmented_region
                        .mask_crop
                    ),
                )
            )

            classification = (
                self._registry
                .yolov9_classifier
                .classify(
                    prepared_input
                )
            )

            gradcam = (
                self._registry
                .gradcam_explainer
                .explain(
                    prepared_input=(
                        prepared_input
                    ),
                    target_index=(
                        classification
                        .predicted_index
                    ),
                )
            )

            self._validate_consistency(
                classification_index=(
                    classification
                    .predicted_index
                ),
                gradcam_index=(
                    gradcam
                    .predicted_index
                ),
            )

        except (
            ImageDecodeError,
            InvalidMaskError,
            UnetInferenceError,
            ValueError,
        ) as error:
            raise InvalidAnalysisInputError(
                str(error)
            ) from error

        except (
            YoloV9InferenceError,
            GradCamPlusPlusError,
        ) as error:
            raise AnalysisExecutionError(
                "No se pudo completar la inferencia. "
                f"Causa: {error}"
            ) from error

        except AnalysisExecutionError:
            raise

        except Exception as error:
            raise AnalysisExecutionError(
                "Ocurrió un error inesperado durante "
                "el análisis experimental. "
                f"Causa: {type(error).__name__}: {error}"
            ) from error

        processing_time_ms = (
            perf_counter()
            - started_at
        ) * 1000.0

        probabilities = tuple(
            AnalysisProbability(
                class_index=(
                    probability
                    .class_index
                ),
                class_name=(
                    probability
                    .class_name
                ),
                probability=(
                    probability
                    .probability
                ),
            )
            for probability
            in classification.probabilities
        )

        images = AnalysisImages(
            original=original_rgb,

            unet_probability=(
                segmentation
                .probability_map
            ),

            unet_mask=(
                segmentation
                .binary_mask
            ),

            original_crop=(
                segmented_region
                .original_crop
            ),

            segmented_crop=(
                segmented_region
                .segmented_crop
            ),

            datasetp_input=(
                prepared_input
                .datasetp_image
            ),

            yolov9_input=(
                prepared_input
                .model_image
            ),

            gradcam_raw_heatmap=(
                gradcam
                .raw_heatmap
            ),

            gradcam_raw_overlay=(
                gradcam
                .raw_overlay
            ),

            gradcam_tissue_heatmap=(
                gradcam
                .tissue_heatmap
            ),

            gradcam_tissue_overlay=(
                gradcam
                .tissue_overlay
            ),
        )

        return AnemiaAnalysisResult(
            predicted_index=(
                classification
                .predicted_index
            ),

            predicted_class=(
                classification
                .predicted_class
            ),

            confidence=(
                classification
                .confidence
            ),

            probabilities=probabilities,

            segmented_percentage=(
                segmentation
                .segmented_percentage
            ),

            bounding_box=(
                segmented_region
                .bounding_box
            ),

            gradcam_target_index=(
                gradcam
                .target_index
            ),

            gradcam_target_class=(
                gradcam
                .target_class
            ),

            gradcam_target_score=(
                gradcam
                .target_score
            ),

            gradcam_activation_shape=(
                gradcam
                .activation_shape
            ),

            processing_time_ms=(
                processing_time_ms
            ),

            images=images,
        )

    def _validate_uploaded_bytes(
        self,
        image_bytes: bytes,
    ) -> None:
        if not isinstance(
            image_bytes,
            bytes,
        ):
            raise InvalidAnalysisInputError(
                "El contenido recibido debe ser binario."
            )

        if not image_bytes:
            raise InvalidAnalysisInputError(
                "No se recibió ninguna imagen."
            )

        image_size = len(
            image_bytes
        )

        if image_size > self._maximum_upload_bytes:
            maximum_megabytes = (
                self._maximum_upload_bytes
                / 1024
                / 1024
            )

            raise InvalidAnalysisInputError(
                "La imagen supera el tamaño máximo "
                f"permitido de {maximum_megabytes:.0f} MB."
            )

    @staticmethod
    def _validate_consistency(
        classification_index: int,
        gradcam_index: int,
    ) -> None:
        """
        Grad-CAM++ vuelve a ejecutar el mismo modelo con la
        misma entrada. La predicción debe permanecer estable.
        """

        if (
            classification_index
            != gradcam_index
        ):
            raise AnalysisExecutionError(
                "La clasificación y Grad-CAM++ "
                "produjeron clases diferentes."
            )
