import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    Response,
    status,
)

from app.api.mappers.analysis_mapper import (
    map_analysis_response,
)
from app.schemas.analysis import (
    AnalysisResponse,
)
from app.services import (
    AnalysisExecutionError,
    AnemiaAnalysisService,
    InvalidAnalysisInputError,
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    tags=["Analysis"],
)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analizar una imagen ocular",
    description=(
        "Segmenta la conjuntiva mediante U-Net, "
        "clasifica mediante YOLOv9-T y genera "
        "una explicación Grad-CAM++."
    ),
)
async def analyze_image(
    request: Request,
    response: Response,
    image: Annotated[
        bytes,
        File(
            description=(
                "Imagen ocular en formato JPEG, PNG o WebP."
            )
        ),
    ],
) -> AnalysisResponse:
    analysis_service: (
        AnemiaAnalysisService
        | None
    ) = getattr(
        request.app.state,
        "analysis_service",
        None,
    )

    if analysis_service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "El servicio de análisis todavía "
                "no está disponible."
            ),
        )

    try:
        result = await analysis_service.analyze(
            image
        )

        mapped_response = map_analysis_response(
            result
        )

    except InvalidAnalysisInputError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(
                error
            ),
        ) from error

    except AnalysisExecutionError as error:
        # No registrar bytes, Base64, nombre del archivo
        # ni predicción del usuario.
        logger.exception(
            "El pipeline de análisis no pudo completarse."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "No se pudo completar el análisis "
                "experimental."
            ),
        ) from error

    except Exception as error:
        logger.exception(
            "Error inesperado en el endpoint de análisis."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Ocurrió un error interno durante "
                "el procesamiento."
            ),
        ) from error

    response.headers[
        "X-Processing-Time-Ms"
    ] = str(
        mapped_response.processing_time_ms
    )

    return mapped_response
