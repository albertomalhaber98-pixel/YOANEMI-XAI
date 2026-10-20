import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.ml.gradcam.capture import (
    ActivationGradientCapture,
)
from app.ml.gradcam.exceptions import (
    GradCamPlusPlusError,
)
from app.ml.gradcam.types import (
    GradCamPlusPlusResult,
)
from app.ml.gradcam.visualization import (
    create_full_overlay,
    create_tissue_only_overlay,
    restrict_cam_to_tissue,
)
from app.ml.yolov9.model import (
    YoloV9Model,
)
from app.ml.yolov9.types import (
    YoloV9PreparedInput,
)


class GradCamPlusPlusExplainer:
    """
    Genera Grad-CAM++ sobre el clasificador YOLOv9-T.

    La capa objetivo es head.conv, antes del pooling global.
    """

    def __init__(
        self,
        yolo_model: YoloV9Model,
        overlay_alpha: float = 0.45,
    ) -> None:
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError(
                "overlay_alpha debe estar entre 0 y 1."
            )

        self._yolo_model = yolo_model
        self._overlay_alpha = overlay_alpha

    def explain(
        self,
        prepared_input: YoloV9PreparedInput,
        target_index: int | None = None,
    ) -> GradCamPlusPlusResult:
        if not self._yolo_model.is_loaded:
            raise GradCamPlusPlusError(
                "YOLOv9-T todavía no ha sido cargado."
            )

        raw_model = self._yolo_model.model
        class_names = self._yolo_model.class_names

        target_layer = self._resolve_target_layer(
            raw_model
        )

        for parameter in raw_model.parameters():
            parameter.requires_grad_(
                False
            )

        capture = ActivationGradientCapture(
            target_layer
        )

        try:
            raw_model.zero_grad(
                set_to_none=True
            )

            grad_input = (
                prepared_input.tensor
                .clone()
                .detach()
                .requires_grad_(
                    True
                )
            )

            with torch.enable_grad():
                output = raw_model(
                    grad_input
                )

                logits = self._extract_logits(
                    output
                )

                self._validate_logits(
                    logits=logits,
                    number_of_classes=len(
                        class_names
                    ),
                )

                probabilities_tensor = torch.softmax(
                    logits,
                    dim=1,
                )

                predicted_index = int(
                    probabilities_tensor
                    .argmax(
                        dim=1
                    )[0]
                    .item()
                )

                resolved_target_index = (
                    predicted_index
                    if target_index is None
                    else int(target_index)
                )

                if not (
                    0
                    <= resolved_target_index
                    < len(class_names)
                ):
                    raise GradCamPlusPlusError(
                        "La clase objetivo de Grad-CAM++ "
                        "no es válida."
                    )

                target_score = logits[
                    0,
                    resolved_target_index,
                ]

                target_score.backward()

            activations = capture.activations
            gradients = capture.gradients

            if activations is None:
                raise GradCamPlusPlusError(
                    "No se capturaron activaciones."
                )

            if gradients is None:
                raise GradCamPlusPlusError(
                    "No se capturaron gradientes."
                )

            if (
                activations.ndim != 4
                or gradients.ndim != 4
            ):
                raise GradCamPlusPlusError(
                    "Las activaciones y gradientes deben "
                    "tener formato BCHW."
                )

            raw_cam = self._calculate_gradcam_plus_plus(
                activations=activations,
                gradients=gradients,
                output_height=(
                    prepared_input
                    .model_image
                    .shape[0]
                ),
                output_width=(
                    prepared_input
                    .model_image
                    .shape[1]
                ),
            )

            tissue_cam = restrict_cam_to_tissue(
                cam=raw_cam,
                tissue_mask=(
                    prepared_input
                    .model_tissue_mask
                ),
            )

            (
                raw_heatmap,
                raw_overlay,
            ) = create_full_overlay(
                rgb_image=(
                    prepared_input
                    .model_image
                ),
                cam=raw_cam,
                alpha=self._overlay_alpha,
            )

            (
                tissue_heatmap,
                tissue_overlay,
            ) = create_tissue_only_overlay(
                rgb_image=(
                    prepared_input
                    .model_image
                ),
                masked_cam=tissue_cam,
                tissue_mask=(
                    prepared_input
                    .model_tissue_mask
                ),
                alpha=self._overlay_alpha,
            )

            probabilities_array = (
                probabilities_tensor[0]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    float
                )
            )

            return GradCamPlusPlusResult(
                raw_cam=raw_cam,
                tissue_cam=tissue_cam,
                raw_heatmap=raw_heatmap,
                raw_overlay=raw_overlay,
                tissue_heatmap=tissue_heatmap,
                tissue_overlay=tissue_overlay,
                predicted_index=predicted_index,
                predicted_class=class_names[
                    predicted_index
                ],
                confidence=float(
                    probabilities_array[
                        predicted_index
                    ]
                ),
                target_index=resolved_target_index,
                target_class=class_names[
                    resolved_target_index
                ],
                target_score=float(
                    target_score
                    .detach()
                    .cpu()
                    .item()
                ),
                probabilities=tuple(
                    float(value)
                    for value
                    in probabilities_array
                ),
                activation_shape=tuple(
                    int(value)
                    for value
                    in activations.shape
                ),
            )

        except GradCamPlusPlusError:
            raise

        except Exception as error:
            raise GradCamPlusPlusError(
                "No fue posible generar Grad-CAM++. "
                f"Causa: {type(error).__name__}: {error}"
            ) from error

        finally:
            capture.remove()

            raw_model.zero_grad(
                set_to_none=True
            )

    @staticmethod
    def _resolve_target_layer(
        raw_model: nn.Module,
    ) -> nn.Module:
        if not hasattr(
            raw_model,
            "model",
        ):
            raise GradCamPlusPlusError(
                "La arquitectura no contiene "
                "la secuencia model."
            )

        classification_head = (
            raw_model.model[-1]
        )

        if not hasattr(
            classification_head,
            "conv",
        ):
            raise GradCamPlusPlusError(
                "La cabeza Classify no contiene "
                "el atributo conv."
            )

        target_layer = (
            classification_head.conv
        )

        if not isinstance(
            target_layer,
            nn.Module,
        ):
            raise GradCamPlusPlusError(
                "La capa objetivo no es un módulo PyTorch."
            )

        return target_layer

    @staticmethod
    def _extract_logits(
        output: object,
    ) -> torch.Tensor:
        if isinstance(
            output,
            torch.Tensor,
        ):
            return output

        if isinstance(
            output,
            (
                tuple,
                list,
            ),
        ):
            if not output:
                raise GradCamPlusPlusError(
                    "YOLOv9-T devolvió una salida vacía."
                )

            first_output = output[0]

            if not isinstance(
                first_output,
                torch.Tensor,
            ):
                raise GradCamPlusPlusError(
                    "YOLOv9-T no devolvió logits válidos."
                )

            return first_output

        raise GradCamPlusPlusError(
            "Salida YOLOv9-T no compatible."
        )

    @staticmethod
    def _validate_logits(
        logits: torch.Tensor,
        number_of_classes: int,
    ) -> None:
        if logits.ndim != 2:
            raise GradCamPlusPlusError(
                "Los logits deben tener forma "
                "(batch, clases)."
            )

        if logits.shape[0] != 1:
            raise GradCamPlusPlusError(
                "Grad-CAM++ requiere un batch de tamaño 1."
            )

        if logits.shape[1] != number_of_classes:
            raise GradCamPlusPlusError(
                "La cantidad de logits no coincide "
                "con las clases."
            )

        if not torch.isfinite(
            logits
        ).all():
            raise GradCamPlusPlusError(
                "Los logits contienen NaN o infinito."
            )

    @staticmethod
    def _calculate_gradcam_plus_plus(
        activations: torch.Tensor,
        gradients: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> np.ndarray:
        gradient_squared = gradients.pow(
            2
        )

        gradient_cubed = gradients.pow(
            3
        )

        activation_sum = activations.sum(
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )

        denominator = (
            2.0
            * gradient_squared
            + activation_sum
            * gradient_cubed
        )

        epsilon = 1e-8

        denominator_safe = torch.where(
            denominator >= 0,
            denominator + epsilon,
            denominator - epsilon,
        )

        alphas = (
            gradient_squared
            / denominator_safe
        )

        alphas = torch.nan_to_num(
            alphas,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        positive_gradients = torch.relu(
            gradients
        )

        channel_weights = (
            alphas
            * positive_gradients
        ).sum(
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )

        cam = (
            channel_weights
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                output_height,
                output_width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
        ]

        cam_min = cam.min()
        cam_max = cam.max()

        if (
            cam_max
            - cam_min
        ).abs().item() < 1e-12:
            raise GradCamPlusPlusError(
                "Grad-CAM++ produjo un mapa sin variación."
            )

        normalized_cam = (
            cam
            - cam_min
        ) / (
            cam_max
            - cam_min
            + 1e-12
        )

        return np.ascontiguousarray(
            normalized_cam
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )
