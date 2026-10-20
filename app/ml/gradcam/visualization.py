import cv2
import numpy as np

from app.ml.gradcam.exceptions import (
    GradCamPlusPlusError,
)
from app.ml.preprocessing.types import UInt8Image


def restrict_cam_to_tissue(
    cam: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    if cam.shape != tissue_mask.shape:
        raise GradCamPlusPlusError(
            "El CAM y la máscara deben tener "
            "las mismas dimensiones."
        )

    tissue = tissue_mask > 0

    masked_cam = np.zeros_like(
        cam,
        dtype=np.float32,
    )

    masked_cam[tissue] = cam[tissue]

    maximum = float(
        masked_cam.max()
    )

    if maximum > 1e-12:
        masked_cam /= maximum

    return np.ascontiguousarray(
        masked_cam
    )


def cam_to_heatmap(
    cam: np.ndarray,
) -> UInt8Image:
    cam_uint8 = np.clip(
        cam * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    heatmap_bgr = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return np.ascontiguousarray(
        heatmap_rgb
    )


def create_full_overlay(
    rgb_image: UInt8Image,
    cam: np.ndarray,
    alpha: float,
) -> tuple[
    UInt8Image,
    UInt8Image,
]:
    _validate_overlay_inputs(
        rgb_image=rgb_image,
        cam=cam,
        alpha=alpha,
    )

    heatmap_rgb = cam_to_heatmap(
        cam
    )

    overlay_rgb = cv2.addWeighted(
        rgb_image,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0,
    )

    return (
        heatmap_rgb,
        np.ascontiguousarray(
            overlay_rgb
        ),
    )


def create_tissue_only_overlay(
    rgb_image: UInt8Image,
    masked_cam: np.ndarray,
    tissue_mask: np.ndarray,
    alpha: float,
) -> tuple[
    UInt8Image,
    UInt8Image,
]:
    _validate_overlay_inputs(
        rgb_image=rgb_image,
        cam=masked_cam,
        alpha=alpha,
    )

    if tissue_mask.shape != masked_cam.shape:
        raise GradCamPlusPlusError(
            "La máscara de tejido y el CAM "
            "tienen dimensiones diferentes."
        )

    tissue = tissue_mask > 0

    heatmap_rgb = cam_to_heatmap(
        masked_cam
    )

    tissue_heatmap = np.zeros_like(
        heatmap_rgb,
        dtype=np.uint8,
    )

    tissue_heatmap[tissue] = (
        heatmap_rgb[tissue]
    )

    blended_rgb = cv2.addWeighted(
        rgb_image,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0,
    )

    overlay_rgb = rgb_image.copy()

    overlay_rgb[tissue] = (
        blended_rgb[tissue]
    )

    return (
        np.ascontiguousarray(
            tissue_heatmap
        ),
        np.ascontiguousarray(
            overlay_rgb
        ),
    )


def _validate_overlay_inputs(
    rgb_image: np.ndarray,
    cam: np.ndarray,
    alpha: float,
) -> None:
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
    ):
        raise GradCamPlusPlusError(
            "La imagen del overlay debe ser RGB."
        )

    if cam.ndim != 2:
        raise GradCamPlusPlusError(
            "El mapa Grad-CAM++ debe tener dos dimensiones."
        )

    if rgb_image.shape[:2] != cam.shape:
        raise GradCamPlusPlusError(
            "La imagen y Grad-CAM++ tienen "
            "dimensiones diferentes."
        )

    if not 0.0 <= alpha <= 1.0:
        raise GradCamPlusPlusError(
            "El valor alpha debe estar entre 0 y 1."
        )
