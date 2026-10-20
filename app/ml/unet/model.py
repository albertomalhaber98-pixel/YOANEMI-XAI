from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch import nn


class UnetModelLoadError(
    RuntimeError
):
    pass


class UnetModel:
    """
    Responsable exclusivamente de construir y cargar U-Net.

    La segmentación se implementará después en otro servicio.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model: nn.Module | None = None

    @property
    def model(
        self,
    ) -> nn.Module:
        if self._model is None:
            raise UnetModelLoadError(
                "U-Net todavía no ha sido cargada."
            )

        return self._model

    @property
    def is_loaded(
        self,
    ) -> bool:
        return self._model is not None

    def load(
        self,
    ) -> None:
        if not self._checkpoint_path.is_file():
            raise UnetModelLoadError(
                "No se encontró el checkpoint de U-Net: "
                f"{self._checkpoint_path}"
            )

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        checkpoint = torch.load(
            self._checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = self._extract_state_dict(
            checkpoint
        )

        normalized_state_dict = {
            key.replace(
                "module.",
                "",
            ): value
            for key, value
            in state_dict.items()
        }

        try:
            model.load_state_dict(
                normalized_state_dict,
                strict=True,
            )

        except RuntimeError as error:
            raise UnetModelLoadError(
                "El checkpoint no coincide con la "
                "arquitectura U-Net ResNet34."
            ) from error

        model = (
            model
            .float()
            .to(
                self._device
            )
        )

        model.eval()

        self._model = model

    def unload(
        self,
    ) -> None:
        self._model = None

    @staticmethod
    def _extract_state_dict(
        checkpoint: object,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(
            checkpoint,
            dict,
        ):
            raise UnetModelLoadError(
                "El checkpoint de U-Net no es "
                "un diccionario válido."
            )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

        if not isinstance(
            state_dict,
            dict,
        ):
            raise UnetModelLoadError(
                "No se encontró un state_dict "
                "válido para U-Net."
            )

        return state_dict