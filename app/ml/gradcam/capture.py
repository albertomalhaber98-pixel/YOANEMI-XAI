import torch
from torch import nn

from app.ml.gradcam.exceptions import (
    GradCamPlusPlusError,
)


class ActivationGradientCapture:
    """
    Captura la salida espacial de la capa objetivo
    y su gradiente durante backpropagation.
    """

    def __init__(
        self,
        target_layer: nn.Module,
    ) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module: nn.Module,
        module_input: tuple[object, ...],
        module_output: object,
    ) -> None:
        if not torch.is_tensor(
            module_output
        ):
            raise GradCamPlusPlusError(
                "La capa objetivo no devolvió un tensor."
            )

        self.activations = module_output

        if module_output.requires_grad:
            module_output.register_hook(
                self._gradient_hook
            )

    def _gradient_hook(
        self,
        gradient: torch.Tensor,
    ) -> None:
        self.gradients = gradient

    def remove(
        self,
    ) -> None:
        self._forward_handle.remove()
