import torch


def resolve_device() -> torch.device:
    """
    Selecciona CUDA cuando está disponible.
    En caso contrario utiliza CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def get_device_description(
    device: torch.device,
) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(
            device.index or 0
        )

    return "CPU"