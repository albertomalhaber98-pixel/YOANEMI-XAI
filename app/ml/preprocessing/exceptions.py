class ImageProcessingError(
    RuntimeError
):
    """Error general durante el procesamiento de una imagen."""


class ImageDecodeError(
    ImageProcessingError
):
    """La imagen recibida no pudo decodificarse correctamente."""


class InvalidMaskError(
    ImageProcessingError
):
    """La máscara de segmentación no contiene una región válida."""