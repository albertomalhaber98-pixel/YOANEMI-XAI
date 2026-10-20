class AnemiaAnalysisError(RuntimeError):
    """Error general durante el análisis experimental."""


class InvalidAnalysisInputError(AnemiaAnalysisError):
    """
    La imagen recibida no es válida o no contiene
    una región segmentable.
    """


class AnalysisExecutionError(AnemiaAnalysisError):
    """
    El pipeline no pudo completar la inferencia
    debido a un error interno.
    """
