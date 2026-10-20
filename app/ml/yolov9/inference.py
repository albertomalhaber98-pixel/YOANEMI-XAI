import torch

from app.ml.yolov9.exceptions import YoloV9InferenceError
from app.ml.yolov9.model import YoloV9Model
from app.ml.yolov9.types import (
    ClassProbability,
    YoloV9ClassificationResult,
    YoloV9PreparedInput,
)


class YoloV9Classifier:
    """
    Ejecuta únicamente la clasificación.

    No carga checkpoints, no transforma imágenes
    y no guarda resultados en disco.
    """

    def __init__(
        self,
        yolo_model: YoloV9Model,
    ) -> None:
        self._yolo_model = yolo_model

    def classify(
        self,
        prepared_input: YoloV9PreparedInput,
    ) -> YoloV9ClassificationResult:
        if not self._yolo_model.is_loaded:
            raise YoloV9InferenceError(
                "No se puede clasificar porque YOLOv9-T "
                "todavía no ha sido cargado."
            )

        tensor = prepared_input.tensor
        class_names = self._yolo_model.class_names

        self._validate_input_tensor(tensor)

        try:
            with torch.inference_mode():
                output = self._yolo_model.model(tensor)

        except Exception as error:
            raise YoloV9InferenceError(
                "YOLOv9-T no pudo procesar la imagen."
            ) from error

        logits = self._extract_logits(output)

        self._validate_logits(
            logits=logits,
            number_of_classes=len(class_names),
        )

        probabilities_tensor = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = int(
            probabilities_tensor.argmax(dim=1)[0].item()
        )

        probabilities_array = (
            probabilities_tensor[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        probabilities = tuple(
            ClassProbability(
                class_index=index,
                class_name=class_name,
                probability=float(probabilities_array[index]),
            )
            for index, class_name in enumerate(class_names)
        )

        return YoloV9ClassificationResult(
            predicted_index=predicted_index,
            predicted_class=class_names[predicted_index],
            confidence=float(
                probabilities_array[predicted_index]
            ),
            probabilities=probabilities,
        )

    @staticmethod
    def _extract_logits(
        output: object,
    ) -> torch.Tensor:
        """
        El pipeline validado toma el primer tensor cuando
        el modelo devuelve una tupla o lista.
        """

        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (tuple, list)):
            if not output:
                raise YoloV9InferenceError(
                    "YOLOv9-T devolvió una salida vacía."
                )

            first_output = output[0]

            if not isinstance(first_output, torch.Tensor):
                raise YoloV9InferenceError(
                    "El primer resultado de YOLOv9-T "
                    "no es un tensor."
                )

            return first_output

        raise YoloV9InferenceError(
            "YOLOv9-T devolvió un tipo de salida no compatible: "
            f"{type(output).__name__}."
        )

    @staticmethod
    def _validate_input_tensor(
        tensor: torch.Tensor,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise YoloV9InferenceError(
                "La entrada preparada no contiene un tensor."
            )

        if tensor.ndim != 4:
            raise YoloV9InferenceError(
                "El tensor debe tener cuatro dimensiones BCHW."
            )

        if tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise YoloV9InferenceError(
                "La forma esperada es (1, 3, alto, ancho), "
                f"pero se recibió {tuple(tensor.shape)}."
            )

    @staticmethod
    def _validate_logits(
        logits: torch.Tensor,
        number_of_classes: int,
    ) -> None:
        if logits.ndim != 2:
            raise YoloV9InferenceError(
                "La salida de YOLOv9-T tiene una forma inesperada: "
                f"{tuple(logits.shape)}."
            )

        if logits.shape[0] != 1:
            raise YoloV9InferenceError(
                "La inferencia individual esperaba "
                "un batch de tamaño 1."
            )

        if logits.shape[1] != number_of_classes:
            raise YoloV9InferenceError(
                "La cantidad de salidas del modelo no coincide "
                "con las clases del checkpoint. "
                f"Salidas: {logits.shape[1]}; "
                f"clases: {number_of_classes}."
            )

        if not torch.isfinite(logits).all():
            raise YoloV9InferenceError(
                "YOLOv9-T devolvió valores NaN o infinitos."
            )
