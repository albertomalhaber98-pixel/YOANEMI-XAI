export class AnalysisApiError extends Error {
    constructor(message, status = 0) {
        super(message);

        this.name = "AnalysisApiError";
        this.status = status;
    }
}


export async function analyzeImage(
    file,
    signal,
) {
    const formData = new FormData();

    formData.append(
        "image",
        file,
    );

    let response;

    try {
        response = await fetch(
            "/api/analyze",
            {
                method: "POST",
                body: formData,
                signal,
                cache: "no-store",
                headers: {
                    Accept: "application/json",
                },
            },
        );
    } catch (error) {
        if (error.name === "AbortError") {
            throw error;
        }

        throw new AnalysisApiError(
            "No fue posible comunicarse con el servidor.",
        );
    }

    if (!response.ok) {
        const message = await extractErrorMessage(
            response,
        );

        throw new AnalysisApiError(
            message,
            response.status,
        );
    }

    try {
        return await response.json();
    } catch {
        throw new AnalysisApiError(
            "El servidor devolvió una respuesta inválida.",
            response.status,
        );
    }
}


async function extractErrorMessage(
    response,
) {
    const contentType = (
        response.headers.get(
            "content-type",
        )
        ?? ""
    );

    if (
        contentType.includes(
            "application/json",
        )
    ) {
        try {
            const payload = await response.json();

            if (
                typeof payload.detail === "string"
            ) {
                return payload.detail;
            }

            if (
                Array.isArray(
                    payload.detail,
                )
            ) {
                return (
                    "La solicitud contiene datos inválidos."
                );
            }
        } catch {
            // Continuar con un mensaje HTTP genérico.
        }
    }

    if (response.status === 413) {
        return "La imagen supera el tamaño permitido.";
    }

    if (response.status === 422) {
        return "La imagen recibida no pudo procesarse.";
    }

    if (response.status === 503) {
        return (
            "Los modelos todavía no están disponibles."
        );
    }

    return (
        "No se pudo completar el análisis. "
        + `Código HTTP: ${response.status}.`
    );
}
