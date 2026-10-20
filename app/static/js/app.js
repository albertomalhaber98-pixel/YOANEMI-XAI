import {
    AnalysisApiError,
    analyzeImage,
} from "./api-client.js";

import {
    clearInterface,
    getUiElements,
    renderAnalysis,
    setLoading,
    showSelectedFile,
    showStatus,
} from "./ui.js";


const ALLOWED_IMAGE_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/webp",
]);


const ui = getUiElements();

const maximumUploadMb = Number(
    ui.body.dataset.maxUploadMb
    ?? 5
);

const maximumUploadBytes = (
    maximumUploadMb
    * 1024
    * 1024
);


let selectedFile = null;
let originalObjectUrl = null;
let activeRequestController = null;


ui.imageInput.addEventListener(
    "change",
    handleFileSelection,
);


ui.form.addEventListener(
    "submit",
    handleFormSubmission,
);


ui.clearButton.addEventListener(
    "click",
    resetApplication,
);


window.addEventListener(
    "beforeunload",
    cleanupResources,
);


function handleFileSelection(
    event,
) {
    const file = (
        event.target.files?.[0]
        ?? null
    );

    releaseObjectUrl();

    selectedFile = null;

    ui.resultsSection.hidden = true;

    if (file === null) {
        clearInterface(
            ui
        );

        return;
    }

    const validationError = validateFile(
        file
    );

    if (validationError !== null) {
        ui.form.reset();

        showStatus(
            ui,
            validationError,
            "error",
        );

        return;
    }

    selectedFile = file;

    originalObjectUrl = URL.createObjectURL(
        file
    );

    showSelectedFile(
        ui,
        file,
        originalObjectUrl,
    );
}


async function handleFormSubmission(
    event,
) {
    event.preventDefault();

    if (selectedFile === null) {
        showStatus(
            ui,
            "Selecciona una imagen antes de iniciar el análisis.",
            "error",
        );

        return;
    }

    abortActiveRequest();

    activeRequestController = (
        new AbortController()
    );

    setLoading(
        ui,
        true
    );

    ui.resultsSection.hidden = true;

    showStatus(
        ui,
        "La imagen se está procesando.",
        "info",
    );

    try {
        const result = await analyzeImage(
            selectedFile,
            activeRequestController.signal,
        );

        renderAnalysis(
            ui,
            result,
            originalObjectUrl,
        );
    } catch (error) {
        if (error.name === "AbortError") {
            showStatus(
                ui,
                "El análisis fue cancelado.",
                "info",
            );

            return;
        }

        if (error instanceof AnalysisApiError) {
            showStatus(
                ui,
                error.message,
                "error",
            );

            return;
        }

        showStatus(
            ui,
            "Ocurrió un error inesperado en el navegador.",
            "error",
        );
    } finally {
        activeRequestController = null;

        setLoading(
            ui,
            false
        );

        ui.analyzeButton.disabled = (
            selectedFile === null
        );
    }
}


function validateFile(
    file,
) {
    if (
        file.type
        && !ALLOWED_IMAGE_TYPES.has(
            file.type
        )
    ) {
        return (
            "El formato no está permitido. "
            + "Utiliza JPEG, PNG o WebP."
        );
    }

    if (file.size <= 0) {
        return "El archivo seleccionado está vacío.";
    }

    if (
        file.size
        > maximumUploadBytes
    ) {
        return (
            "La imagen supera el tamaño máximo "
            + `permitido de ${maximumUploadMb} MB.`
        );
    }

    return null;
}


function resetApplication() {
    abortActiveRequest();
    releaseObjectUrl();

    selectedFile = null;

    clearInterface(
        ui
    );
}


function abortActiveRequest() {
    if (
        activeRequestController !== null
    ) {
        activeRequestController.abort();

        activeRequestController = null;
    }
}


function releaseObjectUrl() {
    if (
        originalObjectUrl !== null
    ) {
        URL.revokeObjectURL(
            originalObjectUrl
        );

        originalObjectUrl = null;
    }
}


function cleanupResources() {
    abortActiveRequest();
    releaseObjectUrl();
}

