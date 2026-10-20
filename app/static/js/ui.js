function requiredElement(selector) {
    const element = document.querySelector(selector);

    if (element === null) {
        throw new Error(
            `No se encontró el elemento HTML requerido: ${selector}`
        );
    }

    return element;
}


export function getUiElements() {
    return {
        body: document.body,

        form: requiredElement(
            "#analysis-form"
        ),

        imageInput: requiredElement(
            "#image-input"
        ),

        analyzeButton: requiredElement(
            "#analyze-button"
        ),

        clearButton: requiredElement(
            "#clear-button"
        ),

        selectedFile: requiredElement(
            "#selected-file"
        ),

        selectedFileName: requiredElement(
            "#selected-file-name"
        ),

        selectedFileSize: requiredElement(
            "#selected-file-size"
        ),

        previewSection: requiredElement(
            "#preview-section"
        ),

        originalPreview: requiredElement(
            "#original-preview"
        ),

        loadingSection: requiredElement(
            "#loading-section"
        ),

        resultsSection: requiredElement(
            "#results-section"
        ),

        statusMessage: requiredElement(
            "#status-message"
        ),

        predictionLabel: requiredElement(
            "#prediction-label"
        ),

        predictionModelClass: requiredElement(
            "#prediction-model-class"
        ),

        confidenceValue: requiredElement(
            "#confidence-value"
        ),

        segmentedAreaValue: requiredElement(
            "#segmented-area-value"
        ),

        processingTimeValue: requiredElement(
            "#processing-time-value"
        ),

        probabilityList: requiredElement(
            "#probability-list"
        ),

        resultDisclaimer: requiredElement(
            "#result-disclaimer"
        ),

        resultOriginalImage: requiredElement(
            "#result-original-image"
        ),

        segmentedCropImage: requiredElement(
            "#segmented-crop-image"
        ),

        gradcamTissueOverlayImage: requiredElement(
            "#gradcam-tissue-overlay-image"
        ),
    };
}


export function showSelectedFile(
    ui,
    file,
    objectUrl,
) {
    ui.selectedFileName.textContent = file.name;

    ui.selectedFileSize.textContent = formatFileSize(
        file.size
    );

    ui.selectedFile.hidden = false;

    ui.originalPreview.src = objectUrl;
    ui.originalPreview.hidden = false;

    ui.previewSection.hidden = false;
    ui.analyzeButton.disabled = false;

    clearStatus(ui);
}


export function setLoading(
    ui,
    loading,
) {
    ui.loadingSection.hidden = !loading;

    ui.form.setAttribute(
        "aria-busy",
        String(loading),
    );

    ui.imageInput.disabled = loading;
    ui.clearButton.disabled = loading;
    ui.analyzeButton.disabled = loading;

    ui.analyzeButton.textContent = (
        loading
            ? "Analizando..."
            : "Analizar imagen"
    );
}


export function renderAnalysis(
    ui,
    result,
    originalObjectUrl,
) {
    validateAnalysisResponse(result);

    ui.predictionLabel.textContent = (
        result.predicted_label
    );

    ui.predictionModelClass.textContent = (
        `Clase del modelo: ${result.predicted_class}`
    );

    ui.confidenceValue.textContent = formatPercentage(
        result.confidence_percentage
    );

    ui.segmentedAreaValue.textContent = formatPercentage(
        result.segmentation.area_percentage,
        4,
    );

    ui.processingTimeValue.textContent = formatProcessingTime(
        result.processing_time_ms
    );

    ui.resultDisclaimer.textContent = (
        result.disclaimer
    );

    renderProbabilities(
        ui.probabilityList,
        result.probabilities,
    );

    ui.resultOriginalImage.src = (
        originalObjectUrl
    );

    ui.segmentedCropImage.src = (
        result.images.segmented_crop
    );

    ui.gradcamTissueOverlayImage.src = (
        result.images.gradcam_tissue_overlay
    );

    ui.resultsSection.hidden = false;

    showStatus(
        ui,
        "Análisis completado correctamente.",
        "success",
    );

    ui.resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}


export function clearInterface(
    ui,
) {
    ui.form.reset();

    ui.selectedFile.hidden = true;
    ui.selectedFileName.textContent = "";
    ui.selectedFileSize.textContent = "";

    ui.previewSection.hidden = true;
    ui.originalPreview.removeAttribute("src");

    ui.resultsSection.hidden = true;
    ui.loadingSection.hidden = true;

    ui.probabilityList.replaceChildren();

    clearResultImages(ui);

    ui.analyzeButton.disabled = true;
    ui.imageInput.disabled = false;
    ui.clearButton.disabled = false;

    ui.analyzeButton.textContent = (
        "Analizar imagen"
    );

    clearStatus(ui);
}


export function showStatus(
    ui,
    message,
    type = "info",
) {
    ui.statusMessage.textContent = message;

    ui.statusMessage.className = (
        `status-message status-${type}`
    );

    ui.statusMessage.hidden = false;
}


export function clearStatus(ui) {
    ui.statusMessage.textContent = "";

    ui.statusMessage.className = (
        "status-message"
    );

    ui.statusMessage.hidden = true;
}


function validateAnalysisResponse(result) {
    if (
        result === null
        || typeof result !== "object"
    ) {
        throw new Error(
            "La respuesta del análisis no es válida."
        );
    }

    if (
        result.images === null
        || typeof result.images !== "object"
    ) {
        throw new Error(
            "La respuesta no contiene las imágenes procesadas."
        );
    }

    if (
        typeof result.images.segmented_crop !== "string"
    ) {
        throw new Error(
            "La respuesta no contiene segmented_crop."
        );
    }

    if (
        typeof result.images.gradcam_tissue_overlay !== "string"
    ) {
        throw new Error(
            "La respuesta no contiene gradcam_tissue_overlay."
        );
    }

    if (
        !Array.isArray(result.probabilities)
    ) {
        throw new Error(
            "La respuesta no contiene probabilidades válidas."
        );
    }
}


function renderProbabilities(
    container,
    probabilities,
) {
    container.replaceChildren();

    for (const item of probabilities) {
        const row = document.createElement(
            "div"
        );

        row.className = "probability-item";

        const header = document.createElement(
            "div"
        );

        header.className = (
            "probability-item-header"
        );

        const label = document.createElement(
            "span"
        );

        label.textContent = (
            item.display_label
        );

        const value = document.createElement(
            "strong"
        );

        value.textContent = formatPercentage(
            item.percentage
        );

        header.append(
            label,
            value,
        );

        const track = document.createElement(
            "div"
        );

        track.className = "probability-track";

        track.setAttribute(
            "role",
            "progressbar",
        );

        track.setAttribute(
            "aria-valuemin",
            "0",
        );

        track.setAttribute(
            "aria-valuemax",
            "100",
        );

        track.setAttribute(
            "aria-valuenow",
            String(item.percentage),
        );

        const bar = document.createElement(
            "div"
        );

        bar.className = "probability-bar";

        const percentage = Math.min(
            100,
            Math.max(
                0,
                Number(item.percentage),
            ),
        );

        bar.style.width = `${percentage}%`;

        track.append(bar);

        row.append(
            header,
            track,
        );

        container.append(row);
    }
}


function clearResultImages(ui) {
    const images = [
        ui.resultOriginalImage,
        ui.segmentedCropImage,
        ui.gradcamTissueOverlayImage,
    ];

    for (const image of images) {
        image.removeAttribute("src");
    }
}


function formatPercentage(
    value,
    decimals = 2,
) {
    return `${Number(value).toFixed(decimals)}%`;
}


function formatProcessingTime(milliseconds) {
    const numericValue = Number(milliseconds);

    if (numericValue >= 1000) {
        return `${(numericValue / 1000).toFixed(2)} s`;
    }

    return `${numericValue.toFixed(2)} ms`;
}


function formatFileSize(bytes) {
    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(2)} KB`;
    }

    return `${
        (
            bytes
            / 1024
            / 1024
        ).toFixed(2)
    } MB`;
}
