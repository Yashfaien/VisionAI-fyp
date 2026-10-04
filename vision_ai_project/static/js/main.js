/**
 * VisionAI - Frontend Application Logic
 * Retinal Fundus Image Previews, Inference API Integration & PDF Report Triggering
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const imageInput = document.getElementById("imageInput");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const previewFilename = document.getElementById("previewFilename");
  const previewFilesize = document.getElementById("previewFilesize");
  const removeImageBtn = document.getElementById("removeImageBtn");

  const diagnosticForm = document.getElementById("diagnosticForm");
  const patientNameInput = document.getElementById("patientName");
  const patientAgeInput = document.getElementById("patientAge");
  const patientGenderInput = document.getElementById("patientGender");
  const patientEyeInput = document.getElementById("patientEye");
  const submitBtn = document.getElementById("submitBtn");
  const btnIcon = document.getElementById("btnIcon");
  const btnText = document.getElementById("btnText");

  const alertContainer = document.getElementById("alertContainer");
  const resultsSection = document.getElementById("resultsSection");

  // Result Elements
  const resPatientName = document.getElementById("resPatientName");
  const resPatientAge = document.getElementById("resPatientAge");
  const resPatientGender = document.getElementById("resPatientGender");
  const resPatientEye = document.getElementById("resPatientEye");
  const resTimestamp = document.getElementById("resTimestamp");
  const resScanImage = document.getElementById("resScanImage");
  const resOverlayBadge = document.getElementById("resOverlayBadge");
  const resModelSource = document.getElementById("resModelSource");

  const resultHeroBox = document.getElementById("resultHeroBox");
  const resPrediction = document.getElementById("resPrediction");
  const resConfidence = document.getElementById("resConfidence");
  const resSeverity = document.getElementById("resSeverity");
  const resFollowUp = document.getElementById("resFollowUp");
  const resRecommendation = document.getElementById("resRecommendation");

  // Progress Bars & Values
  const valDR = document.getElementById("valDR");
  const barDR = document.getElementById("barDR");
  const valGlaucoma = document.getElementById("valGlaucoma");
  const barGlaucoma = document.getElementById("barGlaucoma");
  const valCataract = document.getElementById("valCataract");
  const barCataract = document.getElementById("barCataract");
  const valNormal = document.getElementById("valNormal");
  const barNormal = document.getElementById("barNormal");

  // PDF & Reset Actions
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");
  const pdfBtnIcon = document.getElementById("pdfBtnIcon");
  const pdfBtnText = document.getElementById("pdfBtnText");
  const resetBtn = document.getElementById("resetBtn");
  const newScreeningBtn = document.getElementById("newScreeningBtn");

  // State
  let selectedFile = null;
  let currentPreviewUrl = null;
  let latestAnalysisData = null;

  // --------------------------------------------------------------------------
  // Utility: Show Alert Message
  // --------------------------------------------------------------------------
  function showAlert(message, type = "danger") {
    alertContainer.innerHTML = `
      <div class="alert alert-${type} alert-dismissible fade show d-flex align-items-center" role="alert">
        <i class="bi bi-${type === 'danger' ? 'exclamation-octagon-fill' : 'info-circle-fill'} me-2 fs-5"></i>
        <div>${message}</div>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      </div>
    `;
    alertContainer.style.display = "block";
    alertContainer.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function clearAlert() {
    alertContainer.innerHTML = "";
    alertContainer.style.display = "none";
  }

  // --------------------------------------------------------------------------
  // Utility: Format File Size
  // --------------------------------------------------------------------------
  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  // --------------------------------------------------------------------------
  // Image Selection & Preview Handling
  // --------------------------------------------------------------------------
  function handleFileSelection(file) {
    clearAlert();

    if (!file) return;

    // Validate file type
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      showAlert("Invalid file format. Please upload a valid retinal image in JPG, JPEG, or PNG format.");
      resetFileSelection();
      return;
    }

    // Validate file size (16MB maximum)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
      showAlert("The selected image exceeds the 16MB file size limit. Please upload a smaller image.");
      resetFileSelection();
      return;
    }

    selectedFile = file;

    // Read and preview
    const reader = new FileReader();
    reader.onload = (e) => {
      currentPreviewUrl = e.target.result;
      imagePreview.src = currentPreviewUrl;
      previewFilename.innerHTML = `<i class="bi bi-file-image me-1"></i> ${file.name}`;
      previewFilesize.textContent = formatBytes(file.size);

      dropzone.style.display = "none";
      previewContainer.classList.add("active");

      validateFormState();
    };

    reader.onerror = () => {
      showAlert("Failed to read the selected file. Please try again.");
      resetFileSelection();
    };

    reader.readAsDataURL(file);
  }

  function resetFileSelection() {
    selectedFile = null;
    currentPreviewUrl = null;
    imageInput.value = "";
    imagePreview.src = "";
    dropzone.style.display = "block";
    previewContainer.classList.remove("active");
    validateFormState();
  }

  // --------------------------------------------------------------------------
  // Dropzone Events
  // --------------------------------------------------------------------------
  dropzone.addEventListener("click", () => {
    imageInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      imageInput.click();
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  removeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  // Drag and Drop Events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelection(dt.files[0]);
    }
  });

  // --------------------------------------------------------------------------
  // Form Validation & State
  // --------------------------------------------------------------------------
  function validateFormState() {
    const hasImage = selectedFile !== null;
    const hasName = patientNameInput.value.trim().length > 0;
    const hasAge = patientAgeInput.value.trim().length > 0 && Number(patientAgeInput.value) > 0;
    const hasGender = patientGenderInput.value !== "";
    const hasEye = patientEyeInput.value !== "";

    if (hasImage && hasName && hasAge && hasGender && hasEye) {
      submitBtn.removeAttribute("disabled");
    } else {
      submitBtn.setAttribute("disabled", "true");
    }
  }

  [patientNameInput, patientAgeInput, patientGenderInput, patientEyeInput].forEach((input) => {
    input.addEventListener("input", validateFormState);
    input.addEventListener("change", validateFormState);
  });

  // --------------------------------------------------------------------------
  // Prediction Submission Handler
  // --------------------------------------------------------------------------
  diagnosticForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAlert();

    if (!selectedFile) {
      showAlert("Please upload a retinal fundus image before submitting.");
      return;
    }

    // Set UI loading state
    submitBtn.setAttribute("disabled", "true");
    btnIcon.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
    btnText.textContent = "Analyzing Retinal Scan...";

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("name", patientNameInput.value.trim());
    formData.append("age", patientAgeInput.value.trim());
    formData.append("gender", patientGenderInput.value);
    formData.append("eye", patientEyeInput.value);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Image analysis failed on the server.");
      }

      // Store analysis data for PDF export
      latestAnalysisData = data;

      // Populate results in DOM
      renderResults(data);

    } catch (err) {
      console.error("Analysis Error:", err);
      showAlert(`Diagnostic failed: ${err.message}`);
    } finally {
      // Restore button state
      validateFormState();
      btnIcon.innerHTML = `<i class="bi bi-lightning-charge-fill me-1"></i>`;
      btnText.textContent = "Analyse Retinal Scan";
    }
  });

  // --------------------------------------------------------------------------
  // Render Prediction Results
  // --------------------------------------------------------------------------
  function renderResults(data) {
    const { prediction, confidence, probabilities, recommendation, severity, follow_up, patient, model_info, timestamp } = data;

    // 1. Patient Profile
    resPatientName.textContent = patient.name || "Anonymous Patient";
    resPatientAge.textContent = `${patient.age || "N/A"} Years`;
    resPatientGender.textContent = patient.gender || "Not Specified";
    resPatientEye.textContent = patient.eye || "Not Specified";
    resTimestamp.textContent = timestamp || new Date().toLocaleString();

    // 2. Examined Image Display
    if (currentPreviewUrl) {
      resScanImage.src = currentPreviewUrl;
    }

    // 3. Model Engine Info
    if (model_info && model_info.source) {
      resModelSource.textContent = model_info.source;
    }

    // 4. Primary Diagnosis Styling
    resPrediction.textContent = prediction;
    resConfidence.innerHTML = `<i class="bi bi-shield-check"></i> ${confidence.toFixed(1)}% Confidence`;
    resSeverity.textContent = severity;
    resFollowUp.textContent = follow_up;
    resRecommendation.textContent = recommendation;

    // Reset hero box modifier classes
    resultHeroBox.className = "result-hero-box";
    resPrediction.className = "diag-name";
    resConfidence.className = "confidence-badge";

    if (prediction === "Diabetic Retinopathy") {
      resultHeroBox.classList.add("dr");
      resPrediction.classList.add("text-danger");
      resConfidence.classList.add("text-danger");
      resOverlayBadge.className = "position-absolute top-0 end-0 m-2 badge bg-danger";
      resOverlayBadge.textContent = "High Risk: DR";
    } else if (prediction === "Glaucoma") {
      resultHeroBox.classList.add("glaucoma");
      resPrediction.classList.add("text-warning");
      resConfidence.classList.add("text-warning");
      resOverlayBadge.className = "position-absolute top-0 end-0 m-2 badge bg-warning text-dark";
      resOverlayBadge.textContent = "High Risk: Glaucoma";
    } else if (prediction === "Cataract") {
      resultHeroBox.classList.add("cataract");
      resPrediction.classList.add("text-info");
      resConfidence.classList.add("text-info");
      resOverlayBadge.className = "position-absolute top-0 end-0 m-2 badge bg-info text-dark";
      resOverlayBadge.textContent = "Moderate Risk: Cataract";
    } else {
      resultHeroBox.classList.add("normal");
      resPrediction.classList.add("text-success");
      resConfidence.classList.add("text-success");
      resOverlayBadge.className = "position-absolute top-0 end-0 m-2 badge bg-success";
      resOverlayBadge.textContent = "Clear: Normal";
    }

    // 5. Update Probability Breakdown Bars
    const probDR = probabilities["Diabetic Retinopathy"] || 0;
    const probGlaucoma = probabilities["Glaucoma"] || 0;
    const probCataract = probabilities["Cataract"] || 0;
    const probNormal = probabilities["Normal"] || 0;

    valDR.textContent = `${probDR.toFixed(1)}%`;
    valGlaucoma.textContent = `${probGlaucoma.toFixed(1)}%`;
    valCataract.textContent = `${probCataract.toFixed(1)}%`;
    valNormal.textContent = `${probNormal.toFixed(1)}%`;

    // Trigger smooth CSS progress transitions
    setTimeout(() => {
      barDR.style.width = `${Math.min(100, Math.max(0, probDR))}%`;
      barGlaucoma.style.width = `${Math.min(100, Math.max(0, probGlaucoma))}%`;
      barCataract.style.width = `${Math.min(100, Math.max(0, probCataract))}%`;
      barNormal.style.width = `${Math.min(100, Math.max(0, probNormal))}%`;
    }, 100);

    // 6. Reveal Results Section with smooth animation & scroll
    resultsSection.classList.add("visible");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // --------------------------------------------------------------------------
  // PDF Report Download Handler
  // --------------------------------------------------------------------------
  downloadPdfBtn.addEventListener("click", async () => {
    if (!latestAnalysisData) {
      showAlert("No completed diagnostic report available to download.");
      return;
    }

    try {
      // Set PDF button loading state
      downloadPdfBtn.setAttribute("disabled", "true");
      pdfBtnIcon.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
      pdfBtnText.textContent = "Compiling Medical PDF...";

      const response = await fetch("/generate-pdf", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(latestAnalysisData)
      });

      if (!response.ok) {
        throw new Error(`PDF generation server responded with status: ${response.status}`);
      }

      // Read response as Blob
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);

      // Create temporary invisible link to trigger browser download
      const patientNameSlug = (latestAnalysisData.patient.name || "Patient").replace(/[^a-zA-Z0-9_-]/g, "_");
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = downloadUrl;
      a.download = `VisionAI_Report_${patientNameSlug}.pdf`;
      document.body.appendChild(a);
      a.click();

      // Clean up blob URL after download starts
      setTimeout(() => {
        window.URL.revokeObjectURL(downloadUrl);
        a.remove();
      }, 1000);

    } catch (err) {
      console.error("PDF Download Error:", err);
      showAlert(`Failed to compile and download clinical PDF report: ${err.message}`);
    } finally {
      // Restore PDF button state
      downloadPdfBtn.removeAttribute("disabled");
      pdfBtnIcon.innerHTML = `<i class="bi bi-file-earmark-pdf-fill me-1"></i>`;
      pdfBtnText.textContent = "Download Official PDF Report";
    }
  });

  // --------------------------------------------------------------------------
  // Reset / New Screening Handler
  // --------------------------------------------------------------------------
  function resetWorkflow() {
    diagnosticForm.reset();
    resetFileSelection();
    latestAnalysisData = null;

    // Reset progress bars
    barDR.style.width = "0%";
    barGlaucoma.style.width = "0%";
    barCataract.style.width = "0%";
    barNormal.style.width = "0%";

    // Hide results section
    resultsSection.classList.remove("visible");

    // Scroll smoothly to Step 1
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  resetBtn.addEventListener("click", resetWorkflow);
  newScreeningBtn.addEventListener("click", resetWorkflow);

  // Initial validation check
  validateFormState();
});
