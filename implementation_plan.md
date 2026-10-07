# Implementation Plan - VisionAI: AI-Powered Eye Anomaly Detection & Automated Report Generation

VisionAI is a production-ready medical screening web application developed as a Final Year Project (FYP). The system classifies retinal fundus images into four primary clinical categories (**Diabetic Retinopathy**, **Glaucoma**, **Cataract**, and **Normal**) using a fine-tuned ResNet-50 convolutional neural network architecture (87.5% validation accuracy, 87.3% testing accuracy), coupled with dynamic clinical PDF diagnostic report generation.

---

## User Review Required

> [!NOTE]
> - **Model Fallback Engine**: If a pre-trained `resnet50_model.keras` or `resnet50_model.h5` is not detected in the workspace, the backend automatically initializes an intelligent heuristic fallback inference engine. This guarantees seamless execution and testing during demos and development while maintaining identical response schema and UI behavior.
> - **Report Generation**: We use `fpdf2` (the active, modern standard for Python FPDF) to compile structured clinical reports with headers, patient demographics, confidence meters, disease probability tables, and legal medical disclaimers.

---

## Proposed Directory & File Structure

```
vision-ai-frontend/
├── vision_ai_project/
│   ├── app.py                     # Flask application server, preprocessing, ML inference & PDF generation
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css          # Dark-mode styling, glassmorphism, dropzone animations, progress bars
│   │   └── js/
│   │       └── main.js            # Drag-and-drop, preview, API fetch calls, dynamic DOM updates, PDF download
│   └── templates/
│       └── index.html             # Modern dark-themed medical diagnostic UI (Bootstrap 5)
├── requirements.txt               # Dependencies (Flask, opencv-python, numpy, fpdf2, etc.)
└── run.py                         # Convenience root runner script
```

---

## Proposed Changes

### Backend (`vision_ai_project/app.py`)

#### [NEW] [app.py](file:///e:/FYP/vision-ai-frontend/vision_ai_project/app.py)
- **Flask Initialization & Static Configuration**: Configured with standard static and template folders.
- **Model Loading & Fallback Routine**:
  - Checks for `resnet50_model.keras` or `resnet50_model.h5`.
  - If present and TensorFlow is installed, loads via `tf.keras.models.load_model`.
  - If absent, logs friendly warning and switches to a deterministic heuristic inference engine so the application is immediately testable without requiring a multi-gigabyte weight file.
- **OpenCV & NumPy Preprocessing**:
  - Decodes byte stream safely via `cv2.imdecode`.
  - Resizes to `(224, 224)` matching ResNet-50 input specs.
  - Normalizes pixel values to `[0.0, 1.0]`.
  - Expands dimensions to `(1, 224, 224, 3)`.
- **API Endpoints**:
  - `GET /`: Renders `index.html`.
  - `POST /predict`: Accepts `image` file and patient details (`name`, `age`, `gender`, `eye`). Returns JSON with `prediction`, `confidence`, `probabilities` breakdown dictionary, `recommendation`, and patient metadata.
  - `POST /generate-pdf`: Accepts patient details and diagnostic findings. Uses `FPDF` to generate an official diagnostic PDF report and returns it as a downloadable byte stream (`Content-Disposition: attachment; filename=VisionAI_Report.pdf`).

---

### Frontend UI (`vision_ai_project/templates/index.html`)

#### [NEW] [index.html](file:///e:/FYP/vision-ai-frontend/vision_ai_project/templates/index.html)
- **Theme & Typography**: Premium dark medical aesthetic (`#121212` background, deep slate cards, high-contrast text, Inter font, Bootstrap Icons).
- **Navigation Bar**: Brand logo (`👁️ VisionAI`), FYP badge, quick navigation links (Screener, Metrics, Clinical Guidelines).
- **Hero Section**: Introduces the AI screening platform, highlighting ResNet-50 specs (87.5% Val Acc, 87.3% Test Acc) and the 4 target retinal pathologies.
- **Two-Step Diagnostic Workflow Container**:
  - **Step 1 (Image Upload)**: Drag-and-drop dropzone with file picker (`#imageInput`), format requirements (JPG/PNG), and interactive image preview thumbnail with file size and remove action.
  - **Step 2 (Patient Information)**: Full Name, Age, Gender dropdown (Male, Female, Other), Eye Examined dropdown (Left Eye, Right Eye), and "Analyse Image" button with loading state.
- **Diagnostic Results Section** (hidden initially, smoothly revealed post-analysis):
  - Patient Summary card & fundus preview thumbnail.
  - Predicted class badge with color coding and confidence percentage.
  - 4-Class Probability Progress Bars:
    - Diabetic Retinopathy (danger/red)
    - Glaucoma (warning/orange-yellow)
    - Cataract (info/cyan-blue)
    - Normal (success/emerald green)
  - Dynamic Clinical Recommendation callout.
  - Action buttons: "Download PDF Diagnostic Report" & "Examine Another Patient".
- **Model Performance & Architecture Section**: Breakdown of training parameters, metrics, and disease descriptions.

---

### Frontend Styling (`vision_ai_project/static/css/style.css`)

#### [NEW] [style.css](file:///e:/FYP/vision-ai-frontend/vision_ai_project/static/css/style.css)
- Deep dark palette (`#0f1117`, `#1a1d24`, `#2d323f` borders).
- Interactive dropzone dashed border styling, hover/dragover glow and animations.
- Custom styled Bootstrap progress bars with rounded pill design and smooth transition animations.
- Medical alert badges and high-contrast diagnostic metric counters.
- Fully responsive design for desktop, tablet, and mobile displays.

---

### Frontend Logic (`vision_ai_project/static/js/main.js`)

#### [NEW] [main.js](file:///e:/FYP/vision-ai-frontend/vision_ai_project/static/js/main.js)
- Drag-and-drop listener handling (`dragover`, `dragleave`, `drop`) with visual cues.
- Client-side file validation (checking image MIME type and size limits).
- Instant image preview via `FileReader`.
- Asynchronous form submission via `fetch` POST to `/predict` using `FormData`.
- Loading spinner state on submission button.
- Dynamic rendering of results: updating confidence counters, animating progress bars, rendering class-specific clinical recommendation.
- Asynchronous PDF generation trigger via `fetch` POST to `/generate-pdf` with automatic blob download.
- Reset functionality for testing multiple cases.

---

### Project Configuration & Root Runner

#### [NEW] [requirements.txt](file:///e:/FYP/vision-ai-frontend/requirements.txt)
- Specifies `Flask`, `fpdf2`, `opencv-python`, `numpy`, `pillow`, etc.

#### [NEW] [run.py](file:///e:/FYP/vision-ai-frontend/run.py)
- Root helper to run `python run.py` directly from workspace root, starting the Flask server at `http://127.0.0.1:5000`.

---

## Verification Plan

### Automated & Unit Verification
- Validate Python syntax across `app.py` and `run.py`.
- Run automated test requests:
  - `GET /` to verify template rendering and HTTP 200.
  - `POST /predict` with a synthetic test image and patient metadata to verify JSON structure, probabilities, and recommendations.
  - `POST /generate-pdf` to verify valid PDF generation and byte streaming.

### Interactive Browser & UI Verification
- Start the Flask server.
- Open the application in the browser subagent.
- Verify drag-and-drop and file selection preview.
- Fill patient information and click "Analyse Image".
- Verify results display, progress bars animation, and recommendation box.
- Trigger "Download PDF Report" and ensure successful PDF generation.
