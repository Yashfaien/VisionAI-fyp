FROM python:3.10-slim

# Install minimal system libraries required by OpenCV & FPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Install Python dependencies first for faster Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files (app.py, templates/, static/, etc.)
COPY --chown=user . .

# Expose Hugging Face default port
EXPOSE 7860

# Start Flask via Gunicorn on port 7860 (with 180s timeout for gdown model download on startup)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]