FROM python:3.10-slim

# Install minimal system libraries required by OpenCV & FPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy project files
COPY --chown=user . .

# Start Flask via Gunicorn inside vision_ai_project using Render's dynamic $PORT
CMD ["sh", "-c", "gunicorn --chdir vision_ai_project -b 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 180 app:app"]