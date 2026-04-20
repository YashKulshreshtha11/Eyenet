# --- Dockerfile for Hugging Face Spaces (Backend Only) ---
FROM python:3.10-slim

# Create a non-root user (Hugging Face Requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies (OpenCV, etc.)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy backend code and models
COPY --chown=user . .

# Environment setup
ENV ENVIRONMENT=production
ENV MODEL_WEIGHTS_PATH=./weights/eyenet_ensemble.pth
ENV MONGODB_DATABASE_NAME=eyenet_db

# Hugging Face Spaces strictly requires Port 7860
EXPOSE 7860

# Start Backend Server
CMD ["python", "-m", "uvicorn", "backend.app_server:app", "--host", "0.0.0.0", "--port", "7860"]
