# EyeNet Professional Deployment - Unified Container
# This Dockerfile builds the React frontend and serves it via the FastAPI backend.

# ── Phase 1: Build Frontend ──────────────────────────────────────────────────
FROM node:18-slim AS frontend-builder
WORKDIR /build-frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Phase 2: Production Server ────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRONTEND_DIR=/app/frontend

WORKDIR /app

# Install system dependencies for OpenCV and AI libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and database logic
COPY backend ./backend
COPY database ./database
COPY weights ./weights

# Copy the built frontend from Phase 1
COPY --from=frontend-builder /build-frontend/dist ./frontend/dist

EXPOSE 8000

# Start the unified application
CMD ["uvicorn", "backend.app_server:app", "--host", "0.0.0.0", "--port", "8000"]
