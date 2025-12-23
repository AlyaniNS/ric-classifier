# Multi-stage build to reduce final image size
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages to /install directory
# Install numpy first to ensure compatibility
RUN pip install --no-cache-dir --prefix=/install numpy==1.24.3 && \
    pip install --no-cache-dir --prefix=/install \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install \
    torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install \
    Flask==3.1.2 \
    Pillow==10.0.0 \
    timm==0.9.0 \
    matplotlib==3.7.5 \
    chartjs==1.2 \
    gunicorn==21.2.0 \
    Werkzeug==3.1.2

# Final stage - minimal runtime image
FROM python:3.9-slim

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py config.py model.py localization.py ./
COPY templates templates/
COPY static static/
COPY models models/

# Create necessary directories
RUN mkdir -p logs static/uploads && \
    touch logs/.gitkeep static/uploads/.gitkeep

# Cleanup Python cache
RUN find /usr/local -type f -name "*.pyc" -delete && \
    find /usr/local -type d -name "__pycache__" -delete && \
    find /usr/local -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Set Python to not create .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level info --access-logfile -
