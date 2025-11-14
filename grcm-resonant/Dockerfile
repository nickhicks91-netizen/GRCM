# Multi-stage Dockerfile for GRCM
# Optimized for production deployment with minimal size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY grcm/ ./grcm/
COPY configs/ ./configs/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .[optimization,serving,monitoring]

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY grcm/ ./grcm/
COPY configs/ ./configs/
COPY pyproject.toml README.md LICENSE ./

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 grcm && chown -R grcm:grcm /app
USER grcm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRCM_CONFIG=/app/configs/default.yaml \
    MLFLOW_TRACKING_URI=/app/mlruns

# Expose ports
EXPOSE 3000 5000 7860 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from grcm import ResonantConsciousnessModule; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-c", "from grcm import ResonantConsciousnessModule; print('GRCM Docker image ready')"]
