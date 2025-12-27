# VIS Pipeline - Dockerfile
# Base image: Python 3.12-slim for stability and minimal footprint
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install essential system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    sudo \
    systemd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Using the CPU-only index defined in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and create directories
COPY . .
RUN mkdir -p /app/data_local /app/metrics

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py"]
