# VIS Pipeline - Dockerfile
# Base image: Python 3.14.2 slim for minimal footprint
FROM python:3.14.2-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
# gcc/g++/make: Required for compiling numpy and other Python packages with C extensions
# libgl1/libglib2.0-0: Required for OpenCV
# sudo/systemd: Required for VM shutdown capability
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    sudo \
    systemd \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# google-cloud-storage library handles GCS access using VM credentials (no CLI needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data_local /app/metrics

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py"]
