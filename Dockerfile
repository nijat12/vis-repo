# Use an official lightweight Python image
FROM python:3.9-slim

# Install system dependencies for OpenCV (glib)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python libraries
# We use opencv-python-headless because there is no screen in the cloud
RUN pip install --no-cache-dir \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
    opencv-python-headless \
    pandas \
    tqdm \
    requests \
    PyYAML \
    seaborn 

# Copy your code into the container
COPY main.py .

# Command to run the script
CMD ["python", "main.py"]