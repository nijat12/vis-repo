#!/bin/bash
# VIS Pipeline - VM Initialization Script
# This script installs Docker and runs the VIS pipeline container on a Debian VM

set -e  # Exit on error

LOG_FILE="init_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "VIS Pipeline - VM Initialization"
echo "=========================================="
echo "Logging to: $LOG_FILE"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update >> "$LOG_FILE" 2>&1
sudo apt-get upgrade -y >> "$LOG_FILE" 2>&1

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    sudo apt-get install -y ca-certificates curl gnupg >> "$LOG_FILE" 2>&1
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update >> "$LOG_FILE" 2>&1
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >> "$LOG_FILE" 2>&1

    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    echo "✅ Docker installed successfully"
else
    echo "✅ Docker already installed"
fi

# Start Docker service
echo "🚀 Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker >> "$LOG_FILE" 2>&1

# Verify Docker installation
echo "🔍 Verifying Docker installation..."
sudo docker --version

# Build Docker image
echo "🔨 Building VIS Pipeline Docker image..."
cd /home/$USER/vis-repo || { echo "❌ vis-repo directory not found"; exit 1; }

sudo docker build -t vis-pipeline:latest . 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Docker build failed! Check $LOG_FILE for details"
    exit 1
fi

echo "✅ Docker image built successfully"

# Run the container WITHOUT --rm to keep it for log inspection
# This allows:
# 1. Access to VM metadata service for GCS authentication
# 2. Ability to shutdown the VM from within the container
# 3. Process runs in background, you can disconnect SSH
echo "🚀 Running VIS Pipeline container in BACKGROUND..."

# Remove old container if exists
sudo docker rm -f vis-pipeline-run 2>/dev/null || true

sudo docker run \
    --name vis-pipeline-run \
    --detach \
    --network host \
    --privileged \
    -v /home/$USER/vis-repo/metrics:/app/metrics \
    -v /home/$USER/vis-repo/data_local:/app/data_local \
    -v /run/systemd/system:/run/systemd/system \
    vis-pipeline:latest

CONTAINER_ID=$(sudo docker ps -q -f name=vis-pipeline-run)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Container failed to start!"
    echo "Checking logs..."
    sudo docker logs vis-pipeline-run
    exit 1
fi

echo "=========================================="
echo "✅ VIS Pipeline started in BACKGROUND!"
echo "=========================================="
echo "   Container ID: $CONTAINER_ID"
echo "   Init log: $LOG_FILE"
echo ""
echo "📋 USEFUL COMMANDS:"
echo ""
echo "   View live logs:"
echo "   $ sudo docker logs -f vis-pipeline-run"
echo ""
echo "   Check status:"
echo "   $ sudo docker ps -a | grep vis-pipeline"
echo ""
echo "   Stop container:"
echo "   $ sudo docker stop vis-pipeline-run"
echo ""
echo "   View results:"
echo "   $ ls -lh ~/vis-repo/metrics/"
echo ""
echo "   Save logs to file:"
echo "   $ sudo docker logs vis-pipeline-run > pipeline_run.log 2>&1"
echo ""
echo "=========================================="
echo "💡 You can safely disconnect SSH - the process will continue running"
echo "   Results will be saved to: /home/$USER/vis-repo/metrics/"
echo "   If killswitch is enabled, VM will auto-shutdown when done"
echo "=========================================="
