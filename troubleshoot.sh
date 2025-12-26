#!/bin/bash
# Quick troubleshooting script for Docker container issues

echo "=========================================="
echo "VIS Pipeline - Container Troubleshooting"
echo "=========================================="

# Find the container
CONTAINER_ID=$(sudo docker ps -a -q -f name=vis-pipeline-run | head -1)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ No container found with name 'vis-pipeline-run'"
    echo ""
    echo "All containers:"
    sudo docker ps -a
    exit 1
fi

echo "✅ Found container: $CONTAINER_ID"
echo ""

# Check container status
echo "📊 Container Status:"
sudo docker ps -a | grep $CONTAINER_ID
echo ""

# Get exit code
EXIT_CODE=$(sudo docker inspect $CONTAINER_ID --format='{{.State.ExitCode}}')
echo "Exit Code: $EXIT_CODE"
echo ""

# Show last 50 lines of logs
echo "=========================================="
echo "📋 Container Logs (last 50 lines):"
echo "=========================================="
sudo docker logs --tail 50 $CONTAINER_ID

echo ""
echo "=========================================="
echo "💡 Full logs command:"
echo "   sudo docker logs $CONTAINER_ID"
echo ""
echo "💡 Follow logs in real-time:"
echo "   sudo docker logs -f $CONTAINER_ID"
echo "=========================================="
