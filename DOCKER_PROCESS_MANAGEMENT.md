# Managing Background Docker Process

## Quick Reference

### View Live Logs (Follow Mode)
```bash
sudo docker logs -f vis-pipeline-run
```
Press `Ctrl+C` to exit (container keeps running)

### View Recent Logs
```bash
# Last 100 lines
sudo docker logs --tail 100 vis-pipeline-run

# Last 10 minutes
sudo docker logs --since 10m vis-pipeline-run
```

### Check Container Status
```bash
# Is it running?
sudo docker ps | grep vis-pipeline

# All containers (including stopped)
sudo docker ps -a | grep vis-pipeline

# Detailed info
sudo docker inspect vis-pipeline-run
```

### Stop the Container
```bash
# Graceful stop (10 second timeout)
sudo docker stop vis-pipeline-run

# Force stop immediately
sudo docker kill vis-pipeline-run
```

### Find Container ID
```bash
# Get container ID
sudo docker ps -q -f name=vis-pipeline-run

# Or use name directly
sudo docker logs vis-pipeline-run
```

## Detailed Commands

### Monitor Resource Usage
```bash
# CPU, memory, network stats
sudo docker stats vis-pipeline-run

# One-time snapshot
sudo docker stats --no-stream vis-pipeline-run
```

### Execute Commands Inside Container
```bash
# Open interactive shell
sudo docker exec -it vis-pipeline-run /bin/bash

# Run single command
sudo docker exec vis-pipeline-run ls -lh /app/metrics
```

### Check Results While Running
```bash
# From host VM
ls -lh ~/vis-repo/metrics/

# Watch for new files
watch -n 5 ls -lh ~/vis-repo/metrics/
```

### Restart Container
```bash
# Stop and remove
sudo docker stop vis-pipeline-run

# Run again
cd ~/vis-repo && ./init.sh
```

## Troubleshooting

### Container Exited Unexpectedly
```bash
# Check exit code and logs
sudo docker ps -a | grep vis-pipeline
sudo docker logs vis-pipeline-run
```

### Container Not Found
```bash
# List all containers
sudo docker ps -a

# Check if it finished and was removed (--rm flag)
ls -lh ~/vis-repo/metrics/  # Results should still be there
```

### Out of Disk Space
```bash
# Check disk usage
df -h

# Clean up Docker
sudo docker system prune -a
```

### Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Then run without sudo
docker logs vis-pipeline-run
```

## Background Process Workflow

### 1. Start Process
```bash
cd ~/vis-repo
./init.sh
```

### 2. Verify It's Running
```bash
sudo docker ps | grep vis-pipeline
```

### 3. Disconnect SSH (Optional)
```bash
exit
```
The container keeps running even after you disconnect!

### 4. Reconnect Later
```bash
# SSH back in
gcloud compute ssh vis-pipeline-vm --zone=us-central1-f

# Check if still running
sudo docker ps | grep vis-pipeline

# View logs
sudo docker logs -f vis-pipeline-run
```

### 5. Retrieve Results
```bash
# While running or after completion
ls -lh ~/vis-repo/metrics/

# Download to local machine
gcloud compute scp vis-pipeline-vm:~/vis-repo/metrics/*.xlsx ./ --zone=us-central1-f
```

## Monitoring Tips

### Set Up Log Streaming
```bash
# Stream logs to file
sudo docker logs -f vis-pipeline-run > pipeline.log 2>&1 &

# View the file
tail -f pipeline.log
```

### Check Progress
```bash
# Count processed images
sudo docker logs vis-pipeline-run | grep "Processing" | wc -l

# Check for errors
sudo docker logs vis-pipeline-run | grep -i error

# Check for completion
sudo docker logs vis-pipeline-run | grep "COMPLETED"
```

### Monitor GCS Uploads
```bash
# Check if results uploaded
gsutil ls gs://vis-data-2025/results/

# Download from GCS
gsutil cp gs://vis-data-2025/results/results_*.xlsx ./
```

## Automatic Cleanup

The container uses `--rm` flag, so it automatically removes itself when finished.

**What gets cleaned up:**
- ✅ Container (automatically removed)
- ✅ Temporary files inside container

**What persists:**
- ✅ Results in `~/vis-repo/metrics/`
- ✅ Downloaded data in `~/vis-repo/data_local/`
- ✅ Docker image `vis-pipeline:latest`

## Common Scenarios

### Scenario 1: Check if pipeline finished
```bash
sudo docker ps -a | grep vis-pipeline
# If not listed, it finished and was removed
ls -lh ~/vis-repo/metrics/  # Check results
```

### Scenario 2: Pipeline taking too long
```bash
# Check what it's doing
sudo docker logs --tail 50 vis-pipeline-run

# Check resource usage
sudo docker stats vis-pipeline-run
```

### Scenario 3: Need to stop and restart
```bash
# Stop current run
sudo docker stop vis-pipeline-run

# Modify config
nano ~/vis-repo/config.py

# Rebuild and run
cd ~/vis-repo
sudo docker build -t vis-pipeline:latest .
./init.sh
```

### Scenario 4: VM will shutdown soon (SPOT preemption)
```bash
# Quickly save current state
sudo docker logs vis-pipeline-run > emergency-logs.txt

# Results are already in ~/vis-repo/metrics/
# Upload to GCS manually if needed
gsutil cp ~/vis-repo/metrics/*.xlsx gs://vis-data-2025/results/
```
