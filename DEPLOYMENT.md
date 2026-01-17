# Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Edge Deployment (NVIDIA Jetson)](#edge-deployment)
5. [Cloud/On-Premises Deployment](#cloud-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring Setup](#monitoring-setup)

## Prerequisites

### Hardware Requirements

**Minimum (CPU only)**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB

**Recommended (GPU)**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 50GB+

**Edge Devices**
- NVIDIA Jetson Nano/Xavier/Orin
- RAM: 4GB+
- Storage: 32GB+

### Software Requirements

- Python 3.10+
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)
- CUDA 12.1+ (for GPU)

## Local Development

### 1. Setup Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Application

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env

# Configure cameras
cp config/cameras.yaml.example config/cameras.yaml
nano config/cameras.yaml
```

### 3. Run Application

```bash
# Run with Python
python -m src.main

# Or run with logging
python -m src.main 2>&1 | tee logs/output.log
```

## Docker Deployment

### 1. Install Docker and NVIDIA Container Toolkit

**Ubuntu/Debian**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Build and Run

**Using Docker Compose (Recommended)**
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Using Docker Only**
```bash
# Build image
docker build -t ai-video-analytics:latest .

# Run with GPU
docker run -d \
  --name ai-video-analytics \
  --gpus all \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/alerts:/app/alerts \
  -p 8000:8000 \
  --restart unless-stopped \
  ai-video-analytics:latest

# Run without GPU (CPU only)
docker run -d \
  --name ai-video-analytics \
  -e DEVICE=cpu \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/alerts:/app/alerts \
  -p 8000:8000 \
  --restart unless-stopped \
  ai-video-analytics:latest
```

### 3. Verify Deployment

```bash
# Check container status
docker ps

# Check logs
docker logs ai-video-analytics

# Check metrics
curl http://localhost:8000/metrics
```

## Edge Deployment (NVIDIA Jetson)

### 1. Prepare Jetson Device

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. Optimize for Jetson

**Create Jetson-specific Dockerfile**
```dockerfile
# Use Jetson base image
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . .

# Set environment for Jetson
ENV DEVICE=cuda
ENV MODEL_PATH=models/yolov8n.pt
ENV RESIZE_WIDTH=416
ENV RESIZE_HEIGHT=416
ENV USE_TENSORRT=true

CMD ["python3", "-m", "src.main"]
```

### 3. Build and Deploy

```bash
# Build for Jetson
docker build -f Dockerfile.jetson -t ai-video-analytics:jetson .

# Run
docker run -d \
  --runtime nvidia \
  --name ai-video-analytics \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -p 8000:8000 \
  --restart unless-stopped \
  ai-video-analytics:jetson
```

## Cloud/On-Premises Deployment

### 1. Server Setup

**Install Prerequisites**
```bash
# Install Docker and NVIDIA toolkit (see Docker section)

# Install monitoring tools
sudo apt-get install -y htop iotop nethogs

# Configure firewall
sudo ufw allow 8000/tcp  # Metrics port
sudo ufw allow 9090/tcp  # Prometheus port
sudo ufw enable
```

### 2. Production Configuration

**docker-compose.prod.yml**
```yaml
version: '3.8'

services:
  ai-video-analytics:
    build: .
    container_name: ai-video-analytics
    runtime: nvidia
    restart: always
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./models:/app/models
      - ./alerts:/app/alerts
    ports:
      - "8000:8000"
    networks:
      - analytics-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: always
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - analytics-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=changeme
    networks:
      - analytics-network

volumes:
  prometheus-data:
  grafana-data:

networks:
  analytics-network:
    driver: bridge
```

### 3. Deploy

```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Check health
docker-compose -f docker-compose.prod.yml ps
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-video-analytics
  labels:
    app: ai-video-analytics
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-video-analytics
  template:
    metadata:
      labels:
        app: ai-video-analytics
    spec:
      containers:
      - name: ai-video-analytics
        image: ai-video-analytics:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEVICE
          value: "cuda"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
      volumes:
      - name: config
        configMap:
          name: camera-config
      - name: logs
        emptyDir: {}
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-video-analytics
spec:
  selector:
    app: ai-video-analytics
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace video-analytics

# Create ConfigMap from cameras.yaml
kubectl create configmap camera-config \
  --from-file=config/cameras.yaml \
  -n video-analytics

# Deploy application
kubectl apply -f deployment.yaml -n video-analytics
kubectl apply -f service.yaml -n video-analytics

# Check status
kubectl get pods -n video-analytics
kubectl get svc -n video-analytics
```

## Monitoring Setup

### 1. Prometheus Configuration

**config/prometheus.yml**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-video-analytics'
    static_configs:
      - targets: ['ai-video-analytics:8000']
    metrics_path: '/metrics'
```

### 2. Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/changeme)

**Create Dashboard**
1. Add Prometheus data source
2. Import dashboard or create custom panels
3. Add panels for:
   - Frames processed per second
   - People count timeline
   - Alert frequency
   - GPU utilization
   - Memory usage

### 3. Alert Rules

**prometheus-alerts.yml**
```yaml
groups:
  - name: video_analytics
    rules:
      - alert: HighPeopleCount
        expr: people_count > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High people count detected"
          
      - alert: StreamDisconnected
        expr: rate(frames_processed_total[5m]) == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Video stream disconnected"
```

## Backup and Recovery

### 1. Backup Strategy

```bash
# Backup configuration
tar -czf backup-config-$(date +%Y%m%d).tar.gz config/

# Backup alerts database
tar -czf backup-alerts-$(date +%Y%m%d).tar.gz alerts/

# Backup models
tar -czf backup-models-$(date +%Y%m%d).tar.gz models/
```

### 2. Automated Backup

**backup.sh**
```bash
#!/bin/bash
BACKUP_DIR=/backups
DATE=$(date +%Y%m%d-%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup files
tar -czf $BACKUP_DIR/config-$DATE.tar.gz config/
tar -czf $BACKUP_DIR/alerts-$DATE.tar.gz alerts/

# Keep only last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

**Add to crontab**
```bash
# Backup daily at 2 AM
0 2 * * * /path/to/backup.sh
```

## Scaling

### 1. Horizontal Scaling

- Deploy multiple instances with load balancer
- Each instance processes different camera streams
- Use shared storage for alerts/logs

### 2. Vertical Scaling

- Increase CPU/RAM allocation
- Add more powerful GPU
- Optimize model and frame processing

## Troubleshooting

### Check Logs
```bash
# Docker logs
docker logs ai-video-analytics

# Application logs
tail -f logs/video_analytics_*.log

# Alert logs
tail -f alerts/alert_log.json
```

### Performance Monitoring
```bash
# GPU utilization
nvidia-smi -l 1

# Container stats
docker stats ai-video-analytics

# System resources
htop
```

### Common Issues

**GPU not detected**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Out of memory**
- Reduce number of streams
- Decrease frame resolution
- Use smaller model

**Connection issues**
- Check RTSP URL
- Verify network connectivity
- Check firewall rules
