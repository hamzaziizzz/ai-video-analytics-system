# AI Video Analytics System

Production-grade AI Video Analytics system for CCTV — supports people counting, intrusion & safety detection with stable alert logic, multi-stream RTSP ingestion, GPU-optimized inference (YOLO, DeepStream, TensorRT), and scalable edge/on-prem deployment.

## Features

### 🎯 Core Analytics
- **People Counting**: Real-time detection and counting of people in video streams
- **Intrusion Detection**: Zone-based detection of unauthorized entry into restricted areas
- **Safety Detection**: PPE (Personal Protective Equipment) compliance monitoring
- **Multi-Stream Support**: Handle multiple camera feeds simultaneously (configurable limit)

### 🚨 Smart Alert System
- **Stable Alert Logic**: Cooldown-based deduplication prevents alert spam
- **Multiple Notification Channels**: 
  - Webhook integration for real-time alerts
  - Email notifications
  - JSON log files for audit trails
- **Configurable Alert Rules**: Customize thresholds and conditions

### 🚀 High Performance
- **GPU Acceleration**: Full CUDA support for NVIDIA GPUs
- **TensorRT Optimization**: Optional TensorRT export for maximum inference speed
- **Multi-threaded Processing**: Concurrent processing of multiple video streams
- **Frame Buffering**: Smart buffering to handle network latency

### 📊 Monitoring & Observability
- **Prometheus Metrics**: Built-in metrics export for monitoring
- **Structured Logging**: Comprehensive logging with rotation
- **Real-time Statistics**: Track frames processed, alerts sent, and more

### 🐳 Production Ready
- **Docker Support**: Containerized deployment with GPU support
- **Docker Compose**: Multi-service orchestration
- **Configuration Management**: Environment-based and YAML configuration
- **Graceful Shutdown**: Proper cleanup and resource management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Video Analytics System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ RTSP Stream 1│  │ RTSP Stream 2│  │ RTSP Stream N│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │ Stream Manager  │                        │
│                   │  (Multi-thread) │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │ Detection Engine│                        │
│                   │  (YOLO + GPU)   │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │Analytics Engine │                        │
│                   │ • People Count  │                        │
│                   │ • Intrusion     │                        │
│                   │ • Safety        │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│                   ┌────────▼────────┐                        │
│                   │  Alert Manager  │                        │
│                   │  (Deduplication)│                        │
│                   └────────┬────────┘                        │
│                            │                                  │
│         ┌──────────────────┼──────────────────┐              │
│         │                  │                  │               │
│    ┌────▼────┐      ┌──────▼──────┐    ┌────▼────┐         │
│    │ Webhook │      │    Email    │    │  Logs   │         │
│    └─────────┘      └─────────────┘    └─────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (optional, falls back to CPU)
- Docker and Docker Compose (for containerized deployment)
- RTSP camera streams or test videos

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hamzaziizzz/ai-video-analytics-system.git
cd ai-video-analytics-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure the system**
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env

# Configure cameras
nano config/cameras.yaml
```

5. **Download YOLO model** (if not using automatic download)
```bash
# The system will automatically download yolov8n.pt on first run
# Or manually download to models/ directory
```

6. **Run the system**
```bash
python -m src.main
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f ai-video-analytics

# Stop services
docker-compose down
```

### Using Docker Only

```bash
# Build image
docker build -t ai-video-analytics .

# Run container with GPU support
docker run --gpus all \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/alerts:/app/alerts \
  -p 8000:8000 \
  ai-video-analytics
```

## Configuration

### Environment Variables (.env)

```bash
# Application
APP_NAME=AI Video Analytics System
LOG_LEVEL=INFO
DEVICE=cuda  # or cpu

# Model Configuration
MODEL_TYPE=yolov8
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# GPU Settings
USE_TENSORRT=false
TENSORRT_PRECISION=fp16

# Video Streams
MAX_STREAMS=10
FRAME_SKIP=0
RESIZE_WIDTH=640
RESIZE_HEIGHT=480

# Alert Configuration
ALERT_COOLDOWN_SECONDS=30
ENABLE_WEBHOOK=true
WEBHOOK_URL=http://your-webhook-endpoint.com/alerts
ENABLE_EMAIL=false

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8000
```

### Camera Configuration (config/cameras.yaml)

```yaml
cameras:
  - id: "camera_1"
    name: "Main Entrance"
    rtsp_url: "rtsp://username:password@192.168.1.100:554/stream1"
    enabled: true
    analytics:
      people_counting: true
      intrusion_detection: true
      safety_detection: false
    zones:
      - name: "Restricted Area"
        type: "intrusion"
        polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]
        alert_on_entry: true

alert_rules:
  max_people_count: 50
  intrusion_immediate: true
  safety_violation_threshold: 3
```

## API & Metrics

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

Available metrics:
- `frames_processed_total{camera_id}` - Total frames processed per camera
- `people_count{camera_id}` - Current people count per camera
- `alerts_sent_total{alert_type}` - Total alerts sent by type

### Webhook Payload

```json
{
  "alert_type": "intrusion",
  "severity": "critical",
  "camera_id": "camera_1",
  "message": "Intrusion detected in Restricted Area",
  "metadata": {
    "zone": "Restricted Area",
    "detections": 2
  },
  "timestamp": "2026-01-17T15:04:49.920Z"
}
```

## Performance Optimization

### GPU Acceleration

1. **Verify CUDA availability**
```python
import torch
print(torch.cuda.is_available())
```

2. **Enable TensorRT** (requires NVIDIA TensorRT)
```bash
USE_TENSORRT=true
TENSORRT_PRECISION=fp16  # or fp32, int8
```

3. **Optimize model size vs accuracy**
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large (slowest, most accurate)

### Frame Processing

```bash
# Process every frame (highest accuracy)
FRAME_SKIP=0

# Process every 2nd frame (2x faster)
FRAME_SKIP=1

# Process every 3rd frame (3x faster)
FRAME_SKIP=2
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_alert_manager.py -v
```

### Test Coverage

The test suite covers:
- Detection engine functionality
- Alert management and deduplication
- Analytics engine logic
- Zone-based detection

## Monitoring & Troubleshooting

### Logs

Logs are stored in `logs/` directory with automatic rotation:
- Console output: Real-time logs
- File output: `logs/video_analytics_*.log`
- Alert logs: `alerts/alert_log.json`

### Common Issues

**1. CUDA out of memory**
- Reduce `RESIZE_WIDTH` and `RESIZE_HEIGHT`
- Use smaller model (`yolov8n.pt`)
- Reduce `MAX_STREAMS`

**2. RTSP connection failed**
- Verify camera URL and credentials
- Check network connectivity
- Ensure camera supports RTSP

**3. Low FPS**
- Enable GPU acceleration
- Increase `FRAME_SKIP`
- Optimize model size
- Enable TensorRT

## Production Deployment

### Edge Deployment

1. **Hardware Requirements**
   - NVIDIA Jetson (Nano, Xavier, Orin) or
   - x86 system with NVIDIA GPU
   - 4GB+ RAM
   - 10GB+ storage

2. **Optimization**
   - Use TensorRT for maximum performance
   - Enable INT8 quantization (requires calibration)
   - Reduce stream resolution

### On-Premises Deployment

1. **Server Setup**
   - Install Docker and NVIDIA Container Toolkit
   - Configure network access to cameras
   - Set up monitoring (Prometheus + Grafana)

2. **High Availability**
   - Use Docker Swarm or Kubernetes
   - Configure redundant instances
   - Implement health checks

### Security Considerations

- Store credentials in secure vault (not in .env)
- Use HTTPS for webhooks
- Enable authentication for metrics endpoint
- Regular security updates

## Development

### Project Structure

```
ai-video-analytics-system/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration management
│   ├── stream_processor.py     # RTSP stream handling
│   ├── detection_engine.py     # AI detection (YOLO)
│   ├── alert_manager.py        # Alert system
│   └── analytics_engine.py     # Analytics logic
├── tests/
│   ├── test_detection_engine.py
│   ├── test_alert_manager.py
│   └── test_analytics_engine.py
├── config/
│   ├── cameras.yaml            # Camera configuration
│   └── prometheus.yml          # Metrics configuration
├── models/                     # AI models (downloaded)
├── logs/                       # Application logs
├── alerts/                     # Alert logs and database
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image
├── docker-compose.yml          # Multi-service orchestration
└── README.md                   # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or contributions:
- GitHub Issues: [https://github.com/hamzaziizzz/ai-video-analytics-system/issues](https://github.com/hamzaziizzz/ai-video-analytics-system/issues)

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV for video processing
- PyTorch for deep learning
- NVIDIA for CUDA and TensorRT

---

**Built for production. Optimized for performance. Ready for scale.** 🚀