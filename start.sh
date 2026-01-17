#!/bin/bash
# Quick start script for AI Video Analytics System

set -e

echo "========================================="
echo "AI Video Analytics System - Quick Start"
echo "========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from example..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
fi

# Check if config/cameras.yaml exists
if [ ! -f config/cameras.yaml ]; then
    echo "ERROR: config/cameras.yaml not found!"
    echo "Please create camera configuration before starting"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs alerts models

# Check if running with Docker or Python
if command -v docker &> /dev/null; then
    echo ""
    echo "Docker detected. Choose deployment method:"
    echo "1) Docker Compose (recommended)"
    echo "2) Docker only"
    echo "3) Python (local)"
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1)
            echo "Starting with Docker Compose..."
            docker-compose up -d
            echo ""
            echo "Services started! Check logs with: docker-compose logs -f"
            echo "Metrics available at: http://localhost:8000/metrics"
            ;;
        2)
            echo "Building Docker image..."
            docker build -t ai-video-analytics:latest .
            echo "Starting container..."
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
            echo ""
            echo "Container started! Check logs with: docker logs -f ai-video-analytics"
            echo "Metrics available at: http://localhost:8000/metrics"
            ;;
        3)
            echo "Starting with Python..."
            # Check for virtual environment
            if [ ! -d "venv" ]; then
                echo "Creating virtual environment..."
                python3 -m venv venv
            fi
            
            echo "Activating virtual environment..."
            source venv/bin/activate
            
            echo "Installing dependencies..."
            pip install -q -r requirements.txt
            
            echo "Starting application..."
            python -m src.main
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
else
    echo "Docker not found. Starting with Python..."
    
    # Check for virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    
    echo "Starting application..."
    python -m src.main
fi
