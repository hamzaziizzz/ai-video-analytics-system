"""
Main application entry point for AI Video Analytics System.
"""
import sys
import signal
import yaml
from pathlib import Path
from loguru import logger
from prometheus_client import start_http_server, Counter, Gauge

from .config import settings
from .stream_processor import StreamManager
from .detection_engine import DetectionEngine
from .alert_manager import AlertManager
from .analytics_engine import CameraAnalytics


# Prometheus metrics
FRAMES_PROCESSED = Counter('frames_processed_total', 'Total frames processed', ['camera_id'])
PEOPLE_COUNT = Gauge('people_count', 'Current people count', ['camera_id'])
ALERTS_SENT = Counter('alerts_sent_total', 'Total alerts sent', ['alert_type'])


class VideoAnalyticsSystem:
    """Main video analytics system."""
    
    def __init__(self, config_path: str = 'config/cameras.yaml'):
        """
        Initialize the video analytics system.
        
        Args:
            config_path: Path to camera configuration file
        """
        logger.info("Initializing AI Video Analytics System")
        
        # Load camera configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.detection_engine = DetectionEngine()
        self.alert_manager = AlertManager()
        self.stream_manager = StreamManager()
        
        # Initialize camera analytics
        self.camera_analytics = {}
        for camera_config in self.config['cameras']:
            if camera_config.get('enabled', True):
                analytics = CameraAnalytics(
                    camera_config, 
                    self.detection_engine,
                    self.alert_manager
                )
                self.camera_analytics[camera_config['id']] = analytics
                
                # Add stream with callback
                self.stream_manager.add_stream(
                    camera_config['id'],
                    camera_config['rtsp_url'],
                    self._frame_callback
                )
        
        logger.info(f"Initialized {len(self.camera_analytics)} cameras")
        
        # Start metrics server if enabled
        if settings.enable_metrics:
            start_http_server(settings.metrics_port)
            logger.info(f"Metrics server started on port {settings.metrics_port}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to load config file: {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config YAML: {e}")
            raise
    
    def _frame_callback(self, camera_id: str, frame):
        """
        Callback for processing frames from streams.
        
        Args:
            camera_id: ID of camera
            frame: Video frame
        """
        try:
            # Get analytics for this camera
            analytics = self.camera_analytics.get(camera_id)
            if not analytics:
                return
            
            # Process frame
            results = analytics.process_frame(frame)
            
            # Update metrics
            FRAMES_PROCESSED.labels(camera_id=camera_id).inc()
            if 'people_count' in results:
                PEOPLE_COUNT.labels(camera_id=camera_id).set(results['people_count'])
            
            # Log results periodically
            logger.debug(f"Camera {camera_id}: {results['people_count']} people, "
                        f"{len(results.get('intrusions', []))} intrusions, "
                        f"{len(results.get('safety_violations', []))} safety violations")
                        
        except (cv2.error, ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Error processing frame from {camera_id}: {e}")
    
    def start(self):
        """Start the video analytics system."""
        logger.info("Starting video analytics system")
        
        try:
            # Start all streams
            self.stream_manager.start_all()
            
            logger.info("System running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Wait indefinitely
            signal.pause()
            
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown signal received")
            self.stop()
        except (RuntimeError, OSError) as e:
            logger.error(f"System error: {e}")
            self.stop()
    
    def stop(self):
        """Stop the video analytics system."""
        logger.info("Stopping video analytics system")
        
        # Stop all streams
        self.stream_manager.stop_all()
        
        # Log final statistics
        stats = self.alert_manager.get_stats()
        logger.info(f"Alert statistics: {stats}")
        
        logger.info("System stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point."""
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level
    )
    logger.add(
        "logs/video_analytics_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level=settings.log_level
    )
    
    logger.info("="*60)
    logger.info(f"{settings.app_name}")
    logger.info("="*60)
    logger.info(f"Device: {settings.device}")
    logger.info(f"Model: {settings.model_path}")
    logger.info(f"Confidence threshold: {settings.confidence_threshold}")
    logger.info(f"Alert cooldown: {settings.alert_cooldown_seconds}s")
    logger.info("="*60)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("alerts").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Initialize and start system
    system = VideoAnalyticsSystem()
    system.start()


if __name__ == '__main__':
    main()
