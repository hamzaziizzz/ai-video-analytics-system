"""AI Video Analytics System - Production-grade CCTV analytics."""

__version__ = "1.0.0"
__author__ = "AI Video Analytics Team"

from .config import settings
from .detection_engine import DetectionEngine
from .stream_processor import StreamManager
from .alert_manager import AlertManager, Alert, AlertType, AlertSeverity
from .analytics_engine import CameraAnalytics

__all__ = [
    'settings',
    'DetectionEngine',
    'StreamManager',
    'AlertManager',
    'Alert',
    'AlertType',
    'AlertSeverity',
    'CameraAnalytics'
]
