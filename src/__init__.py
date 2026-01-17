"""AI Video Analytics System - Production-grade CCTV analytics."""

__version__ = "1.0.0"
__author__ = "AI Video Analytics Team"

# Lazy imports to avoid dependency issues during testing
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


def __getattr__(name):
    """Lazy import of modules."""
    if name == 'settings':
        from .config import settings
        return settings
    elif name == 'DetectionEngine':
        from .detection_engine import DetectionEngine
        return DetectionEngine
    elif name == 'StreamManager':
        from .stream_processor import StreamManager
        return StreamManager
    elif name in ('AlertManager', 'Alert', 'AlertType', 'AlertSeverity'):
        from .alert_manager import AlertManager, Alert, AlertType, AlertSeverity
        if name == 'AlertManager':
            return AlertManager
        elif name == 'Alert':
            return Alert
        elif name == 'AlertType':
            return AlertType
        elif name == 'AlertSeverity':
            return AlertSeverity
    elif name == 'CameraAnalytics':
        from .analytics_engine import CameraAnalytics
        return CameraAnalytics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
