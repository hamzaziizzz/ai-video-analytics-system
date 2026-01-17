"""
Alert management system with cooldown and deduplication.
Prevents alert spam and manages notification delivery.
"""
import time
import json
import requests
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum
from loguru import logger

from .config import settings


class AlertType(Enum):
    """Types of alerts."""
    PEOPLE_COUNT = "people_count"
    INTRUSION = "intrusion"
    SAFETY_VIOLATION = "safety_violation"
    SYSTEM_ERROR = "system_error"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """Alert data container."""
    
    def __init__(self, 
                 alert_type: AlertType,
                 severity: AlertSeverity,
                 camera_id: str,
                 message: str,
                 metadata: Optional[Dict] = None):
        """
        Initialize alert.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            camera_id: ID of camera that triggered alert
            message: Alert message
            metadata: Additional alert data
        """
        self.alert_type = alert_type
        self.severity = severity
        self.camera_id = camera_id
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'camera_id': self.camera_id,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_key(self) -> str:
        """Get unique key for alert deduplication."""
        return f"{self.camera_id}_{self.alert_type.value}"


class AlertManager:
    """
    Manages alerts with cooldown and deduplication.
    Prevents alert spam by enforcing cooldown periods.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_history: Dict[str, float] = {}  # key -> last_alert_time
        self.cooldown_seconds = settings.alert_cooldown_seconds
        self.alerts_sent = 0
        self.alerts_suppressed = 0
        
        logger.info(f"Alert manager initialized with {self.cooldown_seconds}s cooldown")
        
    def should_send_alert(self, alert: Alert) -> bool:
        """
        Check if alert should be sent based on cooldown.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be sent, False if suppressed
        """
        alert_key = alert.get_key()
        current_time = time.time()
        
        # Check if we've sent this alert recently
        if alert_key in self.alert_history:
            last_alert_time = self.alert_history[alert_key]
            time_since_last = current_time - last_alert_time
            
            if time_since_last < self.cooldown_seconds:
                self.alerts_suppressed += 1
                logger.debug(f"Alert suppressed (cooldown): {alert_key}, "
                           f"time since last: {time_since_last:.1f}s")
                return False
        
        # Update alert history
        self.alert_history[alert_key] = current_time
        return True
    
    def send_alert(self, alert: Alert):
        """
        Send alert if not in cooldown period.
        
        Args:
            alert: Alert to send
        """
        if not self.should_send_alert(alert):
            return
            
        logger.info(f"Sending alert: {alert.message}")
        self.alerts_sent += 1
        
        # Send via webhook
        if settings.enable_webhook and settings.webhook_url:
            self._send_webhook(alert)
        
        # Send via email
        if settings.enable_email:
            self._send_email(alert)
            
        # Log to file
        self._log_alert(alert)
    
    def _send_webhook(self, alert: Alert):
        """
        Send alert via webhook.
        
        Args:
            alert: Alert to send
        """
        try:
            payload = alert.to_dict()
            response = requests.post(
                settings.webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook sent successfully for {alert.get_key()}")
            else:
                logger.warning(f"Webhook failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook error: {e}")
    
    def _send_email(self, alert: Alert):
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = settings.email_from
            msg['To'] = settings.email_to
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.alert_type.value}"
            
            body = f"""
Alert Type: {alert.alert_type.value}
Severity: {alert.severity.value}
Camera: {alert.camera_id}
Time: {alert.timestamp.isoformat()}

Message: {alert.message}

Metadata: {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port) as server:
                server.starttls()
                # Authenticate if credentials are provided
                if settings.email_username and settings.email_password:
                    server.login(settings.email_username, settings.email_password)
                server.send_message(msg)
                
            logger.info(f"Email sent for {alert.get_key()}")
            
        except Exception as e:
            logger.error(f"Email error: {e}")
    
    def _log_alert(self, alert: Alert):
        """
        Log alert to file.
        
        Args:
            alert: Alert to log
        """
        try:
            with open('alerts/alert_log.json', 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def get_stats(self) -> Dict:
        """Get alert statistics."""
        return {
            'alerts_sent': self.alerts_sent,
            'alerts_suppressed': self.alerts_suppressed,
            'active_alert_keys': len(self.alert_history)
        }
    
    def reset_cooldown(self, camera_id: Optional[str] = None, 
                      alert_type: Optional[AlertType] = None):
        """
        Reset cooldown for specific alerts or all alerts.
        
        Args:
            camera_id: Optional camera ID to reset
            alert_type: Optional alert type to reset
        """
        if camera_id and alert_type:
            key = f"{camera_id}_{alert_type.value}"
            if key in self.alert_history:
                del self.alert_history[key]
                logger.info(f"Reset cooldown for {key}")
        elif camera_id:
            # Reset all alerts for camera
            keys_to_remove = [k for k in self.alert_history.keys() if k.startswith(camera_id)]
            for key in keys_to_remove:
                del self.alert_history[key]
            logger.info(f"Reset cooldown for camera {camera_id}")
        else:
            # Reset all
            self.alert_history.clear()
            logger.info("Reset all alert cooldowns")
