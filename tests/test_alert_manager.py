"""
Tests for alert manager.
"""
import pytest
import time
from src.alert_manager import AlertManager, Alert, AlertType, AlertSeverity


class TestAlert:
    """Tests for Alert class."""
    
    def test_alert_init(self):
        """Test alert initialization."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        assert alert.alert_type == AlertType.INTRUSION
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.camera_id == "camera_1"
        assert alert.message == "Test alert"
    
    def test_alert_to_dict(self):
        """Test alert to dictionary conversion."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert",
            {"key": "value"}
        )
        data = alert.to_dict()
        assert data['alert_type'] == 'intrusion'
        assert data['severity'] == 'critical'
        assert data['camera_id'] == 'camera_1'
        assert data['message'] == 'Test alert'
        assert data['metadata']['key'] == 'value'
    
    def test_alert_get_key(self):
        """Test alert key generation."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        key = alert.get_key()
        assert key == "camera_1_intrusion"


class TestAlertManager:
    """Tests for AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager for testing."""
        return AlertManager()
    
    def test_should_send_alert_first_time(self, alert_manager):
        """Test that first alert is sent."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        assert alert_manager.should_send_alert(alert) is True
    
    def test_should_suppress_alert_during_cooldown(self, alert_manager):
        """Test alert suppression during cooldown."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        
        # First alert should be sent
        assert alert_manager.should_send_alert(alert) is True
        
        # Second alert immediately should be suppressed
        assert alert_manager.should_send_alert(alert) is False
    
    def test_should_send_after_cooldown(self, alert_manager):
        """Test alert is sent after cooldown expires."""
        # Set very short cooldown for testing
        alert_manager.cooldown_seconds = 0.1
        
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        
        # First alert
        assert alert_manager.should_send_alert(alert) is True
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Should be sent again
        assert alert_manager.should_send_alert(alert) is True
    
    def test_different_cameras_independent_cooldown(self, alert_manager):
        """Test that different cameras have independent cooldowns."""
        alert1 = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert 1"
        )
        alert2 = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_2",
            "Test alert 2"
        )
        
        # Both should be sent
        assert alert_manager.should_send_alert(alert1) is True
        assert alert_manager.should_send_alert(alert2) is True
    
    def test_get_stats(self, alert_manager):
        """Test statistics retrieval."""
        stats = alert_manager.get_stats()
        assert 'alerts_sent' in stats
        assert 'alerts_suppressed' in stats
        assert 'active_alert_keys' in stats
    
    def test_reset_cooldown(self, alert_manager):
        """Test cooldown reset."""
        alert = Alert(
            AlertType.INTRUSION,
            AlertSeverity.CRITICAL,
            "camera_1",
            "Test alert"
        )
        
        # Send first alert
        alert_manager.should_send_alert(alert)
        
        # Reset cooldown
        alert_manager.reset_cooldown("camera_1", AlertType.INTRUSION)
        
        # Should be able to send again
        assert alert_manager.should_send_alert(alert) is True
