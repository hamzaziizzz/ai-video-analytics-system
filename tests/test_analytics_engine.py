"""
Tests for analytics engine.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.analytics_engine import ZoneAnalytics, CameraAnalytics
from src.detection_engine import DetectionResult


class TestZoneAnalytics:
    """Tests for ZoneAnalytics class."""
    
    @pytest.fixture
    def intrusion_zone_config(self):
        """Create intrusion zone config."""
        return {
            'name': 'Restricted Area',
            'type': 'intrusion',
            'polygon': [[100, 100], [500, 100], [500, 400], [100, 400]],
            'alert_on_entry': True
        }
    
    @pytest.fixture
    def safety_zone_config(self):
        """Create safety zone config."""
        return {
            'name': 'Forklift Zone',
            'type': 'safety',
            'polygon': [[50, 50], [600, 50], [600, 450], [50, 450]],
            'required_ppe': ['helmet', 'vest']
        }
    
    def test_zone_init(self, intrusion_zone_config):
        """Test zone initialization."""
        zone = ZoneAnalytics(intrusion_zone_config)
        assert zone.name == 'Restricted Area'
        assert zone.zone_type == 'intrusion'
        assert len(zone.polygon) == 4
    
    def test_check_intrusion_with_person(self, intrusion_zone_config):
        """Test intrusion detection with person."""
        zone = ZoneAnalytics(intrusion_zone_config)
        
        # Create person detection
        detections = [DetectionResult(0, 'person', 0.9, [200, 200, 300, 300])]
        
        assert zone.check_intrusion(detections) is True
    
    def test_check_intrusion_without_person(self, intrusion_zone_config):
        """Test intrusion detection without person."""
        zone = ZoneAnalytics(intrusion_zone_config)
        
        # No detections
        assert zone.check_intrusion([]) is False
    
    def test_check_safety_no_people(self, safety_zone_config):
        """Test safety check with no people."""
        zone = ZoneAnalytics(safety_zone_config)
        result = zone.check_safety([])
        assert result['violation'] is False
    
    def test_check_safety_with_violations(self, safety_zone_config):
        """Test safety check with PPE violations."""
        zone = ZoneAnalytics(safety_zone_config)
        
        # Person without PPE
        detections = [DetectionResult(0, 'person', 0.9, [200, 200, 300, 300])]
        
        # Need multiple frames to trigger violation
        for _ in range(4):
            result = zone.check_safety(detections)
        
        assert result['violation'] is True


class TestCameraAnalytics:
    """Tests for CameraAnalytics class."""
    
    @pytest.fixture
    def camera_config(self):
        """Create camera config."""
        return {
            'id': 'camera_1',
            'name': 'Test Camera',
            'analytics': {
                'people_counting': True,
                'intrusion_detection': True,
                'safety_detection': False
            },
            'zones': [
                {
                    'name': 'Zone 1',
                    'type': 'intrusion',
                    'polygon': [[100, 100], [500, 100], [500, 400], [100, 400]],
                    'alert_on_entry': True
                }
            ]
        }
    
    @pytest.fixture
    def mock_detection_engine(self):
        """Create mock detection engine."""
        engine = Mock()
        engine.detect = Mock(return_value=[])
        engine.visualize_detections = Mock()
        engine.visualize_zone = Mock()
        return engine
    
    @pytest.fixture
    def mock_alert_manager(self):
        """Create mock alert manager."""
        return Mock()
    
    def test_camera_analytics_init(self, camera_config, mock_detection_engine, mock_alert_manager):
        """Test camera analytics initialization."""
        analytics = CameraAnalytics(camera_config, mock_detection_engine, mock_alert_manager)
        assert analytics.camera_id == 'camera_1'
        assert analytics.camera_name == 'Test Camera'
        assert len(analytics.zones) == 1
    
    def test_process_frame_empty(self, camera_config, mock_detection_engine, mock_alert_manager):
        """Test processing frame with no detections."""
        analytics = CameraAnalytics(camera_config, mock_detection_engine, mock_alert_manager)
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = analytics.process_frame(frame)
        
        assert results['camera_id'] == 'camera_1'
        assert results['people_count'] == 0
        assert len(results['intrusions']) == 0
    
    def test_process_frame_with_people(self, camera_config, mock_detection_engine, mock_alert_manager):
        """Test processing frame with people."""
        # Mock detection to return people
        mock_detection_engine.detect = Mock(return_value=[
            DetectionResult(0, 'person', 0.9, [100, 100, 200, 200]),
            DetectionResult(0, 'person', 0.85, [300, 300, 400, 400])
        ])
        
        analytics = CameraAnalytics(camera_config, mock_detection_engine, mock_alert_manager)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = analytics.process_frame(frame)
        
        assert results['people_count'] == 2
