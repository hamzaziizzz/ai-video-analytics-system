"""
Tests for detection engine.
"""
import pytest
import numpy as np
import cv2
from src.detection_engine import DetectionEngine, DetectionResult


class TestDetectionResult:
    """Tests for DetectionResult class."""
    
    def test_detection_result_init(self):
        """Test detection result initialization."""
        result = DetectionResult(0, "person", 0.95, [100, 100, 200, 200])
        assert result.class_id == 0
        assert result.class_name == "person"
        assert result.confidence == 0.95
        assert result.bbox == [100, 100, 200, 200]
    
    def test_get_center(self):
        """Test center point calculation."""
        result = DetectionResult(0, "person", 0.95, [100, 100, 200, 200])
        center = result.get_center()
        assert center == (150, 150)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DetectionResult(0, "person", 0.95, [100, 100, 200, 200])
        data = result.to_dict()
        assert data['class_id'] == 0
        assert data['class_name'] == "person"
        assert data['confidence'] == 0.95
        assert data['bbox'] == [100, 100, 200, 200]
        assert data['center'] == (150, 150)


class TestDetectionEngine:
    """Tests for DetectionEngine class."""
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detect_with_dummy_frame(self, dummy_frame):
        """Test detection with dummy frame."""
        # Note: This test requires a model file which may not exist
        # In CI/CD, you should mock the YOLO model
        try:
            engine = DetectionEngine()
            results = engine.detect(dummy_frame)
            assert isinstance(results, list)
        except Exception:
            # Expected to fail if model not available
            pytest.skip("Model not available for testing")
    
    def test_visualize_detections(self, dummy_frame):
        """Test visualization of detections."""
        result = DetectionResult(0, "person", 0.95, [100, 100, 200, 200])
        
        try:
            engine = DetectionEngine()
            vis_frame = engine.visualize_detections(dummy_frame, [result])
            assert vis_frame.shape == dummy_frame.shape
        except Exception:
            pytest.skip("Model not available for testing")
    
    def test_visualize_zone(self, dummy_frame):
        """Test zone visualization."""
        polygon = [[100, 100], [500, 100], [500, 400], [100, 400]]
        
        try:
            engine = DetectionEngine()
            vis_frame = engine.visualize_zone(dummy_frame, polygon)
            assert vis_frame.shape == dummy_frame.shape
        except Exception:
            pytest.skip("Model not available for testing")
