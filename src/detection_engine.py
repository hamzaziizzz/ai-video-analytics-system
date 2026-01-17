"""
AI Detection Engine using YOLO models.
Supports GPU acceleration and TensorRT optimization.
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from loguru import logger

from .config import settings


class DetectionResult:
    """Container for detection results."""
    
    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: List[int]):
        """
        Initialize detection result.
        
        Args:
            class_id: Class ID of detected object
            class_name: Class name of detected object
            confidence: Confidence score (0-1)
            bbox: Bounding box [x1, y1, x2, y2]
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.get_center()
        }


class DetectionEngine:
    """
    AI Detection Engine using YOLO.
    Supports people counting, object detection, and custom classes.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize detection engine.
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path or settings.model_path
        self.device = device or settings.device
        
        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        logger.info(f"Loading YOLO model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(self.model_path)
            
            # Export to TensorRT if enabled and on CUDA
            if settings.use_tensorrt and self.device == 'cuda':
                logger.info("Exporting model to TensorRT...")
                try:
                    self.model.export(format='engine', device=0, half=(settings.tensorrt_precision == 'fp16'))
                    logger.info("TensorRT export successful")
                except Exception as e:
                    logger.warning(f"TensorRT export failed: {e}, using PyTorch model")
                    
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        self.confidence_threshold = settings.confidence_threshold
        self.iou_threshold = settings.iou_threshold
        
        logger.info("Detection engine initialized")
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.
        
        Args:
            class_id: Class ID from model
            
        Returns:
            Class name string
        """
        try:
            if class_id < len(self.model.names):
                return self.model.names[class_id]
            else:
                logger.warning(f"Unknown class ID: {class_id}")
                return f"class_{class_id}"
        except Exception as e:
            logger.error(f"Error getting class name for ID {class_id}: {e}")
            return f"class_{class_id}"
        
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detection results
        """
        results = []
        
        try:
            # Resize frame if configured
            if settings.resize_width > 0 and settings.resize_height > 0:
                frame = cv2.resize(frame, (settings.resize_width, settings.resize_height))
            
            # Run inference
            predictions = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            for pred in predictions:
                boxes = pred.boxes
                for box in boxes:
                    # Extract box information
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    
                    # Get class name
                    class_name = self._get_class_name(class_id)
                    
                    result = DetectionResult(class_id, class_name, confidence, bbox)
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Detection error: {e}")
            
        return results
    
    def count_people(self, frame: np.ndarray) -> int:
        """
        Count people in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Number of people detected
        """
        detections = self.detect(frame)
        # Count detections with class 'person' (class_id 0 in COCO dataset)
        people_count = sum(1 for d in detections if d.class_name == 'person')
        return people_count
    
    def detect_in_zone(self, frame: np.ndarray, zone_polygon: List[List[int]]) -> List[DetectionResult]:
        """
        Detect objects within a specific zone.
        
        Args:
            frame: Input frame
            zone_polygon: List of points defining the zone [[x1,y1], [x2,y2], ...]
            
        Returns:
            List of detections within the zone
        """
        all_detections = self.detect(frame)
        zone_detections = []
        
        # Convert polygon to numpy array
        polygon = np.array(zone_polygon, dtype=np.int32)
        
        for detection in all_detections:
            center = detection.get_center()
            # Check if center point is inside polygon
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                zone_detections.append(detection)
                
        return zone_detections
    
    def visualize_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Draw detections on frame for visualization.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return vis_frame
    
    def visualize_zone(self, frame: np.ndarray, zone_polygon: List[List[int]], 
                       color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw zone polygon on frame.
        
        Args:
            frame: Input frame
            zone_polygon: Zone polygon points
            color: BGR color for zone
            
        Returns:
            Frame with zone visualization
        """
        vis_frame = frame.copy()
        polygon = np.array(zone_polygon, dtype=np.int32)
        
        # Draw filled polygon with transparency
        overlay = vis_frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
        
        # Draw polygon border
        cv2.polylines(vis_frame, [polygon], True, color, 2)
        
        return vis_frame
