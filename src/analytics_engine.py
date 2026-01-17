"""
Analytics engine coordinating detection, tracking, and alerting.
Implements people counting, intrusion detection, and safety checks.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from .detection_engine import DetectionEngine, DetectionResult
from .alert_manager import AlertManager, Alert, AlertType, AlertSeverity


class ZoneAnalytics:
    """Analytics for a specific zone."""
    
    def __init__(self, zone_config: Dict):
        """
        Initialize zone analytics.
        
        Args:
            zone_config: Zone configuration from YAML
        """
        self.name = zone_config['name']
        self.zone_type = zone_config['type']
        self.polygon = zone_config['polygon']
        self.alert_on_entry = zone_config.get('alert_on_entry', True)
        self.required_ppe = zone_config.get('required_ppe', [])
        
        # State tracking
        self.violation_frames = 0
        self.violation_threshold = 3  # consecutive frames before alert
        
    def check_intrusion(self, detections: List[DetectionResult]) -> bool:
        """
        Check for intrusion in zone.
        
        Args:
            detections: List of detections in this zone
            
        Returns:
            True if intrusion detected
        """
        if self.zone_type != 'intrusion':
            return False
            
        # Check for person detections
        people_in_zone = [d for d in detections if d.class_name == 'person']
        return len(people_in_zone) > 0
    
    def check_safety(self, detections: List[DetectionResult]) -> Dict:
        """
        Check for safety violations in zone.
        
        Args:
            detections: List of detections in this zone
            
        Returns:
            Dict with violation status and details
        """
        if self.zone_type != 'safety':
            return {'violation': False}
            
        people = [d for d in detections if d.class_name == 'person']
        
        if not people:
            self.violation_frames = 0
            return {'violation': False}
        
        # Check PPE requirements
        violations = []
        for person in people:
            missing_ppe = []
            
            for ppe in self.required_ppe:
                # Check if PPE is detected near person (simplified)
                ppe_detected = any(d.class_name == ppe for d in detections)
                if not ppe_detected:
                    missing_ppe.append(ppe)
            
            if missing_ppe:
                violations.append({
                    'person_bbox': person.bbox,
                    'missing_ppe': missing_ppe
                })
        
        # Track consecutive violation frames
        if violations:
            self.violation_frames += 1
        else:
            self.violation_frames = 0
        
        # Only report violation after threshold
        is_violation = self.violation_frames >= self.violation_threshold
        
        return {
            'violation': is_violation,
            'violations': violations if is_violation else [],
            'frames': self.violation_frames
        }


class CameraAnalytics:
    """Analytics for a single camera."""
    
    def __init__(self, camera_config: Dict, detection_engine: DetectionEngine, 
                 alert_manager: AlertManager):
        """
        Initialize camera analytics.
        
        Args:
            camera_config: Camera configuration from YAML
            detection_engine: Detection engine instance
            alert_manager: Alert manager instance
        """
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.analytics_config = camera_config['analytics']
        self.detection_engine = detection_engine
        self.alert_manager = alert_manager
        
        # Initialize zones
        self.zones = []
        if 'zones' in camera_config:
            for zone_config in camera_config['zones']:
                self.zones.append(ZoneAnalytics(zone_config))
        
        # Analytics state
        self.current_people_count = 0
        self.max_people_count = 50  # from alert rules
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame and perform analytics.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict with analytics results
        """
        results = {
            'camera_id': self.camera_id,
            'people_count': 0,
            'intrusions': [],
            'safety_violations': [],
            'detections': []
        }
        
        # Get all detections
        all_detections = self.detection_engine.detect(frame)
        results['detections'] = [d.to_dict() for d in all_detections]
        
        # People counting
        if self.analytics_config.get('people_counting', False):
            people_count = sum(1 for d in all_detections if d.class_name == 'person')
            results['people_count'] = people_count
            self.current_people_count = people_count
            
            # Check for overcrowding
            if people_count > self.max_people_count:
                alert = Alert(
                    alert_type=AlertType.PEOPLE_COUNT,
                    severity=AlertSeverity.WARNING,
                    camera_id=self.camera_id,
                    message=f"High people count: {people_count} (max: {self.max_people_count})",
                    metadata={'count': people_count, 'max': self.max_people_count}
                )
                self.alert_manager.send_alert(alert)
        
        # Zone-based analytics
        for zone in self.zones:
            zone_detections = self._get_zone_detections(all_detections, zone.polygon)
            
            # Intrusion detection
            if self.analytics_config.get('intrusion_detection', False):
                if zone.check_intrusion(zone_detections):
                    results['intrusions'].append({
                        'zone': zone.name,
                        'count': len([d for d in zone_detections if d.class_name == 'person'])
                    })
                    
                    alert = Alert(
                        alert_type=AlertType.INTRUSION,
                        severity=AlertSeverity.CRITICAL,
                        camera_id=self.camera_id,
                        message=f"Intrusion detected in {zone.name}",
                        metadata={'zone': zone.name, 'detections': len(zone_detections)}
                    )
                    self.alert_manager.send_alert(alert)
            
            # Safety detection
            if self.analytics_config.get('safety_detection', False):
                safety_result = zone.check_safety(zone_detections)
                if safety_result['violation']:
                    results['safety_violations'].append({
                        'zone': zone.name,
                        'violations': safety_result['violations']
                    })
                    
                    alert = Alert(
                        alert_type=AlertType.SAFETY_VIOLATION,
                        severity=AlertSeverity.WARNING,
                        camera_id=self.camera_id,
                        message=f"Safety violation in {zone.name}",
                        metadata={'zone': zone.name, 'violations': safety_result['violations']}
                    )
                    self.alert_manager.send_alert(alert)
        
        return results
    
    def _get_zone_detections(self, detections: List[DetectionResult], 
                            polygon: List[List[int]]) -> List[DetectionResult]:
        """
        Filter detections to those within a zone.
        
        Args:
            detections: All detections
            polygon: Zone polygon
            
        Returns:
            Detections within zone
        """
        zone_detections = []
        poly = np.array(polygon, dtype=np.int32)
        
        for detection in detections:
            center = detection.get_center()
            if cv2.pointPolygonTest(poly, center, False) >= 0:
                zone_detections.append(detection)
                
        return zone_detections
    
    def visualize(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Visualize analytics results on frame.
        
        Args:
            frame: Input frame
            results: Analytics results
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw zones
        for zone in self.zones:
            color = (0, 255, 255)  # Yellow for normal
            if any(z['zone'] == zone.name for z in results.get('intrusions', [])):
                color = (0, 0, 255)  # Red for intrusion
            elif any(z['zone'] == zone.name for z in results.get('safety_violations', [])):
                color = (0, 165, 255)  # Orange for safety
                
            vis_frame = self.detection_engine.visualize_zone(vis_frame, zone.polygon, color)
        
        # Draw detections
        detections = [DetectionResult(d['class_id'], d['class_name'], d['confidence'], d['bbox']) 
                     for d in results.get('detections', [])]
        vis_frame = self.detection_engine.visualize_detections(vis_frame, detections)
        
        # Add info text
        info_y = 30
        cv2.putText(vis_frame, f"Camera: {self.camera_name}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.analytics_config.get('people_counting', False):
            info_y += 30
            cv2.putText(vis_frame, f"People: {results['people_count']}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
