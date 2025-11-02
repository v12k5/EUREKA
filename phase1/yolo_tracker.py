"""
YOLO Tracker Module
===================
Reusable module for object detection and tracking using YOLO + ByteTrack.

Usage:
    from yolo_tracker import YOLOTracker
    
    tracker = YOLOTracker(model_path="Weights/best.onnx")
    detections = tracker.detect_and_track(frame)
"""

import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import numpy as np


class YOLOTracker:
    """
    YOLO-based object detection and tracking class.
    """
    
    def __init__(self, model_path: str, device: str = None, conf_threshold: float = 0.25):
        """
        Initialize YOLO tracker.
        
        Args:
            model_path: Path to YOLO model (.pt, .onnx, .engine)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            conf_threshold: Confidence threshold for detections (0-1)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        if model_path.endswith('.pt'):
            self.model.to(self.device)
        
        print(f"✓ YOLO Tracker initialized on {self.device}")
    
    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[Dict]:
        """
        Perform detection and tracking on a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            persist: Whether to persist tracking across frames
        
        Returns:
            List of detections with format:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'track_id': int or None
                },
                ...
            ]
        """
        # Run YOLO tracking
        results = self.model.track(
            frame, 
            persist=persist, 
            verbose=False, 
            tracker="bytetrack.yaml",
            conf=self.conf_threshold
        )
        
        # Parse results
        detections = []
        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            
            if boxes is not None and len(boxes) > 0:
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
                
                for i, (xyxy, conf, cls_id) in enumerate(zip(xyxys, confs, cls_ids)):
                    detection = {
                        'bbox': [float(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.model.names[int(cls_id)],
                        'track_id': int(track_ids[i]) if track_ids is not None else None
                    }
                    detections.append(detection)
        
        return detections
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict], 
                       show_conf: bool = True, show_id: bool = True,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Annotate frame with detection boxes and labels.
        
        Args:
            frame: Input frame
            detections: List of detections from detect_and_track()
            show_conf: Whether to show confidence scores
            show_id: Whether to show track IDs
            color: BGR color for bounding boxes
        
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [det['class_name']]
            if show_conf:
                label_parts.append(f"{det['confidence']:.2f}")
            label = ' '.join(label_parts)
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw track ID if available
            if show_id and det['track_id'] is not None:
                id_text = f"ID: {det['track_id']}"
                id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x2 - id_size[0], y2 + 5), 
                             (x2, y2 + 15 + id_size[1]), color, -1)
                cv2.putText(annotated_frame, id_text, (x2 - id_size[0], y2 + 15 + id_size[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def get_class_names(self) -> Dict[int, str]:
        """Get dictionary of class IDs to names."""
        return self.model.names


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = YOLOTracker(model_path="Weights/best.onnx")
    
    # Process video
    cap = cv2.VideoCapture("videos/check.mp4")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and track
        detections = tracker.detect_and_track(frame)
        
        # Annotate frame
        annotated_frame = tracker.annotate_frame(frame, detections)
        
        # Display
        cv2.imshow('YOLO Tracking', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print detections every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(detections)} objects detected")
    
    cap.release()
    cv2.destroyAllWindows()