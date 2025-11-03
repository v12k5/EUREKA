"""
Depth Estimator Module
======================
Reusable module for depth estimation using FastDepth.

Usage:
    from depth_estimator import DepthEstimator
    
    estimator = DepthEstimator(model_path="weights/FastDepthV2_L1GN_Best.pth")
    distance = estimator.estimate_depth(cropped_image)
    direction = estimator.get_direction(bbox, frame_width, frame_height)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Union

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import FastDepthV2
from load_pretrained import load_pretrained_fastdepth


class DepthEstimator:
    """
    Depth estimation class using FastDepth model.
    """
    
    def __init__(self, model_path: str, device: str = None, input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize depth estimator.
        
        Args:
            model_path: Path to FastDepth model weights (.pth)
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            input_size: Input size for depth model (height, width)
        """
        self.model_path = model_path
        self.input_size = input_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = FastDepthV2()
        self.model, _ = load_pretrained_fastdepth(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.resize = transforms.Resize(input_size)
        self.to_tensor = transforms.ToTensor()
        
        print(f"✓ Depth Estimator initialized on {self.device}")
    
    def estimate_depth(self, image: Union[np.ndarray, Image.Image]) -> float:
        """
        Estimate depth/distance for an image or crop.
        
        Args:
            image: Input image as numpy array (BGR/RGB) or PIL Image
        
        Returns:
            Estimated distance in meters (float)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if it's from OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])
            else:
                image = Image.fromarray(image)
        
        # Preprocess
        image = self.resize(image)
        input_tensor = self.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model(input_tensor)
        
        # Post-process
        depth_map = pred.squeeze().cpu().numpy()
        distance = float(np.mean(depth_map))
        
        return distance
    
    def estimate_depth_with_stats(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """
        Estimate depth with additional statistics.
        
        Args:
            image: Input image as numpy array (BGR/RGB) or PIL Image
        
        Returns:
            Dictionary with depth statistics:
            {
                'mean': float,
                'median': float,
                'min': float,
                'max': float,
                'std': float
            }
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])
            else:
                image = Image.fromarray(image)
        
        # Preprocess
        image = self.resize(image)
        input_tensor = self.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred = self.model(input_tensor)
        
        # Post-process
        depth_map = pred.squeeze().cpu().numpy()
        
        return {
            'mean': float(np.mean(depth_map)),
            'median': float(np.median(depth_map)),
            'min': float(np.min(depth_map)),
            'max': float(np.max(depth_map)),
            'std': float(np.std(depth_map))
        }
    
    @staticmethod
    def get_direction(bbox: list, frame_width: int, frame_height: int) -> str:
        """
        Determine object direction based on bounding box position.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_width: Width of the frame
            frame_height: Height of the frame
        
        Returns:
            Direction string (e.g., "center", "top-left", "right")
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Define thresholds (3x3 grid)
        left_threshold = frame_width * 0.33
        right_threshold = frame_width * 0.67
        top_threshold = frame_height * 0.33
        bottom_threshold = frame_height * 0.67
        
        # Determine horizontal position
        if center_x < left_threshold:
            horizontal = "left"
        elif center_x > right_threshold:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Determine vertical position
        if center_y < top_threshold:
            vertical = "top"
        elif center_y > bottom_threshold:
            vertical = "bottom"
        else:
            vertical = "center"
        
        # Combine directions
        if horizontal == "center" and vertical == "center":
            return "center"
        elif horizontal == "center":
            return vertical
        elif vertical == "center":
            return horizontal
        else:
            return f"{vertical}-{horizontal}"
    
    def process_detection(self, frame: np.ndarray, bbox: list) -> dict:
        """
        Process a detection to get depth and direction.
        
        Args:
            frame: Full frame (BGR format from OpenCV)
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Dictionary with depth and direction:
            {
                'distance': float,
                'direction': str
            }
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop the region
        crop = frame[y1:y2, x1:x2]
        
        # Estimate depth
        distance = None
        if crop.size > 0:
            distance = self.estimate_depth(crop)
        
        # Get direction
        frame_height, frame_width = frame.shape[:2]
        direction = self.get_direction(bbox, frame_width, frame_height)
        
        return {
            'distance': distance,
            'direction': direction
        }


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize estimator
    estimator = DepthEstimator(model_path="Weights/FastDepthV2_L1GN_Best.pth")
    
    # Load test image
    frame = cv2.imread("test_image.jpg")
    
    # Example bounding box (x1, y1, x2, y2)
    bbox = [100, 100, 300, 400]
    
    # Get depth and direction
    result = estimator.process_detection(frame, bbox)
    
    print(f"Distance: {result['distance']:.2f} meters")
    print(f"Direction: {result['direction']}")
    
    # Or estimate depth from full frame
    distance = estimator.estimate_depth(frame)
    print(f"Full frame depth: {distance:.2f} meters")
    
    # Get detailed statistics
    stats = estimator.estimate_depth_with_stats(frame)
    print(f"Depth statistics: {stats}")