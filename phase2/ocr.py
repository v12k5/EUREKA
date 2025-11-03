"""
Phase 2 - OPTIMIZED OCR Workflow with Spatial Context Fusion
Target: Process center frame of video with direction and distance
"""

import cv2
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from paddleocr import PaddleOCR
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastOCRProcessor:
    """Optimized OCR processor with spatial context"""
    
    def __init__(self, lang='en', confidence_threshold=0.75):
        """Initialize FAST OCR processor with mobile models"""
        logger.info("Initializing Fast PaddleOCR (mobile models)...")
        start_time = time.time()
        
        try:
            self.ocr = PaddleOCR(
                lang=lang,
                use_doc_preprocessor=False,
                use_doc_orientation_classify=False,
                use_textline_orientation=False,
            )
        except Exception as e:
            logger.warning(f"Using minimal OCR config due to: {e}")
            self.ocr = PaddleOCR(lang=lang)
        
        self.confidence_threshold = confidence_threshold
        init_time = time.time() - start_time
        logger.info(f"✓ Fast OCR initialized in {init_time:.2f}s")
    
    def process_frame_with_spatial_context(
        self, 
        frame: np.ndarray, 
        depth_map: Optional[np.ndarray] = None,
        resize_factor: float = 1.0
    ) -> List[Dict]:
        """
        Process frame with spatial context fusion
        
        Args:
            frame: Input frame
            depth_map: Optional depth map from Phase 1 (same size as frame)
            resize_factor: Resize frame before OCR
        
        Returns:
            List of text objects with spatial context
        """
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # Resize frame if needed
        if resize_factor != 1.0:
            new_w = int(w * resize_factor)
            new_h = int(h * resize_factor)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            logger.info(f"Resized frame: {w}x{h} -> {new_w}x{new_h}")
        else:
            frame_resized = frame
        
        # Run OCR
        try:
            results = self.ocr.predict(frame_resized)
        except AttributeError:
            results = self.ocr.ocr(frame_resized)
        
        ocr_time = time.time() - start_time
        logger.info(f"⚡ OCR processing time: {ocr_time:.2f}s")
        
        if results is None or len(results) == 0:
            logger.warning("No text detected")
            return []
        
        # Parse results with spatial context
        text_objects = []
        result = results[0] if isinstance(results, list) else results
        
        if hasattr(result, '__dict__') and 'rec_texts' in result.__dict__:
            rec_texts = result.rec_texts
            rec_scores = result.rec_scores
            rec_polys = result.rec_polys
            
            for text, confidence, bbox in zip(rec_texts, rec_scores, rec_polys):
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert bbox
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                
                # Scale bbox back if resized
                if resize_factor != 1.0:
                    scale = 1.0 / resize_factor
                    bbox = [[int(x * scale), int(y * scale)] for x, y in bbox]
                
                # Calculate center point
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_center = int(np.mean(x_coords))
                y_center = int(np.mean(y_coords))
                
                # Determine direction based on horizontal position
                if x_center < w / 3:
                    direction = "left"
                elif x_center > 2 * w / 3:
                    direction = "right"
                else:
                    direction = "center"
                
                # Get distance from depth map if available
                distance_m = None
                if depth_map is not None:
                    # Sample depth at text center (ensure within bounds)
                    y_sample = min(y_center, depth_map.shape[0] - 1)
                    x_sample = min(x_center, depth_map.shape[1] - 1)
                    depth_value = depth_map[y_sample, x_sample]
                    
                    # Convert depth to meters (this depends on your depth map format)
                    # Assuming depth map is normalized [0, 1] where 0=far, 1=near
                    # Adjust this formula based on your Phase 1 depth map format
                    if depth_value > 0:
                        distance_m = round(float(10.0 * (1.0 - depth_value)), 1)
                
                text_obj = {
                    "content": text,
                    "confidence": round(float(confidence), 2),
                    "direction": direction
                }
                
                # Only add distance if available
                if distance_m is not None:
                    text_obj["distance_m"] = distance_m
                
                text_objects.append(text_obj)
        
        logger.info(f"✓ Found {len(text_objects)} text objects (above {self.confidence_threshold} confidence)")
        return text_objects


class VideoFrameCapture:
    """Optimized video frame capture"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        logger.info(f"Video: {video_path} | {self.duration:.1f}s | {self.total_frames} frames @ {self.fps} fps")
    
    def get_center_frame(self) -> tuple:
        """Capture the center frame of the video"""
        center_frame_num = self.total_frames // 2
        center_time = center_frame_num / self.fps
        
        logger.info(f"📸 Seeking to center frame #{center_frame_num} (at {center_time:.2f}s)")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError("Cannot read center frame")
        
        return frame, center_frame_num, center_time
    
    def release(self):
        self.cap.release()


class DepthMapLoader:
    """Load depth map from Phase 1 (if available)"""
    
    @staticmethod
    def load_depth_map(depth_path: str) -> Optional[np.ndarray]:
        """
        Load depth map from Phase 1
        
        Args:
            depth_path: Path to depth map image
        
        Returns:
            Depth map as numpy array or None if not found
        """
        if not Path(depth_path).exists():
            logger.warning(f"Depth map not found at {depth_path}")
            return None
        
        try:
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth_map is not None:
                # Normalize to [0, 1]
                depth_map = depth_map.astype(np.float32) / 255.0
                logger.info(f"✓ Loaded depth map from {depth_path}")
                return depth_map
        except Exception as e:
            logger.warning(f"Failed to load depth map: {e}")
        
        return None


class OCRResultManager:
    """Result manager with visualization"""
    
    def __init__(self, output_dir='ocr_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: Dict, filename='ocr_output.json'):
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Results saved to {output_path}")
        return output_path
    
    def visualize_results(
        self, 
        frame: np.ndarray, 
        text_objects: List[Dict], 
        output_name='ocr_visualization.jpg'
    ):
        """Visualize OCR results with direction and distance labels"""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        for obj in text_objects:
            text = obj['content']
            confidence = obj['confidence']
            direction = obj['direction']
            distance = obj.get('distance_m', None)
            
            # Calculate approximate bbox for visualization
            # Since we don't store bbox in output, estimate it for visualization
            if direction == "left":
                x_center = w // 6
            elif direction == "right":
                x_center = 5 * w // 6
            else:
                x_center = w // 2
            
            y_center = h // 2
            
            # Create label with all info
            if distance is not None:
                label = f"{text} | {direction} | {distance}m"
            else:
                label = f"{text} | {direction}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Position based on direction
            if direction == "left":
                x = 10
            elif direction == "right":
                x = w - label_w - 10
            else:
                x = (w - label_w) // 2
            
            y = 30 + text_objects.index(obj) * 35
            
            # Draw background and text
            cv2.rectangle(vis_frame, (x - 5, y - label_h - 5), 
                         (x + label_w + 5, y + 5), (0, 255, 0), -1)
            cv2.putText(vis_frame, label, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw direction indicator
            if direction == "left":
                cv2.arrowedLine(vis_frame, (w//2, h//2), (x_center, y_center), 
                               (255, 0, 0), 3, tipLength=0.3)
            elif direction == "right":
                cv2.arrowedLine(vis_frame, (w//2, h//2), (x_center, y_center), 
                               (0, 0, 255), 3, tipLength=0.3)
            else:
                cv2.circle(vis_frame, (x_center, y_center), 10, (0, 255, 255), -1)
        
        # Draw center crosshair
        cv2.drawMarker(vis_frame, (w//2, h//2), (255, 255, 255), 
                      cv2.MARKER_CROSS, 20, 2)
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_frame)
        logger.info(f"✓ Visualization saved to {output_path}")
        return vis_frame


def main():
    """Main execution with spatial context fusion"""
    
    # ===================== CONFIGURATION =====================
    VIDEO_PATH = "videos/test.mp4"
    DEPTH_MAP_PATH = "depth_results/center_frame_depth.png"  # Optional from Phase 1
    CONFIDENCE_THRESHOLD = 0.75
    RESIZE_FACTOR = 0.8
    # =========================================================
    
    total_start = time.time()
    
    logger.info("=" * 70)
    logger.info("⚡ Phase 2 - OCR with Spatial Context Fusion")
    logger.info("=" * 70)
    
    try:
        # Initialize
        video_capture = VideoFrameCapture(VIDEO_PATH)
        ocr_processor = FastOCRProcessor(lang='en', confidence_threshold=CONFIDENCE_THRESHOLD)
        result_manager = OCRResultManager()
        
        # Load depth map if available (from Phase 1)
        depth_map = DepthMapLoader.load_depth_map(DEPTH_MAP_PATH)
        
        # Capture center frame
        logger.info("\n[Step 1] Capturing center frame...")
        capture_start = time.time()
        frame, frame_number, frame_time = video_capture.get_center_frame()
        capture_time = time.time() - capture_start
        logger.info(f"⚡ Frame capture time: {capture_time:.3f}s")
        logger.info(f"Frame #{frame_number} at {frame_time:.2f}s")
        
        # Resize depth map to match frame if needed
        if depth_map is not None:
            if depth_map.shape[:2] != frame.shape[:2]:
                depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
                logger.info(f"Resized depth map to match frame: {frame.shape[:2]}")
        
        # Process with OCR and spatial context
        logger.info(f"\n[Step 2] Processing with OCR + Spatial Context...")
        text_objects = ocr_processor.process_frame_with_spatial_context(
            frame, 
            depth_map=depth_map,
            resize_factor=RESIZE_FACTOR
        )
        
        # Generate structured output
        logger.info("\n[Step 3] Generating structured output...")
        timestamp = datetime.now().isoformat()
        
        ocr_result = {
            "timestamp": timestamp,
            "text_objects": text_objects
        }
        
        # Save results
        result_manager.save_results(ocr_result)
        
        # Visualization
        logger.info("\n[Step 4] Creating visualization...")
        vis_frame = result_manager.visualize_results(frame, text_objects)
        
        # Display the processed frame
        logger.info("\nDisplaying processed frame (press any key to close)...")
        cv2.imshow('OCR Results - Spatial Context', vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Summary
        total_time = time.time() - total_start
        
        logger.info("\n" + "=" * 70)
        logger.info("⚡ PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Frame: #{frame_number} at {frame_time:.2f}s")
        logger.info(f"Depth map: {'Loaded' if depth_map is not None else 'Not available'}")
        
        logger.info("\n" + "=" * 70)
        logger.info("DETECTED TEXT WITH SPATIAL CONTEXT")
        logger.info("=" * 70)
        
        if text_objects:
            for i, obj in enumerate(text_objects, 1):
                logger.info(f"\n  [{i}] '{obj['content']}'")
                logger.info(f"      Confidence: {obj['confidence']:.2%}")
                logger.info(f"      Direction: {obj['direction']}")
                if 'distance_m' in obj:
                    logger.info(f"      Distance: {obj['distance_m']}m")
        else:
            logger.info("\n  No text detected")
        
        logger.info("\n" + "=" * 70)
        logger.info("JSON OUTPUT PREVIEW:")
        logger.info("=" * 70)
        print(json.dumps(ocr_result, indent=2))
        logger.info("=" * 70)
        
        video_capture.release()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()