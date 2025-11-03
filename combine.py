"""
Combined Pipeline: Phase 1 (YOLO+Depth) + Phase 2 (OCR)
========================================================
Integrates object detection, tracking, depth estimation, and OCR

Usage:
    python combine_pipeline.py --mode both --ocr yes --ocr-time 30
    python combine_pipeline.py --mode yolo --ocr no
"""

import os
import sys
import cv2
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

# Add phase directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase2'))

# Import Phase 1 modules
from yolo_tracker import YOLOTracker
from depth_estimator import DepthEstimator

# Import Phase 2 module
from ocr import MedicalOCRProcessor


class CombinedPipeline:
    """
    Combined pipeline integrating Phase 1 (YOLO+Depth) and Phase 2 (OCR)
    """
    
    def __init__(self, 
                 phase1_mode: str,
                 enable_ocr: bool,
                 ocr_timestamp: Optional[float] = None,
                 yolo_model_path: str = None,
                 depth_model_path: str = None,
                 target_fps: int = 2):
        """
        Initialize combined pipeline
        
        Args:
            phase1_mode: 'yolo' or 'both' (yolo+depth)
            enable_ocr: Whether to enable OCR processing
            ocr_timestamp: Timestamp in seconds when to trigger OCR
            yolo_model_path: Path to YOLO model
            depth_model_path: Path to depth model
            target_fps: Target FPS for processing
        """
        self.phase1_mode = phase1_mode
        self.enable_ocr = enable_ocr
        self.ocr_timestamp = ocr_timestamp
        self.target_fps = target_fps
        
        print("\n" + "=" * 60)
        print("INITIALIZING COMBINED PIPELINE")
        print("=" * 60)
        
        # Initialize Phase 1 - YOLO Tracker
        print("\n[Phase 1] Loading YOLO Tracker...")
        if yolo_model_path is None:
            raise ValueError("YOLO model path is required")
        self.tracker = YOLOTracker(model_path=yolo_model_path)
        print("✓ YOLO Tracker loaded")
        
        # Initialize Phase 1 - Depth Estimator (if needed)
        self.depth_estimator = None
        if phase1_mode == 'both':
            print("\n[Phase 1] Loading Depth Estimator...")
            if depth_model_path is None:
                raise ValueError("Depth model path required for 'both' mode")
            self.depth_estimator = DepthEstimator(model_path=depth_model_path)
            print("✓ Depth Estimator loaded")
        
        # Initialize Phase 2 - OCR Processor (if needed)
        self.ocr_processor = None
        if enable_ocr:
            print("\n[Phase 2] Loading OCR Processor...")
            self.ocr_processor = MedicalOCRProcessor()
            print("✓ OCR Processor loaded")
            if ocr_timestamp is not None:
                print(f"✓ OCR will trigger at {ocr_timestamp} seconds")
        
        print("\n" + "=" * 60)
        print(f"✓ Pipeline initialized: Phase1={phase1_mode.upper()}, OCR={enable_ocr}")
        print("=" * 60 + "\n")
    
    def process_video(self, 
                     video_path: str,
                     output_video_path: str,
                     output_json_path: str,
                     show_video: bool = True):
        """
        Process video through combined pipeline
        
        Args:
            video_path: Path to input video
            output_video_path: Path to save output video
            output_json_path: Path to save JSON output
            show_video: Whether to display video in real-time
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, original_fps // self.target_fps)
        
        # Calculate OCR frame number
        ocr_frame_no = None
        if self.enable_ocr and self.ocr_timestamp is not None:
            ocr_frame_no = int(self.ocr_timestamp * original_fps)
            print(f"OCR will process frames around frame #{ocr_frame_no} ({self.ocr_timestamp}s)")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, self.target_fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.target_fps, (width, height))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Original FPS: {original_fps}, Processing at: {self.target_fps} FPS")
        print(f"Resolution: {width}x{height}")
        print(f"Total frames: {total_frames}\n")
        
        frame_count = 0
        processed_count = 0
        all_detections = []
        ocr_frames_buffer = {}  # Store frames around OCR timestamp
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Store frames around OCR timestamp (previous, current, next)
            if self.enable_ocr and ocr_frame_no is not None:
                frame_diff = abs(frame_count - ocr_frame_no)
                if frame_diff <= 1:  # Previous, current, or next frame
                    ocr_frames_buffer[frame_count] = frame.copy()
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames... (Frame {frame_count}/{total_frames})")
            
            # Phase 1: Process with YOLO and optionally Depth
            if self.phase1_mode == 'yolo':
                annotated_frame, detections = self._process_yolo_only(frame)
            else:  # both
                annotated_frame, detections = self._process_yolo_and_depth(frame)
            
            # Add frame info to detections
            timestamp_sec = round(frame_count / original_fps, 2)
            for det in detections:
                det['frame_no'] = frame_count
                det['timestamp_sec'] = timestamp_sec
                det['type'] = 'detection'
                all_detections.append(det)
            
            # Write and display
            out.write(annotated_frame)
            if show_video:
                cv2.imshow('Combined Pipeline', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Phase 2: Process OCR if enabled
        if self.enable_ocr and ocr_frame_no is not None and ocr_frames_buffer:
            print("\n" + "=" * 60)
            print("PROCESSING OCR")
            print("=" * 60)
            ocr_result = self._process_ocr_frames(ocr_frames_buffer, ocr_frame_no, original_fps)
            if ocr_result:
                all_detections.append(ocr_result)
                print("✓ OCR processing complete")
        
        # Save JSON
        self._save_json(output_json_path, all_detections, original_fps, 
                       frame_count, processed_count)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✓ Video saved: {output_video_path}")
        print(f"✓ JSON saved: {output_json_path}")
        print(f"✓ Total detections: {len([d for d in all_detections if d['type'] == 'detection'])}")
        if self.enable_ocr:
            ocr_count = len([d for d in all_detections if d['type'] == 'ocr'])
            print(f"✓ OCR results: {ocr_count}")
        print("=" * 60 + "\n")
    
    def _process_yolo_only(self, frame):
        """Process frame with YOLO tracking only"""
        detections = self.tracker.detect_and_track(frame)
        annotated_frame = self.tracker.annotate_frame(frame, detections)
        return annotated_frame, detections
    
    def _process_yolo_and_depth(self, frame):
        """Process frame with YOLO tracking and depth estimation"""
        # Get YOLO detections
        detections = self.tracker.detect_and_track(frame)
        
        # Add depth information
        for det in detections:
            depth_info = self.depth_estimator.process_detection(frame, det['bbox'])
            det['distance_m'] = round(depth_info['distance'], 2) if depth_info['distance'] else None
            det['direction'] = depth_info['direction']
        
        # Annotate frame
        annotated_frame = self._annotate_with_depth(frame, detections)
        
        return annotated_frame, detections
    
    def _annotate_with_depth(self, frame, detections):
        """Annotate frame with detections including depth info"""
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            if 'distance_m' in det and det['distance_m']:
                label += f" | {det['distance_m']:.2f}m | {det['direction']}"
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw track ID
            if det['track_id'] is not None:
                id_text = f"ID: {det['track_id']}"
                cv2.putText(annotated_frame, id_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _process_ocr_frames(self, frames_buffer: Dict[int, any], 
                           target_frame_no: int, fps: int) -> Optional[Dict]:
        """
        Process OCR on frames around target timestamp
        
        Args:
            frames_buffer: Dictionary of frame_no -> frame
            target_frame_no: Target frame number
            fps: Video FPS
        
        Returns:
            Best OCR result as detection dict
        """
        print(f"\nProcessing OCR on frames: {sorted(frames_buffer.keys())}")
        
        ocr_results = []
        
        for frame_no in sorted(frames_buffer.keys()):
            frame = frames_buffer[frame_no]
            print(f"  Processing frame {frame_no}...")
            
            # Convert frame to format expected by OCR
            result = self.ocr_processor.extract_text_from_frame(frame)
            
            if result['success'] and result['text']:
                ocr_results.append({
                    'frame_no': frame_no,
                    'text': result['text'],
                    'processing_time': result['processing_time']
                })
                print(f"    ✓ Text found: {len(result['text'])} characters")
            else:
                print(f"    ✗ No text detected")
        
        # Select best result (longest text or most content)
        if not ocr_results:
            print("  ⚠ No OCR text detected in any frame")
            return None
        
        # Choose result with most text content
        best_result = max(ocr_results, key=lambda x: len(x['text']))
        print(f"\n  ✓ Best result from frame {best_result['frame_no']}")
        
        # Create OCR detection object
        text_lines = [line.strip() for line in best_result['text'].split('\n') if line.strip()]
        
        ocr_detection = {
            'type': 'ocr',
            'frame_no': best_result['frame_no'],
            'timestamp_sec': round(best_result['frame_no'] / fps, 2),
            'text_objects': [
                {
                    'content': line,
                    'confidence': 1.0,
                    'direction': 'center'
                }
                for line in text_lines
            ],
            'full_text': best_result['text'],
            'processing_time': best_result['processing_time']
        }
        
        return ocr_detection
    
    def _save_json(self, output_path: str, detections: List[Dict], 
                   original_fps: int, total_frames: int, processed_frames: int):
        """Save all detections to JSON"""
        json_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "original_fps": original_fps,
                "target_fps": self.target_fps,
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "phase1_mode": self.phase1_mode,
                "ocr_enabled": self.enable_ocr
            },
            "detections": detections
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Combined Pipeline: Phase 1 (YOLO+Depth) + Phase 2 (OCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO + Depth + OCR at 30 seconds
  python combine_pipeline.py --mode both --ocr yes --ocr-time 30
  
  # Only YOLO + OCR at 15 seconds
  python combine_pipeline.py --mode yolo --ocr yes --ocr-time 15
  
  # YOLO + Depth without OCR
  python combine_pipeline.py --mode both --ocr no
        """
    )
    
    # Phase 1 arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=['yolo', 'both'],
        required=True,
        help="Phase 1 mode: 'yolo' (tracking only) or 'both' (tracking + depth)"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="phase1/Weights/best.onnx",
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="phase1/weights/FastDepthV2_L1GN_Best.pth",
        help="Path to depth model"
    )
    
    # Phase 2 arguments
    parser.add_argument(
        "--ocr",
        type=str,
        choices=['yes', 'no'],
        required=True,
        help="Enable OCR processing"
    )
    parser.add_argument(
        "--ocr-time",
        type=float,
        default=None,
        help="Timestamp in seconds to trigger OCR (required if --ocr yes)"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--source",
        type=str,
        default="input.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="output/output_video.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/output.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=10,
        help="Target FPS for processing"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable real-time video display"
    )
    
    args = parser.parse_args()
    
    # Validate OCR arguments
    enable_ocr = args.ocr == 'yes'
    if enable_ocr and args.ocr_time is None:
        parser.error("--ocr-time is required when --ocr yes")
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(args.output_video) or 'output', exist_ok=True)
        
        # Initialize and run pipeline
        pipeline = CombinedPipeline(
            phase1_mode=args.mode,
            enable_ocr=enable_ocr,
            ocr_timestamp=args.ocr_time,
            yolo_model_path=args.yolo_model,
            depth_model_path=args.depth_model if args.mode == 'both' else None,
            target_fps=args.target_fps
        )
        
        pipeline.process_video(
            video_path=args.source,
            output_video_path=args.output_video,
            output_json_path=args.output_json,
            show_video=not args.no_display
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()