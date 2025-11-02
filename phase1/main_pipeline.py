"""
Main Pipeline
=============
Combines YOLO tracking and depth estimation with user options.

Usage:
    # Only YOLO tracking
    python main_pipeline.py --mode yolo
    
    # Only depth estimation (on pre-detected objects from JSON)
    python main_pipeline.py --mode depth --detections-json output.json
    
    # Both YOLO + depth
    python main_pipeline.py --mode both
"""

import os
import cv2
import json
import argparse
from datetime import datetime
from typing import List, Dict

# Import our modules
from yolo_tracker import YOLOTracker
from depth_estimator import DepthEstimator


class Pipeline:
    """
    Main pipeline combining YOLO tracking and depth estimation.
    """
    
    def __init__(self, mode: str, yolo_model_path: str = None, 
                 depth_model_path: str = None, target_fps: int = 10):
        """
        Initialize pipeline.
        
        Args:
            mode: Pipeline mode ('yolo', 'depth', or 'both')
            yolo_model_path: Path to YOLO model
            depth_model_path: Path to FastDepth model
            target_fps: Target FPS for processing
        """
        self.mode = mode
        self.target_fps = target_fps
        
        # Initialize YOLO tracker if needed
        self.tracker = None
        if mode in ['yolo', 'both']:
            if yolo_model_path is None:
                raise ValueError("YOLO model path required for 'yolo' or 'both' mode")
            self.tracker = YOLOTracker(model_path=yolo_model_path)
        
        # Initialize depth estimator if needed
        self.depth_estimator = None
        if mode in ['depth', 'both']:
            if depth_model_path is None:
                raise ValueError("Depth model path required for 'depth' or 'both' mode")
            self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        
        print(f"\n✓ Pipeline initialized in '{mode}' mode")
    
    def process_video(self, video_path: str, output_video_path: str, 
                     output_json_path: str, show_video: bool = True):
        """
        Process video through the pipeline.
        
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
        frame_skip = max(1, original_fps // self.target_fps)
        
        # Setup video writer with H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
        out = cv2.VideoWriter(output_video_path, fourcc, self.target_fps, (width, height))
        
        # Fallback to mp4v if avc1 fails
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.target_fps, (width, height))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Original FPS: {original_fps}, Processing at: {self.target_fps} FPS")
        print(f"Mode: {self.mode.upper()}")
        print(f"Resolution: {width}x{height}\n")
        
        frame_count = 0
        processed_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")
            
            # Process based on mode
            if self.mode == 'yolo':
                annotated_frame, detections = self._process_yolo_only(frame)
            elif self.mode == 'both':
                annotated_frame, detections = self._process_yolo_and_depth(frame)
            else:
                # Depth mode requires pre-existing detections
                annotated_frame = frame
                detections = []
            
            # Add frame info to detections
            for det in detections:
                det['frame_no'] = frame_count
                det['processed_frame_no'] = processed_count
                all_detections.append(det)
            
            # Write and display
            out.write(annotated_frame)
            if show_video:
                cv2.imshow(f'Pipeline - {self.mode.upper()} Mode', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Save JSON
        self._save_json(output_json_path, all_detections, original_fps, 
                       frame_count, processed_count)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Video saved: {output_video_path}")
        print(f"✓ JSON saved: {output_json_path}")
        print(f"✓ Total detections: {len(all_detections)}")
    
    def _process_yolo_only(self, frame):
        """Process frame with YOLO tracking only."""
        detections = self.tracker.detect_and_track(frame)
        annotated_frame = self.tracker.annotate_frame(frame, detections)
        return annotated_frame, detections
    
    def _process_yolo_and_depth(self, frame):
        """Process frame with YOLO tracking and depth estimation."""
        # Get YOLO detections
        detections = self.tracker.detect_and_track(frame)
        
        # Add depth information to each detection
        for det in detections:
            depth_info = self.depth_estimator.process_detection(frame, det['bbox'])
            det['distance_m'] = round(depth_info['distance'], 2) if depth_info['distance'] else None
            det['direction'] = depth_info['direction']
        
        # Annotate frame with depth info
        annotated_frame = self._annotate_with_depth(frame, detections)
        
        return annotated_frame, detections
    
    def _annotate_with_depth(self, frame, detections):
        """Annotate frame with detections including depth info."""
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
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw track ID
            if det['track_id'] is not None:
                id_text = f"ID: {det['track_id']}"
                id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x2 - id_size[0], y2 + 5), 
                             (x2, y2 + 15 + id_size[1]), (0, 255, 0), -1)
                cv2.putText(annotated_frame, id_text, (x2 - id_size[0], y2 + 15 + id_size[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def _save_json(self, output_path: str, detections: List[Dict], 
                   original_fps: int, total_frames: int, processed_frames: int):
        """Save detections to JSON."""
        json_output = {
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Modular Pipeline for Object Detection, Tracking, and Depth Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Only YOLO tracking
  python main_pipeline.py --mode yolo --yolo-model Weights/best.onnx
  
  # YOLO + Depth estimation
  python main_pipeline.py --mode both --yolo-model Weights/best.onnx --depth-model weights/FastDepthV2_L1GN_Best.pth
  
  # Process at 5 FPS
  python main_pipeline.py --mode both --target-fps 5
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['yolo', 'depth', 'both'],
        required=True,
        help="Pipeline mode: 'yolo' (tracking only), 'depth' (depth only), 'both' (tracking + depth)"
    )
    parser.add_argument(
        "--yolo-model", 
        type=str, 
        default="Weights/best.onnx",
        help="Path to YOLO model (.pt, .onnx, .engine)"
    )
    parser.add_argument(
        "--depth-model", 
        type=str, 
        default="weights/FastDepthV2_L1GN_Best.pth",
        help="Path to FastDepth model (.pth)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default="videos/check.mp4",
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
        help="Target FPS for processing (default: 10)"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Disable real-time video display"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directories
        os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        
        # Initialize pipeline
        pipeline = Pipeline(
            mode=args.mode,
            yolo_model_path=args.yolo_model if args.mode in ['yolo', 'both'] else None,
            depth_model_path=args.depth_model if args.mode in ['depth', 'both'] else None,
            target_fps=args.target_fps
        )
        
        # Process video
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