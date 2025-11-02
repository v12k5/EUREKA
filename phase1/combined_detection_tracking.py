import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from collections import defaultdict

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import FastDepthV2
from load_pretrained import load_pretrained_fastdepth


def predict_depth(model, crop_image, device):
    """
    Performs depth estimation on a cropped image.
    
    Args:
        model: The FastDepth model.
        crop_image: A PIL Image of the cropped region.
        device: Device to run inference on ('cuda' or 'cpu').
    
    Returns:
        Estimated distance in meters.
    """
    output_size = (224, 224)
    
    resize = transforms.Resize(output_size)
    crop_image = resize(crop_image)
    
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(crop_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        model.eval()
        pred = model(input_tensor)
    
    depth_map = pred.squeeze().cpu().numpy()
    distance = float(np.mean(depth_map))  # Convert to Python float
    
    return distance


def get_direction(bbox, frame_width, frame_height):
    """
    Determines the direction of the object based on its bounding box position.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        frame_width: Width of the video frame
        frame_height: Height of the video frame
    
    Returns:
        Direction string (e.g., "center", "top-left", "right", etc.)
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    left_threshold = frame_width * 0.33
    right_threshold = frame_width * 0.67
    top_threshold = frame_height * 0.33
    bottom_threshold = frame_height * 0.67
    
    if center_x < left_threshold:
        horizontal = "left"
    elif center_x > right_threshold:
        horizontal = "right"
    else:
        horizontal = "center"
    
    if center_y < top_threshold:
        vertical = "top"
    elif center_y > bottom_threshold:
        vertical = "bottom"
    else:
        vertical = "center"
    
    if horizontal == "center" and vertical == "center":
        return "center"
    elif horizontal == "center":
        return vertical
    elif vertical == "center":
        return horizontal
    else:
        return f"{vertical}-{horizontal}"


def annotate_frame(frame, boxes, model, depth_model=None, enable_depth=False, device='cpu'):
    """
    Annotate the frame with detection boxes, labels, confidences, track IDs, and optionally distance.
    
    Args:
        frame: Input frame.
        boxes: YOLO boxes object.
        model: YOLO model for accessing class names.
        depth_model: FastDepth model (optional).
        enable_depth: Whether to calculate and display distance.
        device: Device to run depth model on.
    
    Returns:
        Annotated frame and list of detection data.
    """
    detections = []
    
    if boxes is None:
        return frame, detections
    
    xyxys = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
    
    names = model.names
    frame_height, frame_width = frame.shape[:2]
    
    for i, (xyxy, conf, cls_id) in enumerate(zip(xyxys, confs, cls_ids)):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = names[int(cls_id)]
        track_id = int(track_ids[i]) if track_ids is not None else None
        
        # Calculate distance and direction if enabled
        distance = None
        direction = None
        if enable_depth and depth_model is not None:
            # Crop and convert to PIL for depth estimation
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                distance = predict_depth(depth_model, pil_crop, device)
            
            direction = get_direction(xyxy, frame_width, frame_height)
        
        # Store detection data
        detection = {
            "id": track_id,
            "label": class_name,
            "bbox": [float(x) for x in xyxy.tolist()]  # Convert to Python float
        }
        if enable_depth:
            detection["direction"] = direction
            detection["distance_m"] = round(float(distance), 2) if distance is not None else None
        
        detections.append(detection)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        if enable_depth and distance is not None:
            label = f"{class_name} {conf:.2f} | {distance:.2f}m | {direction}"
        else:
            label = f"{class_name} {conf:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw track ID
        if track_id is not None:
            id_text = f"ID: {track_id}"
            id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x2 - id_size[0], y2 + 5), (x2, y2 + 15 + id_size[1]), (0, 255, 0), -1)
            cv2.putText(frame, id_text, (x2 - id_size[0], y2 + 15 + id_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame, detections


def process_video(yolo_model, depth_model, source, output_video_path, output_json_path, enable_depth, show_video, device):
    """
    Process video with YOLO tracking and optional depth estimation.
    
    Args:
        yolo_model: Loaded YOLO model.
        depth_model: Loaded FastDepth model (optional).
        source: Path to input video.
        output_video_path: Path to save output video.
        output_json_path: Path to save JSON output.
        enable_depth: Whether to enable depth estimation.
        show_video: Whether to display video in real-time.
        device: Device to run models on.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing video: {source}")
    print(f"Output video: {output_video_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Depth estimation: {'ENABLED' if enable_depth else 'DISABLED'}")
    print(f"Resolution: {width}x{height}, FPS: {fps}\n")
    
    frame_count = 0
    all_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")
        
        # Run YOLO tracking
        results = yolo_model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        # Annotate frame and get detection data
        res = results[0]
        annotated_frame, detections = annotate_frame(
            frame, 
            res.boxes, 
            yolo_model, 
            depth_model, 
            enable_depth,
            device
        )
        
        # Store detections with frame info
        for det in detections:
            det["frame_no"] = frame_count
            all_detections.append(det)
        
        # Write frame
        out.write(annotated_frame)
        
        # Display frame
        if show_video:
            cv2.imshow('Detection & Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Save JSON output
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "total_frames": frame_count,
        "detections": all_detections
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Video saved: {output_video_path}")
    print(f"✓ JSON saved: {output_json_path}")
    print(f"✓ Total frames: {frame_count}")
    print(f"✓ Total detections: {len(all_detections)}")


def main():
    parser = argparse.ArgumentParser(description="Combined YOLO Detection + Tracking with Optional Depth Estimation")
    parser.add_argument("--yolo-model", type=str, default="Weights/best.pt", help="Path to YOLO weights file")
    parser.add_argument("--depth-model", type=str, default="weights/FastDepthV2_L1GN_Best.pth", help="Path to FastDepth weights file")
    parser.add_argument("--source", type=str, default="videos/check.mp4", help="Path to input video")
    parser.add_argument("--output-video", type=str, default="output/output_video.mp4", help="Output video path")
    parser.add_argument("--output-json", type=str, default="output/output.json", help="Output JSON path")
    parser.add_argument("--enable-depth", action="store_true", help="Enable depth estimation and direction detection")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time video display")
    
    args = parser.parse_args()
    
    try:
        # Load YOLO model
        print("Loading YOLO model...")
        if not os.path.exists(args.yolo_model):
            raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")
        
        yolo_model = YOLO(args.yolo_model)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolo_model.to(device)
        print(f"✓ YOLO model loaded on {device}")
        
        # Load FastDepth model if depth estimation is enabled
        depth_model = None
        if args.enable_depth:
            print("Loading FastDepth model...")
            if not os.path.exists(args.depth_model):
                raise FileNotFoundError(f"FastDepth model not found: {args.depth_model}")
            
            depth_model = FastDepthV2()
            depth_model, _ = load_pretrained_fastdepth(depth_model, args.depth_model)
            if torch.cuda.is_available():
                depth_model = depth_model.cuda()
            print("✓ FastDepth model loaded")
        
        # Create output directories if they don't exist
        output_video_dir = os.path.dirname(args.output_video)
        output_json_dir = os.path.dirname(args.output_json)
        
        if output_video_dir and not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir, exist_ok=True)
        
        if output_json_dir and not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir, exist_ok=True)
        
        # Process video
        show_video = not args.no_display
        process_video(
            yolo_model,
            depth_model,
            args.source,
            args.output_video,
            args.output_json,
            args.enable_depth,
            show_video,
            device
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()