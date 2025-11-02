import os
import argparse
import cv2
import json  # For JSON output
from pathlib import Path
from ultralytics import YOLO
import torch  # For device detection

def load_model(model_path: str) -> YOLO:
    """
    Load the custom YOLO model from the specified weights file.
    
    Args:
        model_path (str): Path to the .pt weights file.
    
    Returns:
        YOLO: Initialized YOLO model instance.
    
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    # Auto-set device (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: {device}")
    return model

def annotate_frame(frame: cv2.Mat, results, model: YOLO) -> cv2.Mat:
    """
    Annotate the frame with detection boxes, labels, confidences, and track IDs.
    
    Args:
        frame (cv2.Mat): Input frame.
        results: YOLO results object (list of Results).
        model (YOLO): Model for accessing class names.
    
    Returns:
        cv2.Mat: Annotated frame.
    """
    if len(results) == 0:
        return frame
    
    res = results[0]  # Single-frame result
    boxes = res.boxes
    if boxes is None:
        return frame
    
    # Extract data
    xyxys = boxes.xyxy.cpu().numpy()  # (n, 4) array
    confs = boxes.conf.cpu().numpy()  # (n,) array
    cls_ids = boxes.cls.cpu().numpy()  # (n,) array
    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None  # (n,) array
    
    names = model.names  # Dict of class names
    
    for i, (xyxy, conf, cls_id) in enumerate(zip(xyxys, confs, cls_ids)):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = names[int(cls_id)]
        label = f"{class_name} {conf:.2f}"
        
        # Draw bounding box (green by default; can customize per class)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw track ID if tracking is active
        if track_ids is not None:
            track_id = int(track_ids[i])
            id_text = f"ID: {track_id}"
            id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x2 - id_size[0], y2 + 20), (x2, y2 + 20 + id_size[1] + 10), (0, 255, 0), -1)
            cv2.putText(frame, id_text, (x2 - id_size[0], y2 + 20 + id_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def process_video(model: YOLO, source: str, output_path: str, json_path: str, show_video: bool = True) -> None:
    """
    Process a video source with YOLO detection + ByteTrack, annotate frames, save video, and export JSON.
    
    Args:
        model (YOLO): Loaded model.
        source (str): Path to input video or '0' for webcam.
        output_path (str): Path to save output video.
        json_path (str): Path to save JSON detections.
        show_video (bool): Whether to display frames in real-time.
    """
    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {source}")
    print(f"Output video: {output_path} (FPS: {fps}, Resolution: {width}x{height})")
    print(f"JSON output: {json_path}")
    
    frame_count = 0
    detections_list = []  # List to collect all detections for JSON
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"Processing frame {frame_count}...")
        
        # Optional: Resize frame for faster processing (uncomment if needed)
        # scale = 0.5
        # frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Run inference with ByteTrack (persist=True for stateful tracking)
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        # Extract detections for JSON
        res = results[0]
        boxes = res.boxes
        if boxes is not None:
            cls = boxes.cls.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            xyxy = boxes.xyxy.cpu().numpy()
            for i in range(len(cls)):
                det = {
                    "frame_no": frame_count,
                    "class": model.names[int(cls[i])],
                    "track_id": int(ids[i]) if ids is not None else None,
                    "bbox": xyxy[i].tolist()
                }
                detections_list.append(det)
        
        # Annotate frame
        annotated_frame = annotate_frame(frame, results, model)
        
        # Write to output (resize back if scaled)
        # if scale != 1.0:
        #     annotated_frame = cv2.resize(annotated_frame, (width, height))
        out.write(annotated_frame)
        
        # Optional real-time display
        if show_video:
            cv2.imshow('YOLO + ByteTrack Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
                break
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(detections_list, f, indent=4)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video saved to: {output_path}")
    print(f"JSON saved to: {json_path} (Total detections: {len(detections_list)})")

def main():
    """
    Main entry point. Parse args and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="YOLO Detection + ByteTrack Tracking")
    parser.add_argument("--model", type=str, default="Weights/best.pt", help="Path to YOLO weights file")
    parser.add_argument("--source", type=str, default="videos/check.mp4", help="Video path or '0' for webcam")
    parser.add_argument("--output", type=str, default="output/output_video.mp4", help="Output video path")
    parser.add_argument("--json-output", type=str, default="output/output.json", help="Output JSON path")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time video display")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Process video
        show_video = not args.no_display
        process_video(model, args.source, args.output, args.json_output, show_video)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()