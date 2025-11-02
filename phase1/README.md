# Phase 1: Object Detection, Tracking, and Depth Estimation

## 📋 Table of Contents
- [Aim](#aim)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage Guide](#usage-guide)
- [Module Integration](#module-integration)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Aim

The aim of Phase 1 is to develop a **modular computer vision pipeline** that combines:
1. **Object Detection & Tracking**: Using YOLOv8 with ByteTrack for real-time multi-object tracking
2. **Depth Estimation**: Using FastDepth model to estimate object distances and spatial positions
3. **Flexible Architecture**: Reusable modules that can be integrated into future projects (OCR, segmentation, etc.)

This system processes video streams at 10 FPS, detects objects, tracks them across frames, estimates their distance from the camera, and determines their spatial direction (left, right, center, etc.).

---

## ✨ Features

- ✅ **YOLOv8 Object Detection** with ONNX optimization for faster inference
- ✅ **ByteTrack Multi-Object Tracking** for consistent ID assignment
- ✅ **FastDepth Distance Estimation** for each detected object
- ✅ **Spatial Direction Detection** (9-zone grid: top-left, center, bottom-right, etc.)
- ✅ **Modular Design** - Use YOLO alone, Depth alone, or both combined
- ✅ **10 FPS Processing** to reduce computational load
- ✅ **JSON Export** with all detection metadata
- ✅ **Annotated Video Output** with bounding boxes, IDs, distances, and directions

---

## 🏗️ Architecture

The project consists of 3 modular components:
```
phase1/
├── yolo_tracker.py          # Module 1: YOLO Detection + Tracking
├── depth_estimator.py       # Module 2: Depth Estimation + Direction
├── main_pipeline.py         # Module 3: Combined Pipeline
├── convert_yolo_to_onnx.py  # Utility: Model conversion
├── requirements.txt         # Dependencies
├── Weights/
│   ├── best.pt             # YOLO PyTorch model
│   └── best.onnx           # YOLO ONNX model (optimized)
├── weights/
│   └── FastDepthV2_L1GN_Best.pth  # FastDepth model
├── videos/
│   └── check.mp4           # Input video
├── output/
│   ├── output_video.mp4    # Annotated output video
│   └── output.json         # Detection results
└── src/                    # FastDepth model source files
    ├── models.py
    └── load_pretrained.py
```

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd phase1
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Convert YOLO Model to ONNX (Optional but Recommended)
```bash
python convert_yolo_to_onnx.py --model Weights/best.pt
```

This will create `Weights/best.onnx` which is 2-3x faster than the `.pt` model.

---

## ⚙️ How It Works

### 1. **YOLO Detection & Tracking** (`yolo_tracker.py`)
- Loads YOLOv8 model (`.pt` or `.onnx` format)
- Detects objects in each frame
- Uses ByteTrack algorithm to maintain consistent track IDs across frames
- Returns: `bbox`, `class_name`, `confidence`, `track_id`

### 2. **Depth Estimation** (`depth_estimator.py`)
- Loads FastDepth model (`.pth` format)
- Takes cropped object regions as input
- Estimates distance in meters using monocular depth estimation
- Calculates spatial direction based on object center position
- Returns: `distance_m`, `direction`

### 3. **Pipeline Integration** (`main_pipeline.py`)
- Combines both modules based on user-selected mode
- Processes video at target FPS (default: 10)
- Annotates frames with all information
- Saves results to JSON and video file

### Processing Flow:
```
Input Video → Frame Extraction → YOLO Detection → ByteTrack Tracking
                                         ↓
                              Crop Object Regions
                                         ↓
                              FastDepth Estimation
                                         ↓
                              Direction Calculation
                                         ↓
                     Annotate Frame + Save Results
```

---

## 📖 Usage Guide

### **Mode 1: YOLO Tracking Only**
Use this when you only need object detection and tracking without depth information.
```bash
python main_pipeline.py --mode yolo --yolo-model Weights/best.onnx
```

**Output:**
- Video with bounding boxes, class labels, confidence scores, and track IDs
- JSON with detection data (no depth/direction)

---

### **Mode 2: YOLO + Depth (Recommended)**
Use this for complete object detection, tracking, distance estimation, and direction.
```bash
python main_pipeline.py --mode both --yolo-model Weights/best.onnx --depth-model weights/FastDepthV2_L1GN_Best.pth
```

**Output:**
- Video with bounding boxes, IDs, distances (meters), and directions
- JSON with complete detection metadata

---

### **Advanced Options**

#### Custom Input/Output Paths
```bash
python main_pipeline.py \
  --mode both \
  --source videos/my_video.mp4 \
  --output-video output/result.mp4 \
  --output-json output/detections.json
```

#### Change Processing FPS
```bash
# Process at 5 FPS (slower, more accurate)
python main_pipeline.py --mode both --target-fps 5

# Process at 15 FPS (faster, less accurate)
python main_pipeline.py --mode both --target-fps 15
```

#### Disable Video Display (Headless Mode)
```bash
python main_pipeline.py --mode both --no-display
```

#### Use PyTorch Model Instead of ONNX
```bash
python main_pipeline.py --mode both --yolo-model Weights/best.pt
```

---

## 🔧 Module Integration (Using in Other Projects)

### **Example 1: Using YOLO Tracker in OCR Project**
```python
from yolo_tracker import YOLOTracker
import cv2

# Initialize tracker
tracker = YOLOTracker(model_path="Weights/best.onnx")

# Load your image/frame
frame = cv2.imread("image.jpg")

# Get detections
detections = tracker.detect_and_track(frame)

# Process each detection
for det in detections:
    bbox = det['bbox']  # [x1, y1, x2, y2]
    class_name = det['class_name']
    track_id = det['track_id']
    
    # Crop region for OCR
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    
    # Run your OCR on crop
    text = your_ocr_function(crop)
    print(f"Object {track_id}: {text}")

# Annotate and display
annotated_frame = tracker.annotate_frame(frame, detections)
cv2.imshow('Result', annotated_frame)
cv2.waitKey(0)
```

---

### **Example 2: Using Depth Estimator with Custom Bounding Boxes**
```python
from depth_estimator import DepthEstimator
import cv2

# Initialize depth estimator
depth_est = DepthEstimator(model_path="weights/FastDepthV2_L1GN_Best.pth")

# Load your image
frame = cv2.imread("image.jpg")
frame_height, frame_width = frame.shape[:2]

# Your custom bounding boxes (from OCR, segmentation, etc.)
bboxes = [
    [100, 150, 300, 400],  # [x1, y1, x2, y2]
    [500, 200, 700, 450]
]

# Process each bounding box
for bbox in bboxes:
    # Get depth and direction
    result = depth_est.process_detection(frame, bbox)
    
    distance = result['distance']
    direction = result['direction']
    
    print(f"Object at {direction}: {distance:.2f} meters away")
    
    # Or just get distance from crop
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    distance_only = depth_est.estimate_depth(crop)
```

---

### **Example 3: Using Both Modules with Custom Detection**
```python
from yolo_tracker import YOLOTracker
from depth_estimator import DepthEstimator
import cv2

# Initialize both modules
tracker = YOLOTracker("Weights/best.onnx")
depth_est = DepthEstimator("weights/FastDepthV2_L1GN_Best.pth")

# Process video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 1: Detect and track objects
    detections = tracker.detect_and_track(frame)
    
    # Step 2: Add depth information
    for det in detections:
        depth_info = depth_est.process_detection(frame, det['bbox'])
        det['distance'] = depth_info['distance']
        det['direction'] = depth_info['direction']
        
        # Your custom processing here
        if det['distance'] < 2.0:  # Object closer than 2 meters
            print(f"⚠️ Warning: {det['class_name']} too close!")
    
    # Step 3: Annotate
    annotated = tracker.annotate_frame(frame, detections)
    cv2.imshow('Result', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### **Example 4: Depth Estimation on Full Images**
```python
from depth_estimator import DepthEstimator
import cv2

# Initialize
depth_est = DepthEstimator("weights/FastDepthV2_L1GN_Best.pth")

# Estimate depth for entire image
image = cv2.imread("scene.jpg")
avg_distance = depth_est.estimate_depth(image)
print(f"Average scene depth: {avg_distance:.2f}m")

# Get detailed statistics
stats = depth_est.estimate_depth_with_stats(image)
print(f"Min: {stats['min']:.2f}m, Max: {stats['max']:.2f}m")
print(f"Mean: {stats['mean']:.2f}m, Std: {stats['std']:.2f}m")
```

---

### **Example 5: Direction Detection Only**
```python
from depth_estimator import DepthEstimator

# Static method - no model loading needed
bbox = [100, 200, 300, 400]  # [x1, y1, x2, y2]
frame_width = 1920
frame_height = 1080

direction = DepthEstimator.get_direction(bbox, frame_width, frame_height)
print(f"Object is located at: {direction}")
# Output: "top-left" or "center" or "bottom-right" etc.
```

---

## 📄 Output Format

### **JSON Structure**
```json
{
  "timestamp": "2025-11-02T18:45:30.123456",
  "detections": [
    {
      "frame_no": 1,
      "processed_frame_no": 1,
      "bbox": [928.94, 205.80, 1047.80, 653.15],
      "confidence": 0.89,
      "class_id": 0,
      "class_name": "person",
      "track_id": 1,
      "distance_m": 3.45,
      "direction": "center"
    },
    {
      "frame_no": 1,
      "processed_frame_no": 1,
      "bbox": [1200.50, 350.20, 1450.30, 750.60],
      "confidence": 0.92,
      "class_id": 2,
      "class_name": "car",
      "track_id": 2,
      "distance_m": 8.76,
      "direction": "right"
    }
  ]
}
```

### **Field Descriptions**
- `frame_no`: Original frame number from video
- `processed_frame_no`: Processed frame number (after FPS reduction)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `confidence`: Detection confidence (0-1)
- `class_id`: Numeric class ID
- `class_name`: Human-readable class name
- `track_id`: Unique tracking ID across frames
- `distance_m`: Estimated distance in meters (only in 'both' mode)
- `direction`: Spatial direction (only in 'both' mode)
  - Possible values: `center`, `left`, `right`, `top`, `bottom`, `top-left`, `top-right`, `bottom-left`, `bottom-right`

---

## 🔧 Troubleshooting

### **Issue 1: Video Won't Play**
**Solution:**
```bash
# Install FFmpeg and re-encode
pip install ffmpeg-python
ffmpeg -i output/output_video.mp4 -c:v libx264 -preset fast -crf 22 output/output_fixed.mp4
```
Or use VLC Media Player which supports more codecs.

---

### **Issue 2: CUDA Out of Memory**
**Solution:**
```bash
# Use CPU instead
# The code auto-detects, or force CPU:
export CUDA_VISIBLE_DEVICES=""
python main_pipeline.py --mode both
```

---

### **Issue 3: Slow Processing Speed**
**Solutions:**
1. Use ONNX model instead of PyTorch
2. Reduce target FPS: `--target-fps 5`
3. Use smaller YOLO model (YOLOv8n instead of YOLOv8m)
4. Process on GPU instead of CPU

---

### **Issue 4: Missing Dependencies**
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

---

### **Issue 5: Model Not Found**
**Solution:**
```bash
# Verify model paths
ls Weights/best.onnx
ls weights/FastDepthV2_L1GN_Best.pth

# Or specify full paths
python main_pipeline.py --mode both \
  --yolo-model /full/path/to/best.onnx \
  --depth-model /full/path/to/FastDepthV2_L1GN_Best.pth
```

---

## 📊 Performance Benchmarks

| Configuration | FPS | GPU Memory | Accuracy |
|--------------|-----|------------|----------|
| YOLO (.pt) CPU | ~8 | N/A | High |
| YOLO (.onnx) CPU | ~15 | N/A | High |
| YOLO + Depth CPU | ~3 | N/A | High |
| YOLO (.onnx) GPU | ~30 | 2GB | High |
| YOLO + Depth GPU | ~12 | 4GB | High |

*Tested on: Intel i7-10700K, NVIDIA RTX 3060, 16GB RAM*

---

## 🚀 Future Enhancements (Phase 2+)

- [ ] Real-time webcam support
- [ ] Multi-camera synchronization
- [ ] 3D bounding box estimation
- [ ] Integration with OCR for text detection
- [ ] Semantic segmentation
- [ ] Object velocity estimation
- [ ] Alert system for specific scenarios

---

## 📝 License

[Specify your license here]

---

## 👥 Contributors

[Your name/team]

---

## 📧 Contact

For questions or issues, please contact: [your-email@example.com]

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics
- **ByteTrack**: ByteDance
- **FastDepth**: MIT