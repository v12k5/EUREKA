# Combined Pipeline: Phase 1 + Phase 2 Integration

## 📁 Project Structure

```
project_root/
├── input.mp4                      # Your input video
├── combine_pipeline.py            # Main integration script
├── phase1/
│   ├── main_pipeline.py
│   ├── yolo_tracker.py
│   ├── depth_estimator.py
│   ├── Weights/
│   │   └── best.onnx             # YOLO model
│   └── weights/
│       └── FastDepthV2_L1GN_Best.pth  # Depth model
├── phase2/
│   └── ocr.py                     # OCR processor
└── output/
    ├── output_video.mp4           # Processed video
    └── output.json                # Detection results
```

## 🚀 Installation

### 1. Install Dependencies

```bash
# Phase 1 dependencies
pip install opencv-python numpy torch ultralytics onnxruntime

# Phase 2 dependencies
pip install transformers pillow

# Optional: For GPU acceleration
pip install onnxruntime-gpu
```

### 2. Download Models

Ensure you have:
- **YOLO model**: `phase1/Weights/best.onnx`
- **Depth model** (if using): `phase1/weights/FastDepthV2_L1GN_Best.pth`
- **OCR model**: Auto-downloaded from HuggingFace on first run

## 📖 Usage Examples

### Example 1: YOLO + Depth + OCR
Process video with object detection, depth estimation, and OCR at 30 seconds:

```bash
python combine_pipeline.py \
  --mode both \
  --ocr yes \
  --ocr-time 30
```

### Example 2: YOLO Only + OCR
Process video with only object detection and OCR at 15 seconds:

```bash
python combine_pipeline.py \
  --mode yolo \
  --ocr yes \
  --ocr-time 15
```

### Example 3: YOLO + Depth (No OCR)
Process video with object detection and depth, without OCR:

```bash
python combine_pipeline.py \
  --mode both \
  --ocr no
```

### Example 4: Custom Input/Output
Specify custom input video and output paths:

```bash
python combine_pipeline.py \
  --mode both \
  --ocr yes \
  --ocr-time 25 \
  --source videos/my_video.mp4 \
  --output-video results/output.mp4 \
  --output-json results/detections.json
```

### Example 5: No Display (Faster Processing)
Run without real-time video display:

```bash
python combine_pipeline.py \
  --mode both \
  --ocr yes \
  --ocr-time 30 \
  --no-display
```

## ⚙️ Command Line Arguments

### Required Arguments

| Argument | Choices | Description |
|----------|---------|-------------|
| `--mode` | `yolo`, `both` | Phase 1 mode: tracking only or tracking + depth |
| `--ocr` | `yes`, `no` | Enable or disable OCR processing |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ocr-time` | None | Timestamp (seconds) to trigger OCR (required if `--ocr yes`) |
| `--source` | `input.mp4` | Path to input video |
| `--output-video` | `output/output_video.mp4` | Output video path |
| `--output-json` | `output/output.json` | Output JSON path |
| `--yolo-model` | `phase1/Weights/best.onnx` | YOLO model path |
| `--depth-model` | `phase1/weights/FastDepthV2_L1GN_Best.pth` | Depth model path |
| `--target-fps` | `10` | Processing FPS (lower = faster) |
| `--no-display` | False | Disable real-time video display |

## 📊 Output Format

### JSON Structure

The output JSON file contains both detection results and OCR results (if enabled):

```json
{
  "metadata": {
    "timestamp": "2025-11-04T10:30:45.123456",
    "original_fps": 30,
    "target_fps": 10,
    "total_frames": 900,
    "processed_frames": 300,
    "phase1_mode": "both",
    "ocr_enabled": true
  },
  "detections": [
    {
      "type": "detection",
      "frame_no": 150,
      "timestamp_sec": 5.0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100, 200, 300, 500],
      "track_id": 5,
      "distance_m": 2.5,
      "direction": "center"
    },
    {
      "type": "detection",
      "frame_no": 151,
      "timestamp_sec": 5.03,
      "class_name": "car",
      "confidence": 0.88,
      "bbox": [400, 150, 700, 400],
      "track_id": 12,
      "distance_m": 8.3,
      "direction": "right"
    },
    {
      "type": "ocr",
      "frame_no": 900,
      "timestamp_sec": 30.0,
      "text_objects": [
        {
          "content": "EXIT",
          "confidence": 1.0,
          "direction": "center"
        },
        {
          "content": "Emergency Exit Only",
          "confidence": 1.0,
          "direction": "center"
        }
      ],
      "full_text": "EXIT\nEmergency Exit Only",
      "processing_time": 2.34
    }
  ]
}
```

### Detection Types

1. **`type: "detection"`** - Object detection results from Phase 1
   - Includes: class, bbox, track_id, distance (if depth enabled), direction
   
2. **`type: "ocr"`** - Text recognition results from Phase 2
   - Includes: text_objects, full_text, processing_time

## 🔍 How OCR Works

When you specify `--ocr yes --ocr-time 30`:

1. **Frame Selection**: The pipeline captures 3 frames:
   - Previous frame (29.97s)
   - Target frame (30.00s)
   - Next frame (30.03s)

2. **Processing**: OCR runs on all 3 frames

3. **Best Result**: The frame with the most detected text is selected

4. **Output**: OCR result is added to JSON as a detection with `type: "ocr"`

## ⚡ Performance Tips

### Speed Optimization
```bash
# Lower FPS for faster processing
python combine_pipeline.py --mode yolo --ocr yes --ocr-time 30 --target-fps 5

# Disable display
python combine_pipeline.py --mode both --ocr yes --ocr-time 30 --no-display
```

### Quality Optimization
```bash
# Higher FPS for better tracking
python combine_pipeline.py --mode both --ocr yes --ocr-time 30 --target-fps 15
```

### GPU Acceleration
- YOLO: Automatically uses GPU if available
- Depth: Automatically uses GPU if available
- OCR: Automatically uses GPU if CUDA is available

## 🐛 Troubleshooting

### Issue: "Model not found"
```bash
# Check if model files exist
ls phase1/Weights/best.onnx
ls phase1/weights/FastDepthV2_L1GN_Best.pth
```

### Issue: "OCR takes too long"
- First run downloads the model (~1.5GB) - subsequent runs are faster
- OCR model is cached in `~/.cache/huggingface/`

### Issue: "CUDA out of memory"
```bash
# Use CPU for OCR
export CUDA_VISIBLE_DEVICES=""
python combine_pipeline.py --mode yolo --ocr yes --ocr-time 30
```

### Issue: "Video codec not supported"
- Output uses H264 (avc1) codec
- Fallback to mp4v if H264 unavailable
- Install: `pip install opencv-python-headless`

## 📝 Notes

1. **First OCR Run**: Takes 5-15 minutes to download model (one-time only)
2. **Subsequent Runs**: OCR loads in ~5-10 seconds from cache
3. **Processing Time**: Depends on video length and complexity
   - 1-minute video @ 10 FPS: ~2-3 minutes
   - OCR processing: ~1-3 seconds per frame set

## 🎯 Example Workflow

```bash
# 1. Place your video in project root
cp /path/to/video.mp4 input.mp4

# 2. Run combined pipeline
python combine_pipeline.py \
  --mode both \
  --ocr yes \
  --ocr-time 30 \
  --target-fps 10

# 3. Check results
ls output/
# output_video.mp4  (annotated video)
# output.json       (all detections + OCR)

# 4. View JSON
cat output/output.json | python -m json.tool
```

## 📚 Integration Details

### Phase 1 (Object Detection & Depth)
- **YOLO**: Object detection and tracking
- **ByteTrack**: Multi-object tracking
- **FastDepth**: Monocular depth estimation

### Phase 2 (OCR)
- **Model**: Donut-based medical prescription OCR
- **Specialization**: Optimized for medical text
- **Accuracy**: Works well on prescriptions, labels, signs

### Combined Output
- Unified JSON with both detection and OCR results
- Video with clean annotations (no OCR overlay)
- Frame-accurate timestamps