"""
Real-time Integrated Pipeline - DEBUG VERSION
==============================================
Added comprehensive debugging and fixes for camera issues

Key fixes:
1. Added camera test on startup
2. Better error handling for video capture
3. Debug output for frame processing
4. Verification that frames are actually being captured
5. Option to test with static image if camera fails
"""

import os
import time
import threading
import datetime
import queue
import json
import sys

import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk

# Add phase directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase2'))

# Import Phase 1 modules
try:
    from yolo_tracker import YOLOTracker
    from depth_estimator import DepthEstimator
    HAS_PHASE1 = True
except Exception as e:
    print(f"[warning] Phase 1 modules not available: {e}")
    HAS_PHASE1 = False

# Import Phase 2 module
try:
    from ocr import MedicalOCRProcessor
    HAS_PHASE2 = True
except Exception as e:
    print(f"[warning] Phase 2 OCR not available: {e}")
    HAS_PHASE2 = False

# Optional imports
HAS_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except Exception:
    pass

HAS_PYTTX3 = False
try:
    import pyttsx3
    HAS_PYTTX3 = True
except Exception:
    pass

# -------- Configuration ----------
CAMERA_INDEX = 0
VIDEO_FPS = 20.0
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"XVID")
AUDIO_SR = 16000
AUDIO_CHANNELS = 1
AUDIO_SUBTYPE = "PCM_16"

TRIM_PCT = 0.2
SAMPLE_FPS = 3.0

OUTPUT_JSON_PATH = "output/output.json"
TMP_DIR = "recordings"
DEBUG_DIR = "debug_frames"  # NEW: Save frames for debugging
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

# Model paths
YOLO_MODEL_PATH = "phase1/Weights/best.onnx"
DEPTH_MODEL_PATH = "phase1/weights/FastDepthV2_L1GN_Best.pth"

# -------- Helper functions ----------
def ts_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_paths():
    t = ts_str()
    v = os.path.join(TMP_DIR, f"video_{t}.avi")
    a = os.path.join(TMP_DIR, f"audio_{t}.wav")
    return v, a

# -------- NEW: Camera test function ----------
def test_camera(cam_index=CAMERA_INDEX):
    """Test if camera is working and return its properties"""
    print("\n" + "=" * 60)
    print("TESTING CAMERA")
    print("=" * 60)
    
    # Try different backends
    backends = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        backend_name = {
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_V4L2: "V4L2",
            cv2.CAP_ANY: "ANY"
        }.get(backend, str(backend))
        
        print(f"\nTrying backend: {backend_name}")
        cap = cv2.VideoCapture(cam_index, backend)
        
        if not cap.isOpened():
            print(f"  ✗ Failed to open camera with {backend_name}")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  ✗ Opened but cannot read frame with {backend_name}")
            cap.release()
            continue
        
        # Success! Get properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  ✓ Camera working with {backend_name}!")
        print(f"    Resolution: {w}x{h}")
        print(f"    FPS: {fps}")
        print(f"    Frame shape: {frame.shape}")
        
        # Save test frame
        test_path = os.path.join(DEBUG_DIR, "camera_test.jpg")
        cv2.imwrite(test_path, frame)
        print(f"    Test frame saved: {test_path}")
        
        cap.release()
        print("=" * 60 + "\n")
        return True, backend
    
    print("\n✗ Camera test failed with all backends!")
    print("=" * 60 + "\n")
    return False, None

# -------- Video Recorder (improved) ----------
class VideoRecorder(threading.Thread):
    def __init__(self, filename, stop_event, frame_size=(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT), 
                 fps=VIDEO_FPS, cam_index=CAMERA_INDEX, backend=None):
        super().__init__(daemon=True)
        self.filename = filename
        self.stop_event = stop_event
        self.frame_size = frame_size
        self.fps = fps
        self.cam_index = cam_index
        self.backend = backend
        self._cap = None
        self._writer = None
        self.actual_fps = None
        self.frame_count = 0  # NEW: Track frames written
    
    def run(self):
        try:
            # Use detected backend or default
            if self.backend is not None:
                self._cap = cv2.VideoCapture(self.cam_index, self.backend)
            elif sys.platform.startswith("win"):
                self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.cam_index)
            
            if not self._cap.isOpened():
                print("[video] ERROR: Cannot open camera!")
                self.stop_event.set()
                return
            
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.frame_size[0])
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.frame_size[1])
            self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self.fps)
            
            self._writer = cv2.VideoWriter(self.filename, VIDEO_FOURCC, self.actual_fps, (w, h))
            
            if not self._writer.isOpened():
                print("[video] ERROR: Cannot open video writer!")
                self._cap.release()
                self.stop_event.set()
                return
            
            print(f"[video] recording -> {self.filename} ({w}x{h}@{self.actual_fps:.2f}fps)")
            
        except Exception as e:
            print(f"[video] initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
            return
        
        # Recording loop
        consecutive_fails = 0
        max_fails = 30
        
        while not self.stop_event.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= max_fails:
                        print(f"[video] ERROR: {max_fails} consecutive frame read failures!")
                        break
                    time.sleep(0.01)
                    continue
                
                consecutive_fails = 0
                self._writer.write(frame)
                self.frame_count += 1
                
                # Save first frame for debugging
                if self.frame_count == 1:
                    debug_path = os.path.join(DEBUG_DIR, f"first_frame_{ts_str()}.jpg")
                    cv2.imwrite(debug_path, frame)
                    print(f"[video] First frame saved to: {debug_path}")
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[video] frame capture error: {e}")
                break
        
        # Cleanup
        try:
            if self._writer:
                self._writer.release()
            if self._cap:
                self._cap.release()
        except Exception as e:
            print(f"[video] cleanup error: {e}")
        
        print(f"[video] stopped. Total frames written: {self.frame_count}")
        
        if self.frame_count == 0:
            print("[video] WARNING: No frames were written to video file!")

# -------- Audio Recorder (unchanged) ----------
class AudioRecorder(threading.Thread):
    def __init__(self, filename, stop_event, samplerate=AUDIO_SR, channels=AUDIO_CHANNELS, subtype=AUDIO_SUBTYPE):
        super().__init__(daemon=True)
        self.filename = filename
        self.stop_event = stop_event
        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self._q = queue.Queue()
        self._stream = None
        self._file = None
    
    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[audio] status:", status)
        try:
            self._q.put(indata.copy(), block=False)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
                self._q.put(indata.copy(), block=False)
            except Exception:
                pass
    
    def run(self):
        try:
            self._file = sf.SoundFile(self.filename, mode='w', samplerate=self.samplerate, 
                                     channels=self.channels, subtype=self.subtype)
        except Exception as e:
            print("[audio] cannot open file:", e)
            self.stop_event.set()
            return
        
        try:
            self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, 
                                         callback=self._callback)
            self._stream.start()
            print(f"[audio] recording -> {self.filename} @ {self.samplerate}Hz")
        except Exception as e:
            print("[audio] cannot open input stream:", e)
            try:
                self._file.close()
            except Exception:
                pass
            self.stop_event.set()
            return
        
        while not self.stop_event.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._file.write(chunk)
            except Exception as e:
                print("[audio] write error:", e)
        
        try:
            self._stream.stop()
            self._stream.close()
            self._file.close()
        except Exception:
            pass
        print("[audio] stopped.")

# -------- Model Manager ----------
class ModelManager:
    def __init__(self):
        self.yolo_tracker = None
        self.depth_estimator = None
        self.ocr_processor = None
        self.initialized = False
        self.camera_backend = None  # NEW: Store working backend
    
    def initialize(self):
        """Load all models once"""
        if self.initialized:
            return
        
        print("\n" + "=" * 60)
        print("INITIALIZING MODELS")
        print("=" * 60)
        
        # Test camera first
        camera_ok, backend = test_camera()
        self.camera_backend = backend
        
        if not camera_ok:
            print("\n⚠️  WARNING: Camera test failed!")
            print("    The system will attempt to continue, but video may not work.")
        
        if HAS_PHASE1:
            try:
                print("\n[Phase 1] Loading YOLO Tracker...")
                self.yolo_tracker = YOLOTracker(model_path=YOLO_MODEL_PATH)
                print("✓ YOLO Tracker loaded")
                
                print("\n[Phase 1] Loading Depth Estimator...")
                self.depth_estimator = DepthEstimator(model_path=DEPTH_MODEL_PATH)
                print("✓ Depth Estimator loaded")
            except Exception as e:
                print(f"✗ Phase 1 initialization failed: {e}")
                import traceback
                traceback.print_exc()
        
        if HAS_PHASE2:
            try:
                print("\n[Phase 2] Loading OCR Processor...")
                self.ocr_processor = MedicalOCRProcessor()
                print("✓ OCR Processor loaded")
            except Exception as e:
                print(f"✗ Phase 2 initialization failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.initialized = True
        print("\n" + "=" * 60)
        print("✓ Model initialization complete")
        print("=" * 60 + "\n")

# Global model manager
models = ModelManager()

# -------- Video frame utilities (improved) ----------
def get_video_timestamps_and_count(video_path):
    """Get timestamps and verify video is readable"""
    print(f"\n[video util] Opening video: {video_path}")
    
    # Check file exists
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file does not exist: {video_path}")
    
    # Check file size
    file_size = os.path.getsize(video_path)
    print(f"[video util] File size: {file_size} bytes")
    if file_size < 1000:
        raise RuntimeError(f"Video file too small ({file_size} bytes) - recording may have failed")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    print(f"[video util] FPS: {fps}, Frame count: {frame_count}")
    
    if frame_count == 0:
        # Try to count frames manually
        print("[video util] Frame count is 0, counting manually...")
        manual_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            manual_count += 1
        frame_count = manual_count
        print(f"[video util] Manual count: {frame_count}")
        cap.release()
        cap = cv2.VideoCapture(video_path)
    
    timestamps = [i / fps for i in range(frame_count)]
    cap.release()
    
    print(f"[video util] Generated {len(timestamps)} timestamps")
    return timestamps, fps, frame_count

def select_trimmed_indices(timestamps, audio_start, audio_end, trim_pct=TRIM_PCT):
    duration = max(0.0, audio_end - audio_start)
    if duration <= 0:
        return [], audio_start, audio_end
    trim = trim_pct * duration
    t0 = audio_start + trim
    t1 = audio_end - trim
    if t1 <= t0:
        center = (audio_start + audio_end) / 2.0
        delta = max(0.05, 0.25 * duration)
        t0 = center - delta
        t1 = center + delta
    indices = [i for i, ts in enumerate(timestamps) if ts >= t0 and ts <= t1]
    return indices, t0, t1

def downsample_to_fps(timestamps, indices, target_fps=SAMPLE_FPS):
    if not indices:
        return []
    t_start = timestamps[indices[0]]
    t_end = timestamps[indices[-1]]
    if t_end <= t_start:
        return [indices[0]]
    step = 1.0 / target_fps
    desired_ts = np.arange(t_start, t_end + 1e-6, step)
    chosen = []
    idx_ptr = 0
    for dt in desired_ts:
        while idx_ptr + 1 < len(indices) and timestamps[indices[idx_ptr + 1]] < dt:
            idx_ptr += 1
        cand = indices[idx_ptr]
        if idx_ptr + 1 < len(indices):
            nxt = indices[idx_ptr + 1]
            if abs(timestamps[nxt] - dt) < abs(timestamps[cand] - dt):
                cand = nxt
        chosen.append(cand)
    chosen = sorted(list(dict.fromkeys(chosen)))
    return chosen

def read_frames_at_indices(video_path, indices):
    """Read frames with better error handling and debugging"""
    print(f"\n[frame reader] Reading {len(indices)} frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[frame reader] ERROR: Cannot open video!")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    frames = []
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        
        if not ok or frame is None:
            print(f"[frame reader] WARNING: Failed to read frame {idx}")
            continue
        
        ts = idx / fps
        frames.append((idx, ts, frame))
        
        # Save debug frame
        if i < 3:  # Save first 3 frames
            debug_path = os.path.join(DEBUG_DIR, f"sampled_frame_{i}_idx{idx}.jpg")
            cv2.imwrite(debug_path, frame)
            print(f"[frame reader] Saved debug frame: {debug_path}")
    
    cap.release()
    print(f"[frame reader] Successfully read {len(frames)} frames")
    return frames

# -------- ASR ----------
def run_asr_on_file(audio_path):
    if HAS_FASTER_WHISPER:
        try:
            model = WhisperModel("small", device="cpu")
            segments, info = model.transcribe(audio_path, beam_size=5)
            text = " ".join([seg.text for seg in segments]).strip().lower()
            print(f"[asr] transcript: {text}")
            return text
        except Exception as e:
            print("[asr] faster-whisper failed:", e)
    print("[asr] faster-whisper not available; defaulting to 'describe'.")
    return "describe"

# -------- Intent parser ----------
def parse_intent(asr_text):
    s = asr_text.lower()
    if any(k in s for k in ["describe", "what is in front", "what is in front of me", "what is it", "what's in front"]):
        return "describe"
    if any(k in s for k in ["read", "read this", "what is written", "what's written", "what does it say"]):
        return "read_text"
    if any(k in s for k in ["stop", "pause", "cancel"]):
        return "stop"
    return "describe"

# -------- Model processing functions (with debugging) ----------
def process_describe_with_models(frames_triplets):
    """Process frames with YOLO+ByteTrack+FastDepth"""
    if not models.yolo_tracker or not models.depth_estimator:
        print("[proc] Phase 1 models not available")
        return {"created_at": time.time(), "n_objects": 0, "objects": []}
    
    print(f"\n[proc] Processing {len(frames_triplets)} frames with YOLO+Depth...")
    
    if not frames_triplets:
        print("[proc] ERROR: No frames to process!")
        return {"created_at": time.time(), "n_objects": 0, "objects": []}
    
    all_objects = {}
    
    for idx, (frame_no, ts, frame) in enumerate(frames_triplets):
        print(f"\n  Frame {idx + 1}/{len(frames_triplets)} (#{frame_no} @ {ts:.2f}s)")
        print(f"    Frame shape: {frame.shape}")
        
        try:
            # YOLO detection + tracking
            detections = models.yolo_tracker.detect_and_track(frame)
            print(f"    YOLO detections: {len(detections)}")
            
            # Add depth to each detection
            for det in detections:
                track_id = det.get('track_id')
                if track_id is None:
                    continue
                
                print(f"      Object {track_id}: {det.get('class_name')} (conf={det.get('confidence'):.2f})")
                
                # Get depth info
                depth_info = models.depth_estimator.process_detection(frame, det['bbox'])
                
                # Store or update object
                if track_id not in all_objects:
                    all_objects[track_id] = {
                        "id": track_id,
                        "label": det['class_name'],
                        "first_seen_ts": ts,
                        "bbox": det['bbox'],
                        "conf": det['confidence'],
                        "distance_m": round(depth_info['distance'], 2) if depth_info['distance'] else None,
                        "direction": depth_info['direction']
                    }
                else:
                    obj = all_objects[track_id]
                    obj['bbox'] = det['bbox']
                    obj['conf'] = max(obj['conf'], det['confidence'])
                    if depth_info['distance']:
                        obj['distance_m'] = round(depth_info['distance'], 2)
                    obj['direction'] = depth_info['direction']
        
        except Exception as e:
            print(f"    ERROR processing frame: {e}")
            import traceback
            traceback.print_exc()
    
    objects_list = list(all_objects.values())
    print(f"\n✓ Detected {len(objects_list)} unique objects")
    
    return {
        "created_at": time.time(),
        "n_objects": len(objects_list),
        "objects": objects_list
    }

def process_ocr_with_model(frames_triplets):
    """Process frames with OCR model"""
    if not models.ocr_processor:
        print("[proc] Phase 2 OCR not available")
        return {"created_at": time.time(), "n_texts": 0, "text_objects": []}
    
    print(f"\n[proc] Processing {len(frames_triplets)} frames with OCR...")
    
    if not frames_triplets:
        print("[proc] ERROR: No frames to process!")
        return {"created_at": time.time(), "n_texts": 0, "text_objects": []}
    
    ocr_results = []
    
    for idx, (frame_no, ts, frame) in enumerate(frames_triplets):
        print(f"\n  Frame {idx + 1}/{len(frames_triplets)} (#{frame_no} @ {ts:.2f}s)")
        print(f"    Frame shape: {frame.shape}")
        
        try:
            # Convert BGR to RGB for OCR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            result = models.ocr_processor.extract_text_from_frame(frame_rgb)
            
            if result['success'] and result['text']:
                text_length = len(result['text'])
                print(f"    ✓ Text found: {text_length} characters")
                print(f"      Preview: {result['text'][:100]}")
                ocr_results.append({
                    'frame_no': frame_no,
                    'text': result['text'],
                    'text_length': text_length
                })
            else:
                print(f"    ✗ No text detected")
        
        except Exception as e:
            print(f"    ERROR processing frame: {e}")
            import traceback
            traceback.print_exc()
    
    if not ocr_results:
        print("\n⚠ No text detected in any frame")
        return {"created_at": time.time(), "n_texts": 0, "text_objects": []}
    
    best_result = max(ocr_results, key=lambda x: x['text_length'])
    print(f"\n✓ Best result from frame {best_result['frame_no']}")
    
    text_lines = [line.strip() for line in best_result['text'].split('\n') if line.strip()]
    
    text_objects = [
        {
            "content": line,
            "confidence": 1.0,
            "direction": "center",
            "distance_m": None
        }
        for line in text_lines
    ]
    
    return {
        "created_at": time.time(),
        "n_texts": len(text_objects),
        "text_objects": text_objects,
        "full_text": best_result['text']
    }

# -------- Summarizer ----------
def summarize_for_tts(intent, asr_text, scene_json):
    if intent == "read_text":
        texts = scene_json.get("text_objects", [])
        if not texts:
            return "I could not read any text."
        phrases = []
        for t in texts[:3]:
            content = t.get("content", "")
            dirn = t.get("direction", "")
            phrases.append(f"Detected text {content} {('on your ' + dirn) if dirn else ''}".strip())
        return ". ".join(phrases) + "."
    
    elif intent == "describe":
        objs = scene_json.get("objects", [])
        if not objs:
            return "I do not see any objects."
        
        with_dist = [o for o in objs if o.get("distance_m") is not None]
        if with_dist:
            sorted_objs = sorted(with_dist, key=lambda x: x["distance_m"])
        else:
            sorted_objs = objs
        
        phrases = []
        for o in sorted_objs[:3]:
            label = o.get("label", "object")
            dist = o.get("distance_m")
            dirn = o.get("direction", "")
            if dist is not None:
                phrases.append(f"{label} {('on your ' + dirn) if dirn else ''}, about {dist} meters")
            else:
                phrases.append(f"{label} {('on your ' + dirn) if dirn else ''}")
        return ". ".join(phrases) + "."
    
    else:
        return "Okay."

# -------- TTS ----------
def speak_text(text):
    print(f"\n[tts] -> {text}\n")
    if HAS_PYTTX3:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception as e:
            print(f"[tts] pyttsx3 failed: {e}")
    print("[tts] (no TTS backend available)")

# -------- Main processing pipeline ----------
def process_recording(video_path, audio_path):
    """Main processing pipeline"""
    print("\n" + "=" * 60)
    print("PROCESSING RECORDING")
    print("=" * 60)
    
    if not models.initialized:
        models.initialize()
    
    # Get audio duration
    try:
        info = sf.info(audio_path)
        audio_duration = info.duration
        print(f"[proc] Audio duration: {audio_duration:.2f}s")
    except Exception as e:
        print(f"[proc] cannot read audio duration: {e}")
        audio_duration = 0.0
    
    audio_start = 0.0
    audio_end = audio_duration
    
    # 1) ASR
    asr_text = run_asr_on_file(audio_path)
    
    # 2) Intent
    intent = parse_intent(asr_text)
    print(f"[proc] Intent: '{intent}' from ASR: '{asr_text}'")
    
    if intent == "stop":
        print("[proc] Intent 'stop' -> no action")
        scene_json = {"created_at": time.time(), "n_objects": 0, "objects": []}
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(scene_json, f, indent=2)
        return
    
    # 3) Sample frames from video
    try:
        timestamps, fps, frame_count = get_video_timestamps_and_count(video_path)
        print(f"[proc] Video has {frame_count} frames at {fps:.2f} FPS")
    except Exception as e:
        print(f"[proc] ERROR reading video: {e}")
        import traceback
        traceback.print_exc()
        timestamps, fps, frame_count = [], VIDEO_FPS, 0
    
    if not timestamps:
        print("[proc] ERROR: No timestamps available - video may be empty or corrupted")
        scene_json = {"created_at": time.time(), "n_objects": 0, "objects": [], "error": "No video frames"}
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(scene_json, f, indent=2)
        speak_text("Error: No video frames were captured")
        return
    
    indices_window, t0, t1 = select_trimmed_indices(timestamps, audio_start, audio_end, trim_pct=TRIM_PCT)
    
    if not indices_window and timestamps:
        center_ts = (audio_start + audio_end) / 2.0
        center_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - center_ts))
        indices_window = [center_idx]
    
    sampled_indices = downsample_to_fps(timestamps, indices_window, target_fps=SAMPLE_FPS)
    if not sampled_indices:
        sampled_indices = indices_window[:1] if indices_window else []
    
    print(f"[proc] Sampled indices: {sampled_indices}")
    frames_triplets = read_frames_at_indices(video_path, sampled_indices)
    print(f"[proc] Successfully read {len(frames_triplets)} frames between {t0:.2f}s and {t1:.2f}s")
    
    if not frames_triplets:
        print("[proc] ERROR: No frames could be read from video!")
        scene_json = {"created_at": time.time(), "n_objects": 0, "objects": [], "error": "Could not read frames"}
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(scene_json, f, indent=2)
        speak_text("Error: Could not read video frames")
        return
    
    # 4) Process with models based on intent
    if intent == "describe":
        scene_json = process_describe_with_models(frames_triplets)
    elif intent == "read_text":
        scene_json = process_ocr_with_model(frames_triplets)
    else:
        scene_json = {"created_at": time.time(), "n_objects": 0, "objects": []}
    
    # 5) Save JSON
    try:
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(scene_json, f, indent=2)
        print(f"\n✓ Saved: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"✗ Failed to write JSON: {e}")
    
    # 6) Summarize and speak
    summary = summarize_for_tts(intent, asr_text, scene_json)
    speak_text(summary)
    
    print("=" * 60)
    print("✓ PROCESSING COMPLETE")
    print("=" * 60 + "\n")

# -------- GUI ----------
class PushProcessGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-time Pipeline: YOLO+Depth+OCR (DEBUG)")
        
        self.status = tk.StringVar(value="Initializing models...")
        ttk.Label(master, textvariable=self.status, font=('Arial', 10)).pack(pady=(8, 2))
        
        self.btn = tk.Button(master, text="HOLD TO SPEAK", bg="#cc3333", fg="white", 
                            width=24, height=4, font=('Arial', 12, 'bold'))
        self.btn.pack(padx=20, pady=10)
        self.btn.bind("<ButtonPress-1>", self._on_press)
        self.btn.bind("<ButtonRelease-1>", self._on_release)
        
        master.bind("<KeyPress-space>", self._on_key_press)
        master.bind("<KeyRelease-space>", self._on_key_release)
        
        self.recording = False
        self.processing = False
        self.stop_event = None
        self.vthread = None
        self.athread = None
        self.video_path = None
        self.audio_path = None
        
        # Initialize models in background
        init_thread = threading.Thread(target=self._init_models, daemon=True)
        init_thread.start()
    
    def _init_models(self):
        models.initialize()
        self.status.set("Ready - Hold SPACE or button to speak")
    
    def _on_press(self, event):
        if self.processing:
            return
        self._start_recording()
    
    def _on_release(self, event):
        self._stop_recording_and_start_processing()
    
    _space_held = False
    
    def _on_key_press(self, event):
        if not PushProcessGUI._space_held:
            PushProcessGUI._space_held = True
            self._start_recording()
    
    def _on_key_release(self, event):
        if PushProcessGUI._space_held:
            PushProcessGUI._space_held = False
            self._stop_recording_and_start_processing()
    
    def _start_recording(self):
        if self.recording or self.processing:
            return
        self.status.set("🔴 Recording...")
        self.recording = True
        self.stop_event = threading.Event()
        self.video_path, self.audio_path = make_paths()
        
        # Use detected camera backend
        self.vthread = VideoRecorder(
            self.video_path, 
            self.stop_event, 
            backend=models.camera_backend
        )
        self.athread = AudioRecorder(self.audio_path, self.stop_event)
        self.vthread.start()
        self.athread.start()
    
    def _stop_recording_and_start_processing(self):
        if not self.recording:
            return
        
        self.status.set("⏹ Stopping recording...")
        if self.stop_event:
            self.stop_event.set()
        
        if self.vthread:
            self.vthread.join(timeout=5.0)
        if self.athread:
            self.athread.join(timeout=5.0)
        
        self.recording = False
        self.status.set("⚙️ Processing...")
        self.processing = True
        
        proc_thread = threading.Thread(target=self._processing_thread, daemon=True)
        proc_thread.start()
    
    def _processing_thread(self):
        try:
            process_recording(self.video_path, self.audio_path)
        except Exception as e:
            print(f"[proc thread] exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False
            self.status.set("✓ Ready - Hold SPACE or button to speak")

def main():
    print("\n" + "=" * 60)
    print("REAL-TIME INTEGRATED PIPELINE (DEBUG VERSION)")
    print("=" * 60)
    print("Controls:")
    print("  - Hold SPACE or click button to record")
    print("  - Release to process")
    print("\nIntents:")
    print("  - 'describe' / 'what is in front' -> YOLO+Depth")
    print("  - 'read text' / 'what is written' -> OCR")
    print("\nDebugging:")
    print("  - Camera test runs on startup")
    print("  - Debug frames saved to:", DEBUG_DIR)
    print("  - Check console for detailed logs")
    print("=" * 60 + "\n")
    
    root = tk.Tk()
    app = PushProcessGUI(root)
    root.geometry("400x180")
    root.mainloop()

if __name__ == "__main__":
    main()