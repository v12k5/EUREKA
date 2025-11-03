"""
Setup Verification Script
Checks if all dependencies and models are correctly installed
"""

import os
import sys

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'pillow',
        'ultralytics': 'ultralytics',
        'onnxruntime': 'onnxruntime'
    }
    
    all_installed = True
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA/GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
            return False
    except:
        print(f"  ⚠ Could not check CUDA")
        return False

def check_file_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_structure = {
        'phase1': 'Directory for Phase 1 (YOLO+Depth)',
        'phase1/yolo_tracker.py': 'YOLO tracker module',
        'phase1/depth_estimator.py': 'Depth estimator module',
        'phase2': 'Directory for Phase 2 (OCR)',
        'phase2/ocr.py': 'OCR processor module',
        'combine_pipeline.py': 'Main integration script'
    }
    
    all_present = True
    
    for path, description in required_structure.items():
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} - MISSING ({description})")
            all_present = False
    
    return all_present

def check_models():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    models = {
        'phase1/Weights/best.onnx': 'YOLO model (required)',
        'phase1/weights/FastDepthV2_L1GN_Best.pth': 'Depth model (optional - only for --mode both)'
    }
    
    any_yolo = False
    
    for path, description in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {path} ({size_mb:.1f} MB)")
            if 'yolo' in path.lower() or 'best' in path.lower():
                any_yolo = True
        else:
            print(f"  ✗ {path} - MISSING")
            print(f"     {description}")
    
    # OCR model check
    print(f"\n  ℹ OCR model will be auto-downloaded on first run")
    print(f"     Location: ~/.cache/huggingface/")
    print(f"     Size: ~1.5GB (one-time download)")
    
    return any_yolo

def check_input_video():
    """Check if input video exists"""
    print("\nChecking input video...")
    
    if os.path.exists('input.mp4'):
        size_mb = os.path.getsize('input.mp4') / (1024 * 1024)
        print(f"  ✓ input.mp4 ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠ input.mp4 not found")
        print(f"     Place your video as 'input.mp4' in project root")
        return False

def create_output_directory():
    """Create output directory if it doesn't exist"""
    print("\nChecking output directory...")
    
    if not os.path.exists('output'):
        os.makedirs('output')
        print(f"  ✓ Created output/ directory")
    else:
        print(f"  ✓ output/ directory exists")
    
    return True

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("YOLO + Depth + OCR", 
         "python combine_pipeline.py --mode both --ocr yes --ocr-time 30"),
        
        ("YOLO only + OCR",
         "python combine_pipeline.py --mode yolo --ocr yes --ocr-time 15"),
        
        ("YOLO + Depth (no OCR)",
         "python combine_pipeline.py --mode both --ocr no"),
        
        ("Fast processing (5 FPS)",
         "python combine_pipeline.py --mode yolo --ocr yes --ocr-time 30 --target-fps 5")
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"\n{i}. {desc}:")
        print(f"   {cmd}")

def main():
    """Run all verification checks"""
    print("\n" + "=" * 60)
    print("COMBINED PIPELINE SETUP VERIFICATION")
    print("=" * 60)
    
    checks = []
    
    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("CUDA/GPU", check_cuda()))
    checks.append(("File Structure", check_file_structure()))
    checks.append(("Model Files", check_models()))
    checks.append(("Input Video", check_input_video()))
    checks.append(("Output Directory", create_output_directory()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {check_name}")
    
    # Overall status
    critical_checks = ["Python Version", "Dependencies", "File Structure", "Model Files"]
    critical_passed = all(passed for name, passed in checks if name in critical_checks)
    
    print("\n" + "=" * 60)
    
    if critical_passed:
        print("✓ READY TO RUN")
        print("=" * 60)
        print_usage_examples()
    else:
        print("✗ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease fix the issues above before running the pipeline.")
        print("\nInstallation commands:")
        print("  pip install opencv-python numpy torch ultralytics onnxruntime")
        print("  pip install transformers pillow")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()