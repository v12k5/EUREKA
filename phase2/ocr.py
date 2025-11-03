"""
Phase 2 - Medical OCR Processor (Optimized)
Processes input images and outputs structured JSON results
"""

import cv2
import numpy as np
import torch
import json
from datetime import datetime
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import logging
import time
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalOCRProcessor:
    """Medical text OCR processor using Donut-based model"""
    
    def __init__(self, model_name="chinmays18/medical-prescription-ocr"):
        """Initialize the medical OCR model"""
        logger.info("Loading Medical OCR model...")
        
        try:
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_text_from_frame(self, frame):
        """
        Extract text from a frame (for Phase 1 integration)
        
        Args:
            frame: OpenCV frame (numpy array) or PIL Image
        
        Returns:
            dict: OCR results with extracted text
        """
        start_time = time.time()
        
        try:
            # Handle different input types
            if isinstance(frame, Image.Image):
                pil_image = frame.convert("RGB")
                logger.info(f"Input: PIL Image {pil_image.size}")
            elif isinstance(frame, np.ndarray):
                logger.info(f"Input: NumPy array {frame.shape}")
                
                # IMPORTANT: Check if already RGB or BGR
                # If coming from cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), it's already RGB
                # If directly from cv2.VideoCapture, it's BGR
                
                if len(frame.shape) == 2:  # Grayscale
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:  # RGBA
                    frame_rgb = frame[:, :, :3]  # Take RGB channels
                else:  # Assume it's already RGB (caller should convert)
                    frame_rgb = frame
                
                pil_image = Image.fromarray(frame_rgb.astype('uint8'))
                logger.info(f"Converted to PIL: {pil_image.size}")
            else:
                raise ValueError(f"Unsupported frame type: {type(frame)}")
            
            # Process with model
            pixel_values = self.processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            task_prompt = "<s_ocr>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Decode result
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            elapsed_time = time.time() - start_time
            logger.info(f"OCR completed: {len(generated_text)} characters in {elapsed_time:.2f}s")
            
            return {
                "success": True,
                "text": generated_text,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            elapsed_time = time.time() - start_time
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "processing_time": elapsed_time
            }
    
    def process_image(self, image_path):
        """
        Process an image and extract text
        
        Args:
            image_path: Path to input image
        
        Returns:
            dict: OCR results with extracted text
        """
        start_time = time.time()
        
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Processing: {image_path}")
            
            # Read image
            pil_image = Image.open(image_path).convert("RGB")
            
            # Process with model
            pixel_values = self.processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            task_prompt = "<s_ocr>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Decode result
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing completed in {elapsed_time:.2f}s")
            
            return {
                "success": True,
                "text": generated_text,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    def create_output_json(self, text_content, image_path):
        """
        Create structured JSON output
        
        Args:
            text_content: Extracted text
            image_path: Source image path
        
        Returns:
            dict: Structured JSON output
        """
        text_objects = []
        
        if text_content:
            # Split text into lines
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            
            for line in lines:
                # Determine direction based on line position
                # (simplified - all centered for medical prescriptions)
                text_objects.append({
                    "content": line,
                    "confidence": 1.0,
                    "direction": "center"
                })
        
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "text_objects": text_objects,
            "source_image": os.path.basename(image_path)
        }
        
        return output


def process_single_image(image_path, output_path="output.json"):
    """
    Process a single image and save results to JSON
    
    Args:
        image_path: Path to input image
        output_path: Path to save JSON output
    """
    logger.info("=" * 60)
    logger.info("MEDICAL OCR PROCESSOR")
    logger.info("=" * 60)
    
    try:
        # Initialize OCR
        ocr = MedicalOCRProcessor()
        
        # Process image
        result = ocr.process_image(image_path)
        
        if not result["success"]:
            logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
            return None
        
        # Create structured output
        output_json = ocr.create_output_json(result["text"], image_path)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTED TEXT")
        logger.info("=" * 60)
        logger.info(result["text"])
        logger.info("\n" + "=" * 60)
        logger.info("JSON OUTPUT")
        logger.info("=" * 60)
        print(json.dumps(output_json, indent=2))
        logger.info("=" * 60)
        
        return output_json
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def process_multiple_images(image_folder, output_folder="outputs"):
    """
    Process multiple images from a folder
    
    Args:
        image_folder: Path to folder containing images
        output_folder: Path to save JSON outputs
    """
    logger.info("=" * 60)
    logger.info("BATCH MEDICAL OCR PROCESSOR")
    logger.info("=" * 60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Find all images
    image_files = [
        f for f in os.listdir(image_folder) 
        if f.lower().endswith(supported_formats)
    ]
    
    if not image_files:
        logger.warning(f"No images found in {image_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process\n")
    
    try:
        # Initialize OCR once for all images
        ocr = MedicalOCRProcessor()
        
        # Process each image
        results = []
        for idx, image_file in enumerate(image_files, 1):
            logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_file}")
            
            image_path = os.path.join(image_folder, image_file)
            output_path = os.path.join(
                output_folder, 
                f"{os.path.splitext(image_file)[0]}_output.json"
            )
            
            # Process image
            result = ocr.process_image(image_path)
            
            if result["success"]:
                # Create and save JSON
                output_json = ocr.create_output_json(result["text"], image_path)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_json, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved: {output_path}")
                results.append(output_json)
            else:
                logger.error(f"Failed: {image_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"BATCH PROCESSING COMPLETE: {len(results)}/{len(image_files)} successful")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return None


def main():
    """Main execution function"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    else:
        # Default: process input/test.jpg
        input_path = "input/test.jpg"
        output_path = "output.json"
    
    # Check if input is a file or folder
    if os.path.isfile(input_path):
        # Process single image
        process_single_image(input_path, output_path)
    elif os.path.isdir(input_path):
        # Process all images in folder
        output_folder = output_path if output_path != "output.json" else "outputs"
        process_multiple_images(input_path, output_folder)
    else:
        logger.error(f"Input not found: {input_path}")
        logger.info("\nUsage:")
        logger.info("  Single image: python script.py <image_path> [output.json]")
        logger.info("  Batch mode:   python script.py <folder_path> [output_folder]")
        logger.info("  Default:      python script.py  (processes input/test.jpg)")


if __name__ == "__main__":
    main()