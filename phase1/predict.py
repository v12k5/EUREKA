import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import sys
import cv2
import json
from collections import defaultdict

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import FastDepthV2
from load_pretrained import load_pretrained_fastdepth


def predict_depth_on_crop(model, crop_image):
    """
    Performs depth estimation on a cropped image.

    Args:
        model: The FastDepth model.
        crop_image: A PIL Image of the cropped region.

    Returns:
        Estimated distance in meters.
    """
    output_size = (224, 224)

    # --- Preprocessing ---
    resize = transforms.Resize(output_size)
    crop_image = resize(crop_image)

    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(crop_image)
    input_tensor = input_tensor.unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # --- Perform inference ---
    with torch.no_grad():
        model.eval()
        pred = model(input_tensor)

    # --- Post-processing ---
    depth_map = pred.squeeze().cpu().numpy()
    
    # Use mean depth as the object's distance
    distance = np.mean(depth_map)
    
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
    # Calculate center of bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Define thresholds (divide frame into 3x3 grid)
    left_threshold = frame_width * 0.33
    right_threshold = frame_width * 0.67
    top_threshold = frame_height * 0.33
    bottom_threshold = frame_height * 0.67
    
    # Determine horizontal position
    if center_x < left_threshold:
        horizontal = "left"
    elif center_x > right_threshold:
        horizontal = "right"
    else:
        horizontal = "center"
    
    # Determine vertical position
    if center_y < top_threshold:
        vertical = "top"
    elif center_y > bottom_threshold:
        vertical = "bottom"
    else:
        vertical = "center"
    
    # Combine directions
    if horizontal == "center" and vertical == "center":
        return "center"
    elif horizontal == "center":
        return vertical
    elif vertical == "center":
        return horizontal
    else:
        return f"{vertical}-{horizontal}"


if __name__ == '__main__':
    # --- Configuration ---
    model_weights_path = os.path.join('weights', 'FastDepthV2_L1GN_Best.pth')
    video_path = os.path.join('videos', 'check.mp4')
    json_path = os.path.join('output', 'output.json')

    # --- Load the model ---
    print("Loading FastDepth model...")
    model = FastDepthV2()
    model, _ = load_pretrained_fastdepth(model, model_weights_path)
    print("Model loaded successfully.\n")

    # --- Load and group JSON data by frame number ---
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames_data = defaultdict(list)
    for item in data:
        frames_data[item['frame_no']].append(item)

    # --- Process the video ---
    cap = cv2.VideoCapture(video_path)
    
    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        if frame_no in frames_data:
            # Convert the OpenCV frame (BGR) to a PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            for item in frames_data[frame_no]:
                bbox = item['bbox']
                track_id = item['track_id']
                class_name = item['class']

                # Crop the image using the bounding box coordinates
                cropped_image = pil_frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Get the distance for the cropped region
                distance = predict_depth_on_crop(model, cropped_image)
                
                # Get the direction
                direction = get_direction(bbox, frame_width, frame_height)

                # Print distance and direction information
                print(f"Frame {frame_no} | {class_name} (ID: {track_id}) | Direction: {direction:12s} | Distance: {distance:.2f} meters")

    cap.release()
    print("\nFinished processing video.")