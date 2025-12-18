import cv2
import argparse
from ultralytics import YOLO
import sys

def main():
    parser = argparse.ArgumentParser(description="Auto-Crop Video to Person")
    parser.add_argument("input_video", type=str, help="Path to input video")
    parser.add_argument("--output_video", type=str, default=None, help="Path to output video")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin percentage (default 0.2 = 20%)")
    args = parser.parse_args()

    input_path = args.input_video
    if args.output_video:
        output_path = args.output_video
    else:
        # Auto-name: filename_cropped.mp4
        import os
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cropped{ext}"

    print(f"Processing: {input_path}")
    print("Loading YOLO model for detection...")
    model = YOLO("yolov8n.pt") # Use the lightest model for speed

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video.")
        sys.exit(1)

    # 1. Analyze first few frames to find the person and determine Crop Box
    # We assume camera is STATIC. So we find the person in the first valid frame.
    
    x1, y1, x2, y2 = 0, 0, 0, 0
    found_person = False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Scanning for person...")
    for i in range(30): # Check first 30 frames
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, classes=[0], verbose=False) # class 0 = person
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Pick largest person
            boxes = results[0].boxes
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = areas.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            
            x1, y1, x2, y2 = box
            found_person = True
            break
            
    if not found_person:
        print("No person detected in the beginning of the video.")
        sys.exit(1)

    # Calculate Crop with Margin
    box_w = x2 - x1
    box_h = y2 - y1
    margin_x = int(box_w * args.margin)
    margin_y = int(box_h * args.margin)
    
    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(width, x2 + margin_x)
    crop_y2 = min(height, y2 + margin_y)
    
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    print(f"Person detected. Cropping to: {crop_w}x{crop_h} (Original: {width}x{height})")
    print(f"Saving to: {output_path}")

    # Reset Video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Setup Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Crop
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        out.write(cropped_frame)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_frames} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nDone! Created {output_path}")
    print("You can now run app.py on this new video.")

if __name__ == "__main__":
    main()
