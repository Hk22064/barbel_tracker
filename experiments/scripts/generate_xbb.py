import os
import cv2

TARGET_DIR = "Thesis_Materials"

def generate_xbb():
    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_path = os.path.join(root, f)
                img = cv2.imread(full_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Create .xbb filename
                xbb_name = os.path.splitext(full_path)[0] + ".xbb"
                
                # Write XBB content in standard format
                with open(xbb_name, "w") as xbb_file:
                    xbb_file.write(f"%%Title: {f}\n")
                    xbb_file.write(f"%%Creator: python_script\n")
                    xbb_file.write(f"%%BoundingBox: 0 0 {w} {h}\n")
                    xbb_file.write(f"%%HiResBoundingBox: 0.000000 0.000000 {float(w):.6f} {float(h):.6f}\n")
                
                print(f"Generated {xbb_name}: {w}x{h}")

if __name__ == "__main__":
    generate_xbb()
