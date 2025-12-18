import os
import re
import subprocess
import sys

def run_test_suite():
    print("=== MediaPipe VBT Accuracy Test Suite ===")
    
    # Define test directories
    # Assumes running from project root 'barbel_tracker'
    video_dirs = [
        os.path.join("video", "horizontal"),
        os.path.join("video", "vertical")
    ]
    
    # Pattern to extract reps from filename (e.g. "front_9rep.mp4" -> 9)
    rep_pattern = re.compile(r"(\d+)rep")
    
    results = []
    
    output_dir = "output/test_results"
    os.makedirs(output_dir, exist_ok=True)

    for vid_dir in video_dirs:
        if not os.path.exists(vid_dir):
            print(f"Directory not found: {vid_dir}")
            continue
            
        category = os.path.basename(vid_dir)
        print(f"\n--- Scanning: {category} ---")
        
        for filename in os.listdir(vid_dir):
            if not filename.endswith(".mp4"):
                continue
            
            # Extract true reps
            match = rep_pattern.search(filename)
            if not match:
                print(f"[SKIP] {filename} (No 'Nrep' in name)")
                continue
            
            true_reps = int(match.group(1))
            file_path = os.path.join(vid_dir, filename)
            
            # Prepare Output Path
            output_npy = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
            
            print(f"Processing: {filename} (True Reps: {true_reps})")
            
            # Run MediaPipe Headless
            # python VBT_mediapipe/app_headless_export.py <video> <output>
            cmd = f"python VBT_mediapipe/app_headless_export.py \"{file_path}\" \"{output_npy}\""
            
            try:
                # Capture Output
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"  [ERROR] Script Failed: {result.stderr.strip().splitlines()[-1] if result.stderr else 'Unknown'}")
                    results.append({'file': filename, 'category': category, 'true': true_reps, 'detected': -1, 'status': 'Error'})
                    continue
                
                # Parse Detected Reps
                # Looking for: "Total Reps Detect: X" (or similar from app_headless_export.py)
                # Actual print in app_headless_export.py: "Total Reps Detected: {vbt_analyzer.rep_count}"
                
                reps_match = re.search(r"Total Reps Detected: (\d+)", result.stdout)
                if reps_match:
                    detected_reps = int(reps_match.group(1))
                    
                    diff = abs(detected_reps - true_reps)
                    # Allow tolerance? For VBT, we want exact or +/- 1 usually. 
                    # Let's say Pass if exact or +/- 1 (maybe user stopped slightly early/late or start rep ambiguity)
                    # But strict pass is exact.
                    
                    status = "PASS" if detected_reps == true_reps else f"FAIL (Diff: {detected_reps - true_reps})"
                    
                    print(f"  -> Detected: {detected_reps} [{status}]")
                    results.append({'file': filename, 'category': category, 'true': true_reps, 'detected': detected_reps, 'status': status})
                    
                else:
                    print("  [ERROR] Could not parse rep count from stdout.")
                    results.append({'file': filename, 'category': category, 'true': true_reps, 'detected': -1, 'status': 'ParseError'})

            except Exception as e:
                print(f"  [EXCEPTION] {e}")
    
    # --- Summary Report ---
    print("\n" + "="*60)
    print(f"{'Category':<12} | {'Filename':<20} | {'True':<5} | {'Det':<5} | {'Result'}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['category']:<12} | {r['file']:<20} | {r['true']:<5} | {r['detected']:<5} | {r['status']}")
    print("="*60)

if __name__ == "__main__":
    run_test_suite()
