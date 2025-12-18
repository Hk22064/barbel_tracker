import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def run_script(command, name):
    print(f"--- Running {name} ---")
    print(f"Command: {command}")
    
    # Run and capture output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {name}:\n{result.stderr}")
        return None
    
    stdout = result.stdout
    print(stdout)
    
    # Parse FPS
    fps_match = re.search(r"Average FPS: ([\d\.]+)", stdout)
    fps = float(fps_match.group(1)) if fps_match else 0.0
    
    # Parse Reps
    reps_match = re.search(r"Total Reps Detected.*: (\d+)", stdout)
    reps = int(reps_match.group(1)) if reps_match else 0
    
    return {'name': name, 'fps': fps, 'reps': reps}

def main():
    bench_results = []
    
    # 1. YOLO-N (Object)
    # Using analyze_lift_scale.py
    # Note: --output_npy requires path
    cmd_n = "python analyze_lift_scale.py Model_B_Clump_n/train/weights/best.pt test.mp4 --export_npy output/vel_n.npy"
    res_n = run_script(cmd_n, "YOLO11n (Object)")
    if res_n: bench_results.append(res_n)
    
    # 2. YOLO-X (Pose)
    # Using analyze_lift_wrist.py
    # Scale: Using 0.0016 as per Integrated calibration default
    cmd_x = "python analyze_lift_wrist.py yolo11x-pose.pt test.mp4 --scale 0.0016 --export_npy output/vel_x.npy"
    res_x = run_script(cmd_x, "YOLO11x (Pose)")
    if res_x: bench_results.append(res_x)
    
    # 3. MediaPipe
    # Using app_headless_export.py
    cmd_mp = "python VBT_mediapipe/app_headless_export.py test.mp4 output/vel_mp.npy"
    res_mp = run_script(cmd_mp, "MediaPipe")
    if res_mp: bench_results.append(res_mp)
    
    print("\n=== Benchmark Results ===")
    print(f"{'Method':<20} | {'FPS':<10} | {'Reps':<5}")
    print("-" * 40)
    for r in bench_results:
        print(f"{r['name']:<20} | {r['fps']:<10.2f} | {r['reps']:<5}")

    # Generate Graph
    print("\nGenerating Comparison Graph...")
    try:
        vn = np.load("output/vel_n.npy")
        vx = np.load("output/vel_x.npy")
        vmp = np.load("output/vel_mp.npy") # This is likely shorter if frames were skipped or different count
        
        plt.figure(figsize=(12, 6))
        
        # We assume 30 FPS for X axis if not saved
        fps = 30.0 
        
        tn = np.linspace(0, len(vn)/fps, len(vn))
        tx = np.linspace(0, len(vx)/fps, len(vx))
        tmp = np.linspace(0, len(vmp)/fps, len(vmp))
        
        plt.plot(tn, vn, label=f"YOLO11n (Object) - {bench_results[0]['fps']:.1f} FPS", color='red', alpha=0.5, linestyle='--')
        plt.plot(tx, vx, label=f"YOLO11x (Pose) - {bench_results[1]['fps']:.1f} FPS", color='orange', alpha=0.8)
        plt.plot(tmp, vmp, label=f"MediaPipe - {bench_results[2]['fps']:.1f} FPS", color='blue', alpha=0.9)
        
        plt.title("Method Comparison: Velocity & Speed")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("output/benchmark_graph.png")
        print("Graph saved to output/benchmark_graph.png")
        
    except Exception as e:
        print(f"Error generating graph: {e}")

if __name__ == "__main__":
    main()
