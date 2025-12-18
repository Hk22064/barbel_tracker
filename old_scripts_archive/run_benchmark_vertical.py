import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def run_script(command, name):
    print(f"--- Running {name} ---")
    print(f"Command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {name}:\n{result.stderr}")
        return None
    stdout = result.stdout
    print(stdout)
    fps_match = re.search(r"Average FPS: ([\d\.]+)", stdout)
    fps = float(fps_match.group(1)) if fps_match else 0.0
    reps_match = re.search(r"Total Reps Detected.*: (\d+)", stdout)
    reps = int(reps_match.group(1)) if reps_match else 0
    return {'name': name, 'fps': fps, 'reps': reps}

def main():
    video_file = "test_tate.mp4"
    bench_results = []
    
    # 1. YOLO-N (Object) - Expect Fail
    cmd_n = f"python analyze_lift_scale.py Model_B_Clump_n/train/weights/best.pt {video_file} --export_npy output/vel_n_tate.npy"
    res_n = run_script(cmd_n, "YOLO11n (Object)")
    if res_n: bench_results.append(res_n)
    
    # 2. YOLO-X (Pose) - Expect Success?
    cmd_x = f"python analyze_lift_wrist.py yolo11x-pose.pt {video_file} --scale 0.0016 --export_npy output/vel_x_tate.npy"
    res_x = run_script(cmd_x, "YOLO11x (Pose)")
    if res_x: bench_results.append(res_x)
    
    # 3. MediaPipe
    cmd_mp = f"python VBT_mediapipe/app_headless_export.py {video_file} output/vel_mp_tate.npy"
    res_mp = run_script(cmd_mp, "MediaPipe")
    if res_mp: bench_results.append(res_mp)
    
    print("\n=== Vertical Benchmark Results ===")
    print(f"{'Method':<20} | {'FPS':<10} | {'Reps':<5}")
    print("-" * 40)
    for r in bench_results:
        print(f"{r['name']:<20} | {r['fps']:<10.2f} | {r['reps']:<5}")
    
    # Graph Generation
    try:
        vn = np.load("output/vel_n_tate.npy") if os.path.exists("output/vel_n_tate.npy") else []
        vx = np.load("output/vel_x_tate.npy") if os.path.exists("output/vel_x_tate.npy") else []
        vmp = np.load("output/vel_mp_tate.npy") if os.path.exists("output/vel_mp_tate.npy") else []
        
        plt.figure(figsize=(12, 6))
        fps = 30.0 
        
        if len(vn) > 0:
            tn = np.linspace(0, len(vn)/fps, len(vn))
            plt.plot(tn, vn, label="YOLO (Obj)", color='red', linestyle='--')
            
        if len(vx) > 0:
            tx = np.linspace(0, len(vx)/fps, len(vx))
            plt.plot(tx, vx, label="YOLO (Pose)", color='orange')
            
        if len(vmp) > 0:
            tmp = np.linspace(0, len(vmp)/fps, len(vmp))
            plt.plot(tmp, vmp, label="MediaPipe", color='blue')
            
        plt.title("Vertical Video Comparison")
        plt.legend()
        plt.savefig("output/benchmark_graph_tate.png")
        print("Graph saved.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
