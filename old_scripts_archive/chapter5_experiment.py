import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def run_tracker(command, name):
    print(f"--> Running: {name}")
    try:
        # Capture stdout/stderr
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    [FAIL] {name}: {result.stderr.strip().splitlines()[-1] if result.stderr else 'Unknown Error'}")
            return None
        
        stdout = result.stdout
        
        # Parse Metrics
        fps_match = re.search(r"Average FPS: ([\d\.]+)", stdout)
        reps_match = re.search(r"Total Reps Detected.*: (\d+)", stdout)
        
        fps = float(fps_match.group(1)) if fps_match else 0.0
        reps = int(reps_match.group(1)) if reps_match else 0
        
        print(f"    [DONE] FPS: {fps:.2f}, Reps: {reps}")
        return {'name': name, 'fps': fps, 'reps': reps}
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

def main():
    print("=== Thesis Chapter 5: Angle Robustness Experiment ===")
    
    # Files
    video_std = "test.mp4"
    video_vert = "test_tate.mp4"
    
    # Models
    model_obj = "Model_B_Clump_n/train/weights/best.pt"
    model_pose = "yolo11x-pose.pt"
    
    results_std = []
    results_vert = []
    
    # --- Condition A: Standard (Horizontal) ---
    print("\n[Condition A] Standard Video (Horizontal)")
    
    # 1. MediaPipe
    cmd = f"python VBT_mediapipe/app_headless_export.py {video_std} output/vel_mp_std.npy"
    res = run_tracker(cmd, "MediaPipe")
    if res: results_std.append(res)
    
    # 2. YOLOv11x (Pose)
    cmd = f"python analyze_lift_wrist.py {model_pose} {video_std} --scale 0.0016 --export_npy output/vel_yolo_pose_std.npy"
    res = run_tracker(cmd, "YOLOv11x (Pose)")
    if res: results_std.append(res)
    
    # 3. YOLOv11n (Object)
    cmd = f"python analyze_lift_scale.py {model_obj} {video_std} --export_npy output/vel_yolo_obj_std.npy"
    res = run_tracker(cmd, "YOLOv11n (Object)")
    if res: results_std.append(res)

    # --- Condition B: Vertical (Vertical) ---
    print("\n[Condition B] Vertical Video (Vertical)")
    
    # 1. MediaPipe
    cmd = f"python VBT_mediapipe/app_headless_export.py {video_vert} output/vel_mp_vert.npy"
    res = run_tracker(cmd, "MediaPipe")
    if res: results_vert.append(res)
    
    # 2. YOLOv11x (Pose)
    cmd = f"python analyze_lift_wrist.py {model_pose} {video_vert} --scale 0.0016 --export_npy output/vel_yolo_pose_vert.npy"
    res = run_tracker(cmd, "YOLOv11x (Pose)")
    if res: results_vert.append(res)
    
    # 3. YOLOv11n (Object)
    cmd = f"python analyze_lift_scale.py {model_obj} {video_vert} --export_npy output/vel_yolo_obj_vert.npy"
    res = run_tracker(cmd, "YOLOv11n (Object)")
    if res:
        results_vert.append(res)
    else:
        results_vert.append({'name': "YOLOv11n (Object)", 'fps': 0.0, 'reps': 0}) # Explicit fail

    # --- Generate Report ---
    print("\n\n=== Experiment Results (Markdown) ===")
    
    print("\n#### Condition A (Standard)")
    print("| Method | FPS | Reps |")
    print("| :--- | :---: | :---: |")
    for r in results_std:
        print(f"| {r['name']} | {r['fps']:.2f} | {r['reps']} |")
        
    print("\n#### Condition B (Vertical)")
    print("| Method | FPS | Reps |")
    print("| :--- | :---: | :---: |")
    for r in results_vert:
        eval_status = "Success" if r['reps'] >= 5 else ("Partial" if r['reps'] > 0 else "Fail")
        print(f"| {r['name']} | {r['fps']:.2f} | {r['reps']} ({eval_status}) |")

    # --- Generate Graphs ---
    print("\nGenerating Graphs...")
    
    def plot_graph(npy_dict, title, filename):
        plt.figure(figsize=(10, 5))
        fps = 30.0
        for label, path in npy_dict.items():
            if os.path.exists(path):
                data = np.load(path)
                t = np.linspace(0, len(data)/fps, len(data))
                plt.plot(t, data, label=label)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        print(f"Saved: {filename}")

    # Graph A
    plot_graph({
        "MediaPipe": "output/vel_mp_std.npy",
        "YOLOv11x (Pose)": "output/vel_yolo_pose_std.npy",
        "YOLOv11n (Obj)": "output/vel_yolo_obj_std.npy"
    }, "Condition A: Standard", "output/graph_cond_a.png")
    
    # Graph B
    plot_graph({
        "MediaPipe": "output/vel_mp_vert.npy",
        "YOLOv11x (Pose)": "output/vel_yolo_pose_vert.npy",
        "YOLOv11n (Obj)": "output/vel_yolo_obj_vert.npy"
    }, "Condition B: Vertical", "output/graph_cond_b.png")

if __name__ == "__main__":
    main()
