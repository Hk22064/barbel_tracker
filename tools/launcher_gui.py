import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
import os

# Configuration
WINDOW_TITLE = "VBT Analyzer Launcher"
WINDOW_SIZE = "550x650"

# Paths to scripts
# Paths to scripts
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__)) # tools/
PROJECT_ROOT = os.path.dirname(SCRIPT_ROOT) # barbel_tracker/
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

# Ensure proper paths
APP_SCRIPT = os.path.join(SRC_ROOT, "app_mediapipe.py")
REVIEW_SCRIPT = os.path.join(SCRIPT_ROOT, "review_video.py") # Now in tools
TEST_SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "scripts", "test_proposed_method.py") # Moved to experiments/scripts? Or still in root? 
# Wait, I moved run_*.py to experiments/scripts. Did I move test_proposed_method.py?
# Listing in Step 2408 showed test_proposed_method.py.
# Listing in Step 2440 (src) did NOT show test_proposed_method.py.
# Plan said "Move experiment files to experiments". I moved "run_*.py".
# check if I moved test_proposed_method.py.
ACCURACY_SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "scripts", "run_accuracy_experiment.py")

class VBTLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        
        # Tuning Variables
        self.var_margin = tk.DoubleVar(value=0.3)
        self.var_smooth = tk.IntVar(value=5)
        self.var_threshold = tk.DoubleVar(value=0.05)
        self.var_filter = tk.StringVar(value="average")
        self.var_grip = tk.StringVar(value="middle")
        
        # UI Elements
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        lbl_title = tk.Label(self.root, text="VBT Analyzer System", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=10)
        
        # --- Tuning Parameters ---
        frame_tune = tk.LabelFrame(self.root, text="Tuning Parameters (精度調整)", padx=10, pady=10)
        frame_tune.pack(fill="x", padx=20, pady=5)
        
        # Margin
        frame_m = tk.Frame(frame_tune)
        frame_m.pack(fill="x", pady=2)
        tk.Label(frame_m, text="Margin (ROI余白, 0.1-0.5):", width=25, anchor="w").pack(side="left")
        tk.Scale(frame_m, variable=self.var_margin, from_=0.1, to=0.5, resolution=0.1, orient="horizontal").pack(side="left", fill="x", expand=True)

        # Smoothing
        frame_s = tk.Frame(frame_tune)
        frame_s.pack(fill="x", pady=2)
        tk.Label(frame_s, text="Smoothing (移動平均, 1-10):", width=25, anchor="w").pack(side="left")
        tk.Scale(frame_s, variable=self.var_smooth, from_=1, to=10, resolution=1, orient="horizontal").pack(side="left", fill="x", expand=True)

        # Threshold
        frame_t = tk.Frame(frame_tune)
        frame_t.pack(fill="x", pady=2)
        tk.Label(frame_t, text="Threshold (静止判定, m/s):", width=25, anchor="w").pack(side="left")
        tk.Scale(frame_t, variable=self.var_threshold, from_=0.01, to=0.20, resolution=0.01, orient="horizontal").pack(side="left", fill="x", expand=True)

        # Filter Type
        frame_f = tk.Frame(frame_tune)
        frame_f.pack(fill="x", pady=2)
        tk.Label(frame_f, text="Filter (平滑化手法):", width=25, anchor="w").pack(side="left")
        filter_options = ["average", "kalman", "butterworth"]
        tk.OptionMenu(frame_f, self.var_filter, *filter_options).pack(side="left", fill="x", expand=True)

        # Grip Finger
        frame_g = tk.Frame(frame_tune)
        frame_g.pack(fill="x", pady=2)
        tk.Label(frame_g, text="Grip Finger (81cm Line):", width=25, anchor="w").pack(side="left")
        grip_options = ["index", "middle", "ring", "pinky"]
        tk.OptionMenu(frame_g, self.var_grip, *grip_options).pack(side="left", fill="x", expand=True)
        # -------------------------

        # 1. Realtime Analysis (Webcam)
        btn_webcam = tk.Button(self.root, text="Realtime Analysis (Webcam)", font=("Helvetica", 12),
                               bg="#e1f5fe", command=self.run_app_webcam, height=2, width=30)
        btn_webcam.pack(pady=5)
        
        # 2. Analyze Video File
        btn_video = tk.Button(self.root, text="Analyze Video (Quick / No-Rewind)", font=("Helvetica", 12),
                               bg="#e0f2f1", command=self.run_app_video, height=2, width=35)
        btn_video.pack(pady=5)

        # 2b. Analyze Video (Review Mode)
        btn_review = tk.Button(self.root, text="Analyze & Review (Accurate / Can Rewind)", font=("Helvetica", 12),
                               bg="#b9f6ca", command=self.run_review_video, height=2, width=35)
        btn_review.pack(pady=5)
        
        # 3. Run Accuracy Test (General)
        btn_test = tk.Button(self.root, text="Run Accuracy Test (Proposed Method)", font=("Helvetica", 12),
                             bg="#fff3e0", command=self.run_test_script, height=2, width=35)
        btn_test.pack(pady=5)
        
        # 4. Final Accuracy Experiment (Comparison vs Manual)
        btn_final = tk.Button(self.root, text="Run Final Accuracy Experiment (v.s. Manual)", font=("Helvetica", 12),
                              bg="#ffccbc", command=self.run_final_experiment, height=2, width=35)
        btn_final.pack(pady=5)
        
        # Console Output Area
        tk.Label(self.root, text="Log Output:", font=("Helvetica", 10)).pack(anchor="w", padx=20, pady=(10, 0))
        self.log_area = scrolledtext.ScrolledText(self.root, height=10, state='disabled', font=("Consolas", 9))
        self.log_area.pack(padx=20, pady=5, fill="both", expand=True)
        
    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        
    def run_command(self, command):
        """Runs a command in a separate thread to avoid freezing UI."""
        def target():
            self.log(f"Running: {' '.join(command)}")
            try:
                # Use Popen to capture output in real-time
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=SCRIPT_ROOT  # Execute from project root
                )
                
                # Stream stdout
                for line in process.stdout:
                    self.log(line.strip())
                
                # Stream stderr
                for line in process.stderr:
                    self.log(f"[ERR] {line.strip()}")
                    
                process.wait()
                self.log(f"Process finished with code {process.returncode}")
                
            except Exception as e:
                self.log(f"Error: {e}")
                messagebox.showerror("Execution Error", str(e))

        threading.Thread(target=target, daemon=True).start()

    def get_tuning_args(self):
        return [
            "--margin", str(self.var_margin.get()),
            "--smooth", str(self.var_smooth.get()),
            "--threshold", str(self.var_threshold.get()),
            "--filter_type", self.var_filter.get(),
            "--grip_finger", self.var_grip.get()
        ]

    def run_app_webcam(self):
        cmd = [sys.executable, APP_SCRIPT] + self.get_tuning_args()
        self.run_command(cmd)
        
    def run_app_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialdir=os.path.join(SCRIPT_ROOT, "VBT_mediapipe", "video")
        )
        if file_path:
            cmd = [sys.executable, APP_SCRIPT, "--video", file_path] + self.get_tuning_args()
            self.run_command(cmd)

    def run_review_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File for Review",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialdir=os.path.join(SCRIPT_ROOT, "VBT_mediapipe", "video")
        )
        if file_path:
            cmd = [sys.executable, REVIEW_SCRIPT, "--video", file_path] + self.get_tuning_args()
            self.run_command(cmd)
            
    def run_test_script(self):
        cmd = [sys.executable, TEST_SCRIPT] + self.get_tuning_args()
        self.run_command(cmd)

    def run_final_experiment(self):
        cmd = [sys.executable, ACCURACY_SCRIPT] + self.get_tuning_args()
        self.run_command(cmd)

if __name__ == "__main__":
    root = tk.Tk()
    app = VBTLauncherApp(root)
    root.mainloop()
