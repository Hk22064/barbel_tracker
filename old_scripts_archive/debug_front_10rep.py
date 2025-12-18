import cv2
import time
from VBT_mediapipe.yolo11_hybrid_estimator import Yolo11HybridEstimator
from VBT_mediapipe.vbt_analyzer import VBTAnalyzer

VIDEO_PATH = "VBT_mediapipe/video/vertical/front_10rep.mp4"

def debug_run():
    print(f"DEBUG: Running on {VIDEO_PATH}")
    
    estimator = Yolo11HybridEstimator(model_path="yolo11n.pt")
    analyzer = VBTAnalyzer()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Process
        landmarks, _ = estimator.process_frame(frame)
        
        if estimator.locked_box:
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Locked Box {estimator.locked_box}")
                # Save debug image
                debug_img = frame.copy()
                estimator.draw_bbox(debug_img, estimator.locked_box)
                cv2.imwrite(f"debug_front_10rep_lock.jpg", debug_img)
        else:
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: No Lock yet...")

        # Check Analysis
        if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            bar_y = (lw[1] + rw[1]) / 2.0
            
            # Calibration Debug
            if analyzer.calibration_factor is None:
                analyzer.calibrate(lw, rw)
                if analyzer.calibration_factor:
                    print(f"Frame {frame_count}: Calibrated! Factor={analyzer.calibration_factor}")
            
            if analyzer.calibration_factor:
                vel = analyzer.calculate_velocity(bar_y, frame_time)
                prev_reps = analyzer.rep_count
                analyzer.process_rep(vel)
                
                if analyzer.rep_count > prev_reps:
                    print(f"*** REP DETECTED! Total: {analyzer.rep_count} at {frame_time:.2f}s ***")
                
                # Debug internal state if concentric
                # if analyzer.current_state == analyzer.STATE_CONCENTRIC:
                #    print(f"   In Concentric: Vel={vel:.2f}")

    print(f"Final Rep Count: {analyzer.rep_count}")
    cap.release()
    estimator.close()

if __name__ == "__main__":
    debug_run()
