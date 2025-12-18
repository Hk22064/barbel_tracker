import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    """
    MediaPipe Pose estimation wrapper.
    Returns keypoint coordinates in a standardized dictionary format.
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Define the landmarks we care about for easier access
        self.LANDMARK_NAMES = {
            self.mp_pose.PoseLandmark.LEFT_WRIST: 'left_wrist',
            self.mp_pose.PoseLandmark.RIGHT_WRIST: 'right_wrist',
            self.mp_pose.PoseLandmark.LEFT_ELBOW: 'left_elbow',
            self.mp_pose.PoseLandmark.RIGHT_ELBOW: 'right_elbow',
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
            self.mp_pose.PoseLandmark.LEFT_HIP: 'left_hip',
            self.mp_pose.PoseLandmark.RIGHT_HIP: 'right_hip',
            self.mp_pose.PoseLandmark.LEFT_KNEE: 'left_knee',
            self.mp_pose.PoseLandmark.RIGHT_KNEE: 'right_knee',
            self.mp_pose.PoseLandmark.LEFT_ANKLE: 'left_ankle',
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: 'right_ankle',
            self.mp_pose.PoseLandmark.NOSE: 'nose'
        }

    def process_frame(self, frame):
        """
        Process a single image frame and return landmarks.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            
        Returns:
            dict: Dictionary containing keypoint names and [x, y] pixel coordinates.
                  e.g., {'left_wrist': [320, 240], ...}
                  Returns empty dict if no pose detected.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        landmarks_dict = {}
        
        if results.pose_landmarks:
            h, w, _ = frame.shape
            
            for lm_id, name in self.LANDMARK_NAMES.items():
                landmark = results.pose_landmarks.landmark[lm_id]
                
                # Visibility check could be added here if needed
                
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarks_dict[name] = [cx, cy]
                
        return landmarks_dict, results

    def draw_landmarks(self, frame, results):
        """Helper to draw landmarks on the frame using MediaPipe's utility."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
