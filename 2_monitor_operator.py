import cv2
import numpy as np
import json
import math
import mediapipe as mp
import torch
from ultralytics import YOLO
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ========= CONFIGURATION =========
VIDEO_DIRECTORY = BASE_DIR / "videos"  # Directory containing input video files
FRAMES_SKIP = 2  # Skip two frames after each processed frame (3x faster)
SHOW_ROI_ONLY = True  # Display only the region of interest
SAVE_FRAMES = True  # Enable/disable frame export
FRAMES_OUTPUT_DIR = OUTPUT_DIR / "frames_by_state"  # Base directory for exported frames

# Detection settings
HEAD_TILT_THRESHOLD_DEGREES = 17
MIN_POSE_CONFIDENCE = 0.7
CONFIDENCE_THRESHOLD = 0.6
PROCESS_EVERY_N_FRAMES = 2
GENERAL_COOLDOWN = 1.0
PHONE_DETECTED_COOLDOWN = 10.0

# Colors for states
COR_OPERATOR_ABSENT = (255, 0, 0)
COR_HANDS_OFF_CONTROLS = (0, 165, 255)
COR_POSTURE_DEVIATION = (0, 0, 255)
COR_PHONE_DETECTED = (255, 0, 255)
COR_NORMAL = (0, 255, 0)

# ========= INITIALIZATION =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=MIN_POSE_CONFIDENCE,
                    min_tracking_confidence=MIN_POSE_CONFIDENCE)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(str(OUTPUT_DIR / "yolo11x.pt")).to(device)

# ========= FUNCTIONS =========
def load_config():
    try:
        with (OUTPUT_DIR / "roi_config.json").open() as f:
            config = json.load(f)
        return config['roi']
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def calculate_head_tilt(landmarks):
    try:
        left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        vector = np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y])
        angle = math.degrees(math.atan2(abs(vector[1]), abs(vector[0])))
        return min(90, angle)
    except Exception as e:
        print(f"Error calculating head tilt: {e}")
        return 0

def check_wrists_close(pose_landmarks, min_confidence=0.5):
    """Check wrist proximity using 2D Euclidean distance normalized by shoulder width."""
    if not pose_landmarks:
        return False, 0  # Also return normalized distance

    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.visibility < min_confidence or
        right_wrist.visibility < min_confidence):
        return False, 0

    # 2D Euclidean distance between wrists
    dist = np.linalg.norm(
        np.array([left_wrist.x, left_wrist.y]) -
        np.array([right_wrist.x, right_wrist.y])
    )

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width < 0.1:
        return False, 0

    normalized_distance = dist / shoulder_width

    return normalized_distance < 1.3 or (normalized_distance > 2.1), normalized_distance

def check_wrists_low(pose_landmarks, roi_height, min_confidence=0.5):
    """Check whether both visible wrists are in the lower half of the ROI."""
    if not pose_landmarks:
        return False

    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Check visibility
    if (left_wrist.visibility < min_confidence or
        right_wrist.visibility < min_confidence):
        return False

    return (left_wrist.y > 0.5 and right_wrist.y > 0.5)

def filter_phone_detections(boxes, roi_area):
    valid_phones = []
    for box in boxes:
        if int(box.cls) == 67 and box.conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            if 0.01 < (box_area / roi_area) < 0.3:
                valid_phones.append(box)
    return valid_phones

def setup_output_directories():
    """Create output directories for each state if needed"""
    states = ["OPERATOR_ABSENT", "HANDS_OFF_CONTROLS", "POSTURE_DEVIATION", "PHONE_DETECTED", "NORMAL"]

    if not os.path.exists(FRAMES_OUTPUT_DIR):
        os.makedirs(FRAMES_OUTPUT_DIR)

    for state in states:
        state_dir = os.path.join(FRAMES_OUTPUT_DIR, state)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)

    return FRAMES_OUTPUT_DIR

def save_frame(frame, state, frame_counter):
    """Save a frame in the directory for its predicted state."""
    state_dir = os.path.join(FRAMES_OUTPUT_DIR, state)
    frame_filename = f"{frame_counter:06d}.jpg"
    frame_path = os.path.join(state_dir, frame_filename)
    cv2.imwrite(frame_path, frame)
    return frame_path

def export_state_examples():
    """Export one representative ROI image for every detected state."""
    examples_directory = OUTPUT_DIR / "roi_examples"
    examples_directory.mkdir(parents=True, exist_ok=True)
    for state_directory in sorted(path for path in FRAMES_OUTPUT_DIR.iterdir() if path.is_dir()):
        frames = sorted(state_directory.glob("*.jpg"))
        if frames:
            image = cv2.imread(str(frames[0]))
            cv2.imwrite(str(examples_directory / f"{state_directory.name.lower()}_roi_example.png"), image)

def process_video(video_path, roi, frame_counter):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None, frame_counter

    video_name = os.path.basename(video_path)
    x, y, w, h = roi
    roi_area = w * h

    # Window configuration
    cv2.namedWindow(f'Monitoring - {video_name}', cv2.WINDOW_NORMAL)
    if SHOW_ROI_ONLY:
        cv2.resizeWindow(f'Monitoring - {video_name}', w, h)
    else:
        half_w, half_h = w//2, h//2
        cv2.resizeWindow(f'Monitoring - {video_name}', half_w, half_h)

    frames_by_state = {
        "OPERATOR_ABSENT": 0,
        "HANDS_OFF_CONTROLS": 0,
        "POSTURE_DEVIATION": 0,
        "PHONE_DETECTED": 0,
        "NORMAL": 0
    }
    total_frames = 0
    last_state_change = 0
    last_phone_time = None
    current_state = "NORMAL"

    while cap.isOpened():
        # Skip frames as configured
        for _ in range(FRAMES_SKIP + 1):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        total_frames += 1

        # Extract ROI only
        if SHOW_ROI_ONLY:
            display_frame = frame[y:y+h, x:x+w].copy()
            roi_frame = display_frame
        else:
            display_frame = cv2.resize(frame, (w//2, h//2))
            roi_frame = frame[y:y+h, x:x+w]

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        time_since_last_change = timestamp - last_state_change

        # Run detections on the ROI only
        results = model.predict(roi_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_roi)

        # Determine state
        angle = 0
        wrists_close = False
        wrists_low = False
        found_phone = False
        wrists_visible = False
        normalized_distance = -1  # Default value indicating not calculated
        person_detected = pose_results.pose_landmarks is not None  # Indicates whether a pose was detected

        # Check for a mobile phone
        all_boxes = [box for r in results for box in r.boxes]
        valid_phones = filter_phone_detections(all_boxes, roi_area)
        found_phone = len(valid_phones) > 0

        # Check posture and hands
        if person_detected:
            angle = calculate_head_tilt(pose_results.pose_landmarks)

            # Check wrist visibility
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrists_visible = (left_wrist.visibility >= 0.5 and
                            right_wrist.visibility >= 0.5)

            if wrists_visible:
                # Calculate normalized distance
                wrists_close, normalized_distance = check_wrists_close(pose_results.pose_landmarks)
                wrists_low = check_wrists_low(pose_results.pose_landmarks, h)

        # State hierarchy logic - HIGHEST PRIORITY FIRST
        new_state = current_state

        if not person_detected:  # First check whether no person was detected
            if current_state != "OPERATOR_ABSENT" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "OPERATOR_ABSENT"
        elif found_phone: # Mobile phone detected
            if current_state != "PHONE_DETECTED" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "PHONE_DETECTED"
                last_phone_time = timestamp
        elif wrists_visible and wrists_close and wrists_low: # Hands very close in the lower half
            if current_state != "HANDS_OFF_CONTROLS" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "HANDS_OFF_CONTROLS"
        elif current_state == "PHONE_DETECTED":
            if last_phone_time and (timestamp - last_phone_time) < PHONE_DETECTED_COOLDOWN:
                new_state = "PHONE_DETECTED"
            elif time_since_last_change >= GENERAL_COOLDOWN:
                if angle > HEAD_TILT_THRESHOLD_DEGREES:
                    new_state = "POSTURE_DEVIATION"
                elif not wrists_visible or (wrists_close and not wrists_low):
                    new_state = "HANDS_OFF_CONTROLS"
                else:
                    new_state = "NORMAL"
        elif angle > HEAD_TILT_THRESHOLD_DEGREES:
            if current_state != "POSTURE_DEVIATION" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "POSTURE_DEVIATION"
        elif not wrists_visible or (wrists_close and not wrists_low):
            if current_state != "HANDS_OFF_CONTROLS" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "HANDS_OFF_CONTROLS"
        else:
            if current_state != "NORMAL" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "NORMAL"

        # Update state
        if new_state != current_state:
            current_state = new_state
            last_state_change = timestamp

        frames_by_state[current_state] += 1

        # Visualization
        color_map = {
            "OPERATOR_ABSENT": COR_OPERATOR_ABSENT,
            "HANDS_OFF_CONTROLS": COR_HANDS_OFF_CONTROLS,
            "POSTURE_DEVIATION": COR_POSTURE_DEVIATION,
            "PHONE_DETECTED": COR_PHONE_DETECTED,
            "NORMAL": COR_NORMAL
        }

        state_text = {
            "OPERATOR_ABSENT": "OPERATOR ABSENT",
            "HANDS_OFF_CONTROLS": "HANDS OFF CONTROLS",
            "POSTURE_DEVIATION": "POSTURE DEVIATION",
            "PHONE_DETECTED": "PHONE DETECTED",
            "NORMAL": "NORMAL"
        }

        color = color_map[current_state]

        if SHOW_ROI_ONLY:
            # Draw directly on the ROI
            cv2.rectangle(display_frame, (0, 0), (w, h), color, 3)
            cv2.putText(display_frame, f"{state_text[current_state]} ({angle:.1f} degrees)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Always show wrist distance (with conditional formatting)
            dist_text = "Wrist distance: N/A" if normalized_distance < 0 else f"Wrist distance: {normalized_distance:.2f}"
            cv2.putText(display_frame, dist_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw mobile phone bounding boxes
            for box in valid_phones:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(display_frame, "PHONE_DETECTED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Save frame when SAVE_FRAMES is enabled
        if SAVE_FRAMES:
            save_frame(display_frame, current_state, frame_counter)
            frame_counter += 1

        cv2.imshow(f'Monitoring - {video_name}', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f'Monitoring - {video_name}')

    return {
        "video": video_name,
        "total_frames": total_frames,
        "states": {k: v for k, v in frames_by_state.items()},
        "percentages": {k: (v / total_frames * 100) if total_frames > 0 else 0
                        for k, v in frames_by_state.items()}
    }, frame_counter

def discover_videos(directory):
    """Return supported video files found directly in a directory."""
    return sorted(
        path for extension in ("*.mp4", "*.avi", "*.mov", "*.mkv")
        for path in directory.glob(extension)
    )


def select_video_directory():
    """Open a folder picker when the default input directory has no videos."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    selected = filedialog.askdirectory(title="Select a directory containing input videos")
    root.destroy()
    return Path(selected) if selected else None


def main():
    roi = load_config()
    if roi is None:
        return

    # Configure output directories when SAVE_FRAMES is enabled
    if SAVE_FRAMES:
        setup_output_directories()

    video_files = discover_videos(VIDEO_DIRECTORY)
    if not video_files:
        print(f"No supported video files found in: {VIDEO_DIRECTORY}")
        selected_directory = select_video_directory()
        if selected_directory is None:
            print("No video directory selected.")
            return
        video_files = discover_videos(selected_directory)
        if not video_files:
            print(f"No supported video files found in: {selected_directory}")
            return

    reports = []
    frame_counter = 1  # Global frame counter

    for video_file in video_files:
        print(f"\nProcessing: {video_file}")
        report, frame_counter = process_video(video_file, roi, frame_counter)
        if report:
            reports.append(report)

    export_state_examples()
    pose.close()
    cv2.destroyAllWindows()

    # Final report
    print("\n=== FINAL REPORT ===")
    for report in reports:
        print(f"\nVideo: {report['video']}")
        print(f"Total frames: {report['total_frames']}")
        for state, percentage in report['percentages'].items():
            print(f"{state}: {percentage:.1f}%")

if __name__ == "__main__":
    main()