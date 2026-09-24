"""Validate head-tilt measurements and export the graph used in the study."""
import argparse
import csv
import json
import math
import os
from collections import deque
from pathlib import Path

# Reduce non-actionable MediaPipe/TensorFlow diagnostic output before importing it.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "head_pose_validation"
ROI_CONFIG = BASE_DIR / "outputs" / "roi_config.json"
DEFAULT_THRESHOLD_DEGREES = 20.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a head-tilt validation graph from a configured ROI.")
    parser.add_argument("--video", type=Path, help="Video to validate. Defaults to the configured video.")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_DEGREES)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-video", action="store_true", help="Also export an annotated validation video.")
    return parser.parse_args()


def load_configured_roi():
    if not ROI_CONFIG.exists():
        raise FileNotFoundError(f"ROI configuration not found: {ROI_CONFIG}. Run 1_configure_roi.py first.")
    with ROI_CONFIG.open() as file:
        config = json.load(file)
    roi = config.get("roi")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("The ROI configuration does not contain a valid rectangular ROI.")
    return tuple(int(value) for value in roi), config.get("video_path")


def head_tilt_degrees(landmarks, pose_module):
    left_ear = landmarks.landmark[pose_module.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[pose_module.PoseLandmark.RIGHT_EAR]
    return min(90.0, math.degrees(math.atan2(abs(right_ear.y - left_ear.y), abs(right_ear.x - left_ear.x))))


def setup_graph(threshold):
    """Create the reusable graph canvas used in the live preview."""
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.set_ylim(0, 90)
    axis.set_xlim(0, 60)
    axis.axhline(threshold, color="red", linestyle="--")
    axis.set_title("Head Tilt (Last 60 Frames)")
    axis.set_ylabel("Degrees")
    line, = axis.plot([], [], "b-")
    figure.tight_layout()
    return figure, axis, line


def update_graph(figure, axis, line, angles):
    """Update the reusable graph and return it as an OpenCV image."""
    line.set_data(range(len(angles)), angles)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def save_live_artifacts(figure, graph_image, roi_frame, graph_path, roi_path):
    """Save the current graph and annotated ROI without slowing each frame."""
    figure.savefig(graph_path, dpi=300)
    cv2.imwrite(str(roi_path), roi_frame)


def main():
    args = parse_arguments()
    roi, configured_video = load_configured_roi()
    video_path = args.video if args.video else Path(configured_video) if configured_video else None
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError("Provide --video or configure a valid video in 1_configure_roi.py.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000)
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = max(0, total_frames - start_frame)
    if args.max_frames:
        frames_to_process = min(frames_to_process, args.max_frames)

    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("The configured ROI is outside the selected video dimensions.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = OUTPUT_DIR / "head_tilt_last_60_frames.png"
    roi_path = OUTPUT_DIR / "current_roi.png"
    print(f"Processing head-pose validation: {video_path.name}")
    print(f"Frames to process: {frames_to_process or 'unknown'}")
    print("Press Q, Esc, or close the window to stop and keep the generated artifacts.")

    video_writer = None
    if args.save_video:
        video_writer = cv2.VideoWriter(str(OUTPUT_DIR / "annotated_head_tilt.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), frames_per_second, (frame_width, frame_height))

    pose_module = mp.solutions.pose
    drawing_module = mp.solutions.drawing_utils
    samples = []
    history = deque(maxlen=60)
    figure, axis, graph_line = setup_graph(args.threshold)
    processed_frames = 0
    alert_frames = 0
    window_name = "Head-Pose Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    with pose_module.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            success, frame = capture.read()
            if not success or (args.max_frames and processed_frames >= args.max_frames):
                break
            processed_frames += 1
            roi_frame = frame[y : y + height, x : x + width]
            result = pose.process(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
            tilt, state, color = 0.0, "OPERATOR_ABSENT", (255, 0, 0)
            if result.pose_landmarks:
                tilt = head_tilt_degrees(result.pose_landmarks, pose_module)
                state = "POSTURE_DEVIATION" if tilt > args.threshold else "NORMAL"
                color = (0, 0, 255) if state == "POSTURE_DEVIATION" else (0, 255, 0)
                alert_frames += state == "POSTURE_DEVIATION"
                drawing_module.draw_landmarks(roi_frame, result.pose_landmarks, pose_module.POSE_CONNECTIONS)

            samples.append({"frame": processed_frames, "time_seconds": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, "head_tilt_degrees": tilt, "state": state})
            history.append(tilt)
            graph_image = update_graph(figure, axis, graph_line, history)
            if processed_frames == 1 or processed_frames % 60 == 0:
                save_live_artifacts(figure, graph_image, roi_frame, graph_path, roi_path)
            preview_graph = cv2.resize(graph_image, (400, 200))
            preview_height, preview_width = preview_graph.shape[:2]
            frame[10 : 10 + preview_height, 10 : 10 + preview_width] = preview_graph
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
            cv2.putText(frame, f"{state} ({tilt:.1f} degrees)", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            alert_percentage = alert_frames / processed_frames * 100
            cv2.putText(frame, f"Posture alerts: {alert_percentage:.1f}%", (max(10, frame_width - 320), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, frame)
            if video_writer:
                video_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if processed_frames == 1 or processed_frames % 300 == 0:
                if frames_to_process:
                    print(f"Processed {processed_frames}/{frames_to_process} frames ({processed_frames / frames_to_process * 100:.1f}%)")
                else:
                    print(f"Processed {processed_frames} frames")

    capture.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    if not samples:
        raise RuntimeError("No frames were processed.")

    with (OUTPUT_DIR / "head_tilt_measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    save_live_artifacts(figure, graph_image, roi_frame, graph_path, roi_path)
    print(f"Processed frames: {processed_frames}")
    print(f"Alert frames: {alert_frames / processed_frames * 100:.1f}%")
    print(f"Head-tilt graph: {graph_path}")
    print(f"Latest annotated ROI: {roi_path}")


if __name__ == "__main__":
    main()
