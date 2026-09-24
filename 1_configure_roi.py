import json
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_FILE = OUTPUT_DIR / "roi_config.json"
WINDOW_NAME = "Define Operator ROI"
CONFIRMED_ROI_COLOR = (0, 255, 0)
UNCONFIRMED_ROI_COLOR = (0, 255, 255)


def select_video_file():
    """Open a file picker and return a selected video path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    selected = filedialog.askopenfilename(
        title="Select a video for analysis",
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()
    return selected


def load_saved_config():
    """Load the saved configuration, or return an empty configuration."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with CONFIG_FILE.open() as file:
            config = json.load(file)
        return config if isinstance(config, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def valid_roi(roi):
    """Validate and normalize a rectangular ROI."""
    if not isinstance(roi, list) or len(roi) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in roi):
        return None
    return [int(value) for value in roi]


def draw_roi(image, roi, color, thickness):
    x, y, width, height = roi
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)


def setup_roi(video_path, saved_roi):
    """Draw, adjust, and save an ROI. Return True when another video is requested."""
    capture = cv2.VideoCapture(video_path)
    success, frame = capture.read()
    capture.release()
    if not success:
        print("Error reading video")
        return False

    original_height, original_width = frame.shape[:2]
    current_roi = None
    confirmed_roi = valid_roi(saved_roi)
    drawing = False
    change_video = False
    scale_x = scale_y = 1.0
    max_display_width = 1280
    max_display_height = 800
    display_scale = min(max_display_width / original_width, max_display_height / original_height, 1.0)
    display_width = max(1, int(original_width * display_scale))
    display_height = max(1, int(original_height * display_scale))
    scale_x = display_width / original_width
    scale_y = display_height / original_height


    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_roi, confirmed_roi, drawing
        x_original = min(max(int(x / scale_x), 0), original_width - 1)
        y_original = min(max(int(y / scale_y), 0), original_height - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            confirmed_roi = None
            drawing = True
            current_roi = [x_original, y_original, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_roi[2] = x_original - current_roi[0]
            current_roi[3] = y_original - current_roi[1]
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            current_roi[2] = x_original - current_roi[0]
            current_roi[3] = y_original - current_roi[1]
            if current_roi[2] < 0:
                current_roi[0] += current_roi[2]
                current_roi[2] = abs(current_roi[2])
            if current_roi[3] < 0:
                current_roi[1] += current_roi[3]
                current_roi[3] = abs(current_roi[3])

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    print("Instructions:")
    print("1. Draw an ROI around the operator seat")
    print("2. Press SPACE to confirm")
    print("3. Press V to select a different video")
    print("4. Press S to save or Q to quit")

    while True:
        display = frame.copy()
        if confirmed_roi:
            draw_roi(display, confirmed_roi, CONFIRMED_ROI_COLOR, 3)
        elif current_roi:
            draw_roi(display, current_roi, UNCONFIRMED_ROI_COLOR, 2)
        display = cv2.resize(display, (display_width, display_height))
        cv2.putText(
            display,
            "Draw ROI | SPACE: confirm | S: save | V: change video | Q: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("ROI configuration canceled")
            break
        if key == ord(" ") and current_roi:
            confirmed_roi = current_roi.copy()
            print(f"ROI confirmed: {confirmed_roi}")
        elif key == ord("s"):
            if confirmed_roi:
                CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
                with CONFIG_FILE.open("w") as file:
                    json.dump({"video_path": video_path, "roi": confirmed_roi}, file, indent=4)
                print(f"Configuration saved to {CONFIG_FILE}")
                break
            print("Confirm the ROI before saving")
        elif key == ord("v"):
            change_video = True
            break
        elif key == ord("q"):
            print("ROI configuration canceled")
            break

    cv2.destroyAllWindows()
    return change_video


def main():
    config = load_saved_config()
    video_path = config.get("video_path")
    if not isinstance(video_path, str) or not Path(video_path).is_file():
        video_path = None
    elif video_path:
        print(f"Using configured video: {video_path}")

    while True:
        if not video_path:
            video_path = select_video_file()
        if not video_path:
            print("No video selected")
            return
        if not setup_roi(video_path, config.get("roi")):
            return
        video_path = None
        config = {}


if __name__ == "__main__":
    main()
