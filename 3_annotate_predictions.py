import os
import cv2
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Configuration
FRAMES_DIR = OUTPUT_DIR / "frames_by_state"
OUTPUT_CSV = OUTPUT_DIR / "manual_annotations.csv"
STATES_TO_VALIDATE = ["NORMAL", "PHONE_DETECTED", "POSTURE_DEVIATION", "HANDS_OFF_CONTROLS", "OPERATOR_ABSENT"]

KEY_TO_STATE = {
    "1": "NORMAL",
    "2": "PHONE_DETECTED",
    "3": "POSTURE_DEVIATION",
    "4": "HANDS_OFF_CONTROLS",
    "5": "OPERATOR_ABSENT",
    "0": "IGNORE"
}

class PredictionAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Frame Annotation")
        self.label_img = tk.Label(root)
        self.label_img.pack()

        for key, state in KEY_TO_STATE.items():
            if state != "IGNORE":
                btn = tk.Button(root, text=f"{key} - {state}", width=20,
                                command=lambda e=state: self.record_label(e))
                btn.pack(pady=2)

        self.root.bind("<Key>", self.on_key_press)

        self.deduplicate_csv()
        self.existing_records = self.load_existing_csv()
        self.frames = self.load_unannotated_frames()

        if not self.frames:
            messagebox.showinfo("Complete", "All frames have already been annotated!")
            self.show_statistics()
            self.root.quit()
            return

        self.current_index = 0
        self.new_records = []
        self.show_next_frame()

    def deduplicate_csv(self):
        if not os.path.exists(OUTPUT_CSV):
            return

        unique_rows = {}
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) != 3 or not all(c.strip() for c in row):
                    continue
                key = (row[0], row[1])  # file + prediction
                unique_rows[key] = row

        with open(OUTPUT_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "prediction", "ground_truth"])
            for row in unique_rows.values():
                writer.writerow(row)

    def save_csv_row(self, row):
        with open(OUTPUT_CSV, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def load_existing_csv(self):
        annotated = set()
        if os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["file"].strip():
                        annotated.add((row["prediction"], row["file"]))
        return annotated

    def load_unannotated_frames(self):
        frames = []
        for prediction in STATES_TO_VALIDATE:
            directory = os.path.join(FRAMES_DIR, prediction)
            if not os.path.isdir(directory):
                continue
            files = sorted(f for f in os.listdir(directory) if f.endswith(".jpg"))
            for name in files:
                if (prediction, name) not in self.existing_records:
                    path = os.path.join(directory, name)
                    frames.append((path, prediction, name))
        return frames

    def show_next_frame(self):
        if self.current_index >= len(self.frames):
            self.save_csv()
            self.show_statistics()
            messagebox.showinfo("Finished", "Annotation completed!")
            self.root.quit()
            return

        path, prediction, name = self.frames[self.current_index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image_pil = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image_pil)

        self.label_img.configure(image=image_tk)
        self.label_img.image = image_tk
        self.root.title(f"{name} | Predicted: {prediction} | Frame {self.current_index + 1}/{len(self.frames)}")

    def on_key_press(self, event):
        key = event.char
        if key in KEY_TO_STATE:
            label = KEY_TO_STATE[key]
            if label != "IGNORE":
                self.record_label(label)
            else:
                self.skip_frame()

    def record_label(self, label):
        path, prediction, name = self.frames[self.current_index]
        record = [name, prediction, label]
        self.new_records.append(record)
        print(f"Annotated: {name} | Predicted: {prediction} | Ground truth: {label}")
        self.save_csv_row(record)
        self.current_index += 1
        self.show_next_frame()

    def skip_frame(self):
        print(f"Skipped: {self.frames[self.current_index][2]}")
        self.current_index += 1
        self.show_next_frame()

    def save_csv(self):
        if not self.new_records:
            return
        with open(OUTPUT_CSV, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.new_records)
        print(f"New annotations saved to: {OUTPUT_CSV}")

    def show_statistics(self):
        if not os.path.exists(OUTPUT_CSV):
            print("CSV not found for statistics.")
            return

        y_true = []
        y_pred = []

        with open(OUTPUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                y_true.append(row["ground_truth"])
                y_pred.append(row["prediction"])

        print("\n========= PERFORMANCE REPORT =========")
        print(classification_report(y_true, y_pred, labels=STATES_TO_VALIDATE, zero_division=0))

        acc = sum([1 for a, b in zip(y_true, y_pred) if a == b]) / len(y_true)
        print(f"Overall accuracy: {acc:.2%}")

        matrix = confusion_matrix(y_true, y_pred, labels=STATES_TO_VALIDATE)
        print("Confusion matrix:")
        print(matrix)

        messagebox.showinfo("Statistical Summary",
                            f"Accuracy: {acc:.2%}\nTotal Frames: {len(y_true)}\n\nSee the terminal for details.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionAnnotator(root)
    root.mainloop()