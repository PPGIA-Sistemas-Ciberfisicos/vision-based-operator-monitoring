# Mobile Phone Detection for Mining Teleoperation Safety

Train or evaluate YOLO11 mobile-phone detectors using your own private YOLO-format dataset.

## Repository contents

- `models/yolo11n_fold5_best.pt`: released YOLO11-Nano checkpoint.
- `yolo_cross_validator.py`: command-line training and cross-validation script.
- `requirements.txt`: Python dependencies.

## 1. Set up the environment

Tested with Python 3.12 on Linux. Python 3.10 to 3.12 is recommended.

```bash
git clone <repository-url>
cd mobile-phone-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
# Portable CPU installation. For NVIDIA GPU support, install the matching PyTorch build first.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 2. Prepare your dataset

Keep your dataset outside this repository if it is private. The training script expects one directory containing matching image/label pairs:

```text
/path/to/dataset/
├── image_001.jpg
├── image_001.txt
├── image_002.png
└── image_002.txt
```

Each `.txt` file must use standard YOLO detection annotations: one object per line as `class_id x_center y_center width height`, with normalized coordinates. Image filenames may end in `.jpg`, `.jpeg`, or `.png`. Class IDs must be contiguous and start at zero.

## 3. Run inference

```bash
yolo predict model=models/yolo11n_fold5_best.pt source=/path/to/image-or-video
```

Predictions are written to `runs/detect/` by default.

## 4. Run cross-validation training

Run the original five-model, five-fold protocol:

```bash
python yolo_cross_validator.py --dataset /path/to/dataset --output outputs
```

For a quicker experiment, train only YOLO11-Nano with three folds and ten epochs:

```bash
python yolo_cross_validator.py --dataset /path/to/dataset --output outputs --models yolo11n.pt --folds 3 --epochs 10
```

Use `python yolo_cross_validator.py --help` for every option. The default protocol uses seed 42, deterministic execution, 5 folds, 100 epochs, 640-pixel images, batch size 16, and all five YOLO11 variants. Outputs are created only in the directory passed to `--output`.

## Outputs

Training artifacts are written only to the directory passed to `--output`:

- `summary.csv`: mean and standard deviation of precision, recall, mAP@0.5, and mAP@0.5:0.95 across completed folds for each model.
- `runs/<model>/fold_<n>/results.csv`: epoch-level metrics for each run.
- `runs/<model>/fold_<n>/confusion_matrix.png` and `confusion_matrix_normalized.png`: confusion matrices produced by Ultralytics during validation.

The reported confusion matrices are computed from the dataset supplied at runtime; no private data or historical experiment outputs are included in this repository.

## 5. Generate publication-ready tables and confusion matrix

After training, generate tables using the completed fold metrics and a normalized confusion matrix from real validation predictions:

```bash
python generate_report.py --results outputs --benchmark
```

The command writes these files under `outputs/report/`:

- `table_2_detection_performance.csv` and `.png`: mean ± standard deviation for mAP@0.5, mAP@0.5:0.95, precision, and recall.
- `table_3_computational_characteristics.csv` and `.png`: measured throughput, average checkpoint size, and average training duration.
- `confusion_matrix_normalized.png`: a publication-ready matrix based on a real validation fold.

`--benchmark` measures throughput on the current hardware. Use `--matrix-model`, `--matrix-fold`, `--device`, `--imgsz`, and `--batch` to change the evaluation setup.
