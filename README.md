# Vision-Based Operator Monitoring for Safety in Mining Teleoperation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: Mobile Phone Detection](https://img.shields.io/badge/Paper-Mobile_Phone_Detection-blue.svg)](https://doi.org/10.1007/978-3-032-32023-0_50)

This repository preserves the original operator-monitoring implementation used
for the article evaluation.

The complementary mobile-phone-detection study, including its released model
and training/reporting code, is available in
[mobile-phone-detection/](mobile-phone-detection/).

This directory preserves the final workflow used for the article evaluation, translated to English and made portable. It does not include the private videos, manually annotated CSV, generated frames, experiment results, or site-specific ROI.

All five scripts resolve generated artifacts relative to their own location.
They therefore write to [`outputs/`](outputs/) at the root of this papers
directory, regardless of the current working directory from which a script is
started.

## System architecture

![Architecture of the proposed operator-monitoring system](assets/system_architecture.png)

*Figure 1 from the article: input preprocessing, parallel posture and operator-detection pipelines, state classification, and alert prioritization.*

## Workflow

1. `1_configure_roi.py` selects the operator region of interest (ROI) from a representative video.
2. `2_monitor_operator.py` runs the final classifier and exports sampled frames by predicted state.
3. `3_annotate_predictions.py` supports manual ground-truth annotation of exported frames.
4. `4_evaluate_predictions.py` produces the metrics table and normalized confusion matrix. During evaluation, `PHONE_DETECTED` is merged into `HANDS_OFF_CONTROLS`, matching the published evaluation procedure.
5. `5_validate_head_pose.py` generates the head-tilt validation graph used in the article.

The final historical implementation detects five internal states: `NORMAL`, `POSTURE_DEVIATION`, `HANDS_OFF_CONTROLS`, `OPERATOR_ABSENT`, and `PHONE_DETECTED`.

## Setup

Tested with Python 3.12 on Linux. Python 3.10 to 3.12 is recommended. `tkinter` must be available because steps 1 and 3 use a desktop interface. On Debian/Ubuntu, install it with `sudo apt install python3-tk` if it is missing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Portable CPU installation. For NVIDIA GPU support, install the matching PyTorch build first.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Run

Create an ROI configuration from a representative video. If `outputs/roi_config.json` already contains a valid video path, the script reuses it and displays the saved ROI. Press `V` in the ROI window to select a different video:

```bash
python 1_configure_roi.py
```

The dialog writes `outputs/roi_config.json`. Create a `videos/` directory in this directory and add the videos you are authorized to process. Supported formats are MP4, AVI, MOV, and MKV. If that directory has no supported videos, the script opens a folder picker. Then run:

```bash
python 2_monitor_operator.py
```

The script exports sampled frames to `outputs/frames_by_state/`. `yolo11x.pt` is downloaded to `outputs/` if it is not already available locally.

## Evaluate manually annotated data

Run the annotation interface from this directory:

```bash
python 3_annotate_predictions.py
```

It creates `outputs/manual_annotations.csv` with `file`, `prediction`, and `ground_truth` columns. Assign labels with keys `1` to `5`; press `0` to skip a frame. To generate the article-style metrics and normalized confusion matrix:

```bash
python 4_evaluate_predictions.py
```

The output files are `outputs/classification_report.txt`, `outputs/table_1_classification_metrics.png`, and `outputs/confusion_matrix.png`. All generated artifacts are kept in `outputs/` and ignored by Git.

## Head-pose validation

The fifth step is based on the historical `analise_comportamento2.py` script and generates the final 60-frame graph with the 20-degree threshold shown in the article:

```bash
python 5_validate_head_pose.py
```

It updates `outputs/head_pose_validation/head_tilt_last_60_frames.png` and `outputs/head_pose_validation/current_roi.png` during execution, and writes `outputs/head_pose_validation/head_tilt_measurements.csv` when it stops. Press `Q`, `Esc`, or close the preview window to stop safely. Add `--save-video` to also export an annotated video.

## Reproducibility note

These scripts retain the final implementation that produced the annotated evaluation workflow. The paper describes some parameters differently (for example, a 20° posture threshold and YOLO-based absence detection); this repository does not claim that those details were implemented by the historical final script.
