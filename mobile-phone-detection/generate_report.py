"""Create publication-ready tables and a confusion matrix from training outputs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

MODEL_LABELS = {
    "yolo11n": "YOLO11-Nano", "yolo11s": "YOLO11-Small",
    "yolo11m": "YOLO11-Medium", "yolo11l": "YOLO11-Large",
    "yolo11x": "YOLO11-XLarge",
}


def final_metrics(results_dir):
    """Read the last epoch of every completed training run."""
    values = defaultdict(list)
    for csv_path in sorted(results_dir.glob("runs/yolo11*/fold_*/results.csv")):
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            continue
        row = {key.strip(): value.strip() for key, value in rows[-1].items()}
        model = csv_path.parents[1].name
        checkpoint = csv_path.parent / "weights" / "best.pt"
        values[model].append({
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
            "hours": float(row["time"]) / 3600,
            "checkpoint": checkpoint,
            "data": results_dir / f"fold_{csv_path.parent.name.split('_')[-1]}" / "data.yaml",
        })
    return values


def mean_std(items, key):
    data = [item[key] for item in items]
    return float(np.mean(data)), float(np.std(data))


def write_csv(path, headers, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def table_image(path, title, headers, rows):
    fig, axis = plt.subplots(figsize=(12, 0.65 * (len(rows) + 2)))
    axis.axis("off")
    table = axis.table(cellText=[[row[h] for h in headers] for row in rows],
                       colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axis.set_title(title, fontweight="bold", pad=18)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def benchmark(checkpoint, data, imgsz, batch, device):
    """Measure inference throughput on the supplied validation dataset."""
    metrics = YOLO(str(checkpoint)).val(data=str(data), imgsz=imgsz, batch=batch,
                                        device=device, workers=0, plots=False, verbose=False)
    milliseconds = metrics.speed["inference"]
    return 1000 / milliseconds if milliseconds else 0.0, metrics


def confusion_matrix(metrics, output_path):
    """Render a normalized confusion matrix using real validation predictions."""
    matrix = metrics.confusion_matrix.matrix.astype(float)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
    names = list(metrics.names.values()) + ["background"]
    fig, axis = plt.subplots(figsize=(max(8, len(names) * 1.7), max(6, len(names) * 1.4)))
    image = axis.imshow(normalized, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    threshold = normalized.max() / 2 if normalized.size else 0
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center",
                      color="white" if normalized[row, column] > threshold else "black")
    axis.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names,
             yticklabels=names, ylabel="True label", xlabel="Predicted label",
             title="Normalized Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready YOLO11 reports.")
    parser.add_argument("--results", required=True, type=Path, help="Output directory from yolo_cross_validator.py.")
    parser.add_argument("--output", type=Path, help="Report directory (default: <results>/report).")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark each model to populate throughput in Table 3.")
    parser.add_argument("--matrix-model", default="yolo11n", choices=MODEL_LABELS, help="Model used for the confusion matrix.")
    parser.add_argument("--matrix-fold", default=1, type=int, help="Fold used for the confusion matrix.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0 or cpu.")
    args = parser.parse_args()
    report_dir = args.output or args.results / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = final_metrics(args.results)
    if not metrics:
        raise FileNotFoundError("No results.csv files were found under <results>/runs/yolo11*/fold_*.")

    table2 = []
    table3 = []
    for model, items in sorted(metrics.items()):
        label = MODEL_LABELS.get(model, model)
        p, pstd = mean_std(items, "precision")
        r, rstd = mean_std(items, "recall")
        m50, m50std = mean_std(items, "map50")
        m95, m95std = mean_std(items, "map50_95")
        hours, _ = mean_std(items, "hours")
        table2.append({"Model Variant": label, "mAP@0.5 (%)": f"{m50*100:.2f} ± {m50std*100:.2f}",
                       "mAP@0.5:0.95 (%)": f"{m95*100:.2f} ± {m95std*100:.2f}",
                       "Precision (%)": f"{p*100:.2f} ± {pstd*100:.2f}",
                       "Recall (%)": f"{r*100:.2f} ± {rstd*100:.2f}"})
        available = [item["checkpoint"] for item in items if item["checkpoint"].exists()]
        size = np.mean([item.stat().st_size / 1_000_000 for item in available]) if available else float("nan")
        fps = float("nan")
        if args.benchmark and available:
            fps, _ = benchmark(available[0], items[0]["data"], args.imgsz, args.batch, args.device)
        table3.append({"Variant": label, "Throughput (FPS)": "n/a" if np.isnan(fps) else f"{fps:.1f}",
                       "Size (MB)": "n/a" if np.isnan(size) else f"{size:.1f}",
                       "Training (h)": f"{hours:.2f}",
                       "Efficiency": "n/a"})

    measured_fps = [float(row["Throughput (FPS)"]) for row in table3 if row["Throughput (FPS)"] != "n/a"]
    if measured_fps:
        fastest_fps = max(measured_fps)
        for row in table3:
            if row["Throughput (FPS)"] != "n/a":
                row["Efficiency"] = "{:.2f}".format(float(row["Throughput (FPS)"]) / fastest_fps)

    table2_headers = list(table2[0])
    table3_headers = list(table3[0])
    write_csv(report_dir / "table_2_detection_performance.csv", table2_headers, table2)
    write_csv(report_dir / "table_3_computational_characteristics.csv", table3_headers, table3)
    table_image(report_dir / "table_2_detection_performance.png", "Table 2: Detection performance (mean ± std)", table2_headers, table2)
    table_image(report_dir / "table_3_computational_characteristics.png", "Table 3: Computational characteristics", table3_headers, table3)

    matrix_model = args.matrix_model if args.matrix_model in metrics else next(iter(metrics))
    selected = metrics[matrix_model][args.matrix_fold - 1]
    if not selected["checkpoint"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {selected['checkpoint']}")
    _, validation = benchmark(selected["checkpoint"], selected["data"], args.imgsz, args.batch, args.device)
    confusion_matrix(validation, report_dir / "confusion_matrix_normalized.png")
    print(f"Report written to: {report_dir}")


if __name__ == "__main__":
    main()
