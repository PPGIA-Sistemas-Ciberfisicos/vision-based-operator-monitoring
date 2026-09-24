import argparse
import csv
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import KFold
import numpy as np
from ultralytics import YOLO
import torch

class YOLOCrossValidator:
    def __init__(self, dataset_path, output_path, n_splits=5, models=None):
        """
        Initializes the YOLO cross-validator

        Args:
            dataset_path: Path to the dataset (directory containing images and labels)
            output_path: Path where results are saved
            n_splits: Number of folds for cross-validation.
            models: Optional list of pretrained YOLO model filenames.
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.n_splits = n_splits
        self.models = models or ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

        # Create directories
        self.output_path.mkdir(parents=True, exist_ok=True)

    def prepare_dataset_structure(self):
        """Prepares the dataset directory structure."""
        print("Preparing dataset structure...")

        # Collect all images
        image_files = list(self.dataset_path.glob("*.jpg")) + \
                     list(self.dataset_path.glob("*.jpeg")) + \
                     list(self.dataset_path.glob("*.png"))

        # Keep only images with a corresponding .txt file
        valid_images = []
        for img in image_files:
            label_file = img.with_suffix('.txt')
            if label_file.exists():
                valid_images.append(img)

        print(f"Found {len(valid_images)} images with corresponding labels")
        return valid_images

    def create_fold_datasets(self, images):
        """Creates datasets for every fold."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        images_array = np.array(images)

        fold_configs = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(images_array)):
            print(f"\nPreparing fold {fold + 1}/{self.n_splits}...")

            train_images = images_array[train_idx]
            val_images = images_array[val_idx]

            # Create the directory structure for this fold
            fold_path = self.output_path / f"fold_{fold + 1}"
            train_img_path = fold_path / "images" / "train"
            train_lbl_path = fold_path / "labels" / "train"
            val_img_path = fold_path / "images" / "val"
            val_lbl_path = fold_path / "labels" / "val"

            for path in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
                path.mkdir(parents=True, exist_ok=True)

            # Copy training images and labels
            for img in train_images:
                shutil.copy2(img, train_img_path / img.name)
                label = img.with_suffix('.txt')
                shutil.copy2(label, train_lbl_path / label.name)

            # Copy validation images and labels
            for img in val_images:
                shutil.copy2(img, val_img_path / img.name)
                label = img.with_suffix('.txt')
                shutil.copy2(label, val_lbl_path / label.name)

            # Create the YAML configuration file
            yaml_config = {
                'path': str(fold_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': self.get_num_classes(train_images[0].with_suffix('.txt')),
                'names': self.get_class_names()
            }

            yaml_path = fold_path / "data.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)

            fold_configs.append(yaml_path)
            print(f"Fold {fold + 1}: {len(train_images)} training, {len(val_images)} validation")

        return fold_configs

    def get_num_classes(self, label_file):
        """Returns the number of classes in the dataset."""
        classes = set()
        for label in self.dataset_path.glob("*.txt"):
            try:
                with open(label, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            classes.add(class_id)
            except:
                continue
        return len(classes) if classes else 1

    def get_class_names(self):
        """Returns class names (adjust to match your dataset)."""
        num_classes = self.get_num_classes(next(self.dataset_path.glob("*.txt")))
        return [f"class_{i}" for i in range(num_classes)]

    def train_model(self, model_name, fold_config, fold_num, epochs=100, imgsz=640, batch=16):
        """Trains a YOLO model on a specific fold."""
        print(f"\n{'='*60}")
        print(f"Training {model_name} - Fold {fold_num}")
        print(f"{'='*60}")

        # Load model
        model = YOLO(model_name)

        # Project and experiment names
        project_name = self.output_path / "runs" / model_name.replace('.pt', '')
        experiment_name = f"fold_{fold_num}"

        # Train
        results = model.train(
            data=str(fold_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project_name),
            name=experiment_name,
            patience=50,
            save=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=8,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            val=True
        )

        # Validate
        metrics = model.val()

        return results, metrics

    def run_cross_validation(self, epochs=100, imgsz=640, batch=16):
        """Runs cross-validation training for every model."""
        print("\n" + "="*60)
        print("STARTING YOLO11 CROSS-VALIDATION")
        print("="*60)

        # Prepare dataset
        images = self.prepare_dataset_structure()
        if len(images) == 0:
            raise ValueError("No valid images were found in the dataset!")

        # Create folds
        fold_configs = self.create_fold_datasets(images)

        # Results
        all_results = {}

        # Train every model
        for model_name in self.models:
            print(f"\n\n{'#'*60}")
            print(f"MODEL: {model_name}")
            print(f"{'#'*60}")

            model_results = []

            # Train on every fold
            for fold_num, fold_config in enumerate(fold_configs, 1):
                try:
                    results, metrics = self.train_model(
                        model_name,
                        fold_config,
                        fold_num,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch
                    )

                    model_results.append({
                        'fold': fold_num,
                        'results': results,
                        'metrics': metrics
                    })

                    print(f"\n✓ Fold {fold_num} completed!")
                    print(f"mAP50: {metrics.box.map50:.4f}")
                    print(f"mAP50-95: {metrics.box.map:.4f}")

                except Exception as e:
                    print(f"\n✗ Error in fold {fold_num}: {str(e)}")
                    continue

            all_results[model_name] = model_results

        # Final summary
        self.print_summary(all_results)
        self.save_summary_table(all_results)

        return all_results

    def save_summary_table(self, all_results):
        """Writes aggregate cross-validation metrics to a CSV table."""
        summary_path = self.output_path / "summary.csv"
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "model", "completed_folds", "precision_mean", "precision_std",
                "recall_mean", "recall_std", "map50_mean", "map50_std",
                "map50_95_mean", "map50_95_std"
            ])
            writer.writeheader()
            for model_name, results in all_results.items():
                if not results:
                    continue
                precision = [item["metrics"].box.mp for item in results]
                recall = [item["metrics"].box.mr for item in results]
                map50 = [item["metrics"].box.map50 for item in results]
                map50_95 = [item["metrics"].box.map for item in results]
                writer.writerow({
                    "model": model_name,
                    "completed_folds": len(results),
                    "precision_mean": np.mean(precision),
                    "precision_std": np.std(precision),
                    "recall_mean": np.mean(recall),
                    "recall_std": np.std(recall),
                    "map50_mean": np.mean(map50),
                    "map50_std": np.std(map50),
                    "map50_95_mean": np.mean(map50_95),
                    "map50_95_std": np.std(map50_95),
                })
        print(f"Summary table saved to: {summary_path}")

    def print_summary(self, all_results):
        """Prints a summary of the results."""
        print("\n\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        for model_name, results in all_results.items():
            if not results:
                continue

            print(f"\n{model_name}:")

            map50_scores = [r['metrics'].box.map50 for r in results]
            map_scores = [r['metrics'].box.map for r in results]

            print(f"  mAP50     - Mean: {np.mean(map50_scores):.4f} ± {np.std(map50_scores):.4f}")
            print(f"  mAP50-95  - Mean: {np.mean(map_scores):.4f} ± {np.std(map_scores):.4f}")
            print(f"  Completed folds: {len(results)}/{self.n_splits}")


def parse_args():
    """Parses command-line arguments for cross-validation training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 models with deterministic K-fold cross-validation."
    )
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Directory containing image files and matching YOLO .txt labels.")
    parser.add_argument("--output", default="outputs", type=Path,
                        help="Directory where folds, checkpoints, and metrics are written.")
    parser.add_argument("--folds", default=5, type=int,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Training epochs per fold (default: 100).")
    parser.add_argument("--imgsz", default=640, type=int,
                        help="Square input size in pixels (default: 640).")
    parser.add_argument("--batch", default=16, type=int,
                        help="Batch size (default: 16).")
    parser.add_argument("--models", nargs="+",
                        default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
                        help="Pretrained model files to compare.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validator = YOLOCrossValidator(args.dataset, args.output, args.folds, args.models)
    validator.run_cross_validation(args.epochs, args.imgsz, args.batch)
    print(f"\n✓ Training completed! Results saved to: {args.output}")
