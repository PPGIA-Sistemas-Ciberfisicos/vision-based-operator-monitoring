import pandas as pd
from sklearn.metrics import classification_report
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Path configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "manual_annotations.csv")

# Display-name mapping
label_display_names = {
    'NORMAL': 'Normal Operation',
    'POSTURE_DEVIATION': 'Posture Deviation',
    'HANDS_OFF_CONTROLS': 'Hands Off Controls',
    'OPERATOR_ABSENT': 'Operator Absence'
}

def save_table_image(metrics_by_class, output_path):
    """Render the article-style classification metrics table as a PNG."""
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    rows = [
        [name, "{:.2f}".format(values["Precision"]), "{:.2f}".format(values["Recall"]),
         "{:.2f}".format(values["F1-score"]), str(values["Support"])]
        for name, values in metrics_by_class.items()
    ]
    figure, axis = plt.subplots(figsize=(9, 2.7))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_text_props(weight="bold")
    figure.text(0.5, 0.04, "Table 1: Classification Metrics by Behavior Class", ha="center", fontsize=13)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# 1. Load CSV data
try:
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['ground_truth', 'prediction']):
        raise ValueError("CSV must contain columns 'ground_truth' and 'prediction'.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# 2. Preprocess data
df['ground_truth'] = df['ground_truth'].str.upper().str.strip()
df['prediction'] = df['prediction'].str.upper().str.strip()

# Merge PHONE_DETECTED into HANDS_OFF_CONTROLS, as in the published evaluation.
df['ground_truth'] = df['ground_truth'].replace('PHONE_DETECTED', 'HANDS_OFF_CONTROLS')
df['prediction'] = df['prediction'].replace('PHONE_DETECTED', 'HANDS_OFF_CONTROLS')

# 3. Generate report
try:
    report = classification_report(
        df['ground_truth'],
        df['prediction'],
        labels=['NORMAL', 'POSTURE_DEVIATION', 'HANDS_OFF_CONTROLS', 'OPERATOR_ABSENT'],
        target_names=['NORMAL', 'POSTURE_DEVIATION', 'HANDS_OFF_CONTROLS', 'OPERATOR_ABSENT'],
        output_dict=True
    )
except Exception as e:
    print(f"Error generating report: {e}")
    print("\nUnique values in ground_truth:", df['ground_truth'].unique())
    print("Unique values in prediction:", df['prediction'].unique())
    exit()

# 4. Format output
formatted_report = {}
for label, metrics in report.items():
    if label in label_display_names:
        formatted_report[label_display_names[label]] = {
            'Precision': round(metrics['precision'], 2),
            'Recall': round(metrics['recall'], 2),
            'F1-score': round(metrics['f1-score'], 2),
            'Support': int(metrics['support'])
        }

# 5. Display results
print("\n{:<20} {:<10} {:<10} {:<10} {:<10}".format(
    'Class', 'Precision', 'Recall', 'F1-score', 'Support'))
print("-" * 60)

for class_name, metrics in formatted_report.items():
    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
        class_name,
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1-score'],
        metrics['Support']))
table_image_path = os.path.join(output_dir, "table_1_classification_metrics.png")
save_table_image(formatted_report, table_image_path)
print(f"Classification table image saved to: {table_image_path}")


# Additional metrics
print("\nAdditional Metrics:")
print(f"Overall Accuracy: {report['accuracy']:.2f}")
print(f"Macro F1-score: {report['macro avg']['f1-score']:.2f}")
print(f"Weighted F1-score: {report['weighted avg']['f1-score']:.2f}")

# 6. Save results to a file
output_path = os.path.join(output_dir, "classification_report.txt")
with open(output_path, 'w') as f:
    f.write("Classification Report (PHONE_DETECTED merged into HANDS_OFF_CONTROLS)\n")
    f.write("-" * 60 + "\n")
    for class_name, metrics in formatted_report.items():
        f.write(f"{class_name:<20} {metrics['Precision']:<10.2f} {metrics['Recall']:<10.2f} "
                f"{metrics['F1-score']:<10.2f} {metrics['Support']:<10}\n")
    f.write("\nAdditional Metrics:\n")
    f.write(f"Overall Accuracy: {report['accuracy']:.2f}\n")
    f.write(f"Macro F1-score: {report['macro avg']['f1-score']:.2f}\n")
    f.write(f"Weighted F1-score: {report['weighted avg']['f1-score']:.2f}\n")
    f.write(f"\nTotal records analyzed: {len(df)}\n")

print(f"\nReport saved to: {output_path}")

# 7. Generate normalized confusion matrix
labels = ['NORMAL', 'POSTURE_DEVIATION', 'HANDS_OFF_CONTROLS', 'OPERATOR_ABSENT']
display_labels = ['Normal Operation', 'Posture Deviation', 'Hands Off Controls', 'Operator Absence']

# Confusion matrix (row normalized)
cm = confusion_matrix(df['ground_truth'], df['prediction'], labels=labels, normalize='true')

# Plot the matrix
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix', fontsize=20, pad=20)

# Add values to the matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

plt.xticks(np.arange(len(display_labels)), display_labels, rotation=45, fontsize=16, ha='right')
plt.yticks(np.arange(len(display_labels)), display_labels, fontsize=16)
plt.ylabel('True Label', fontsize=18, labelpad=20)
plt.xlabel('Predicted Label', fontsize=18, labelpad=20)
plt.tight_layout()

# Save image
image_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight', transparent=False)
plt.close()

print(f"Confusion matrix image saved to: {image_path}")
