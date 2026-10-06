# ============================================================
# YOLOv8n - COMPLETE EXPERIMENTAL EVIDENCE
# Railway Animal Intrusion Detection
# ============================================================

import os
import json
import csv
import time
import shutil
import subprocess
from pathlib import Path

import yaml
import numpy as np

# ------------------------------------------------------------
# 1. INSTALL / IMPORT
# ------------------------------------------------------------

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing Ultralytics...")
    subprocess.check_call(["pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO

import ultralytics
import torch

print("=" * 70)
print("YOLOv8n EXPERIMENTAL EVIDENCE")
print("=" * 70)

print("Ultralytics :", ultralytics.__version__)
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    device = 0
else:
    print("GPU         : CPU")
    device = "cpu"


# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

DATASET = Path("/kaggle/input/datasets/eswar03/railway-animal-intrusion")

PROJECT = Path("/kaggle/working/Railway_Animal_Project")

MODEL_DIR = PROJECT / "models" / "YOLOv8n" / "finetuned"

EVIDENCE_DIR = PROJECT / "experimental_evidence" / "YOLOv8n"

TRAIN_DIR = EVIDENCE_DIR / "training"
METRICS_DIR = EVIDENCE_DIR / "metrics"
GRAPHS_DIR = EVIDENCE_DIR / "graphs"
CONFUSION_DIR = EVIDENCE_DIR / "confusion_matrix"

for folder in [
    PROJECT,
    MODEL_DIR,
    EVIDENCE_DIR,
    TRAIN_DIR,
    METRICS_DIR,
    GRAPHS_DIR,
    CONFUSION_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

print("\nProject directory:")
print(PROJECT)


# ------------------------------------------------------------
# 3. DATASET VERIFICATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

classes = [
    "Bear",
    "Buffalo",
    "Camel",
    "Cat",
    "Cow",
    "Deer",
    "Dog",
    "Donkey",
    "Elephant",
    "Leopard",
    "Lion",
    "Tiger"
]

train_images = DATASET / "train" / "images"
train_labels = DATASET / "train" / "labels"

valid_images = DATASET / "valid" / "images"
valid_labels = DATASET / "valid" / "labels"

test_images = DATASET / "test" / "images"
test_labels = DATASET / "test" / "labels"

print("Dataset:", DATASET)

print("Train images :", len(list(train_images.glob("*"))))
print("Valid images :", len(list(valid_images.glob("*"))))
print("Test images  :", len(list(test_images.glob("*"))))

print("Classes      :", len(classes))

for i, name in enumerate(classes):
    print(f"{i:2d}: {name}")


# ------------------------------------------------------------
# 4. CREATE LOCAL DATASET YAML
# ------------------------------------------------------------

yaml_path = Path("/kaggle/working/data_local.yaml")

data_config = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(classes),
    "names": classes
}

with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, sort_keys=False)

print("\nDataset YAML:")
print(yaml_path)

with open(yaml_path) as f:
    print(f.read())


# ------------------------------------------------------------
# 5. EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

config = {
    "model": "yolov8n.pt",
    "dataset": str(yaml_path),

    "epochs": 50,
    "imgsz": 640,
    "batch": 16,

    "optimizer": "AdamW",
    "lr0": 0.000625,

    "seed": 0,
    "deterministic": True,

    "patience": 50,
    "close_mosaic": 10,

    "workers": 4,
    "device": device,
    "amp": True,

    "confidence_threshold": 0.25,
    "iou_threshold": 0.50
}

config_path = EVIDENCE_DIR / "training_config.json"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("\nExperiment configuration saved:")
print(config_path)


# ------------------------------------------------------------
# 6. LOAD YOLOv8n
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING YOLOv8n")
print("=" * 70)

model = YOLO("yolov8n.pt")

print("YOLOv8n loaded successfully.")


# ------------------------------------------------------------
# 7. TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING YOLOv8n TRAINING")
print("=" * 70)

training_start = time.perf_counter()

train_results = model.train(
    data=str(yaml_path),

    epochs=50,
    imgsz=640,
    batch=16,

    optimizer="AdamW",
    lr0=0.000625,

    seed=0,
    deterministic=True,

    patience=50,
    close_mosaic=10,

    workers=4,
    device=device,
    amp=True,

    project=str(TRAIN_DIR),
    name="yolov8n_50_epochs",

    plots=True,
    save=True,
    verbose=True
)

training_wall_time = time.perf_counter() - training_start

print("\nTraining completed.")
print(f"Training wall time: {training_wall_time:.2f} seconds")
print(f"Training wall time: {training_wall_time / 60:.2f} minutes")


# ------------------------------------------------------------
# 8. LOCATE BEST MODEL
# ------------------------------------------------------------

best_candidates = [
    TRAIN_DIR / "yolov8n_50_epochs" / "weights" / "best.pt",
    Path("/kaggle/working") / "runs" / "detect" / "yolov8n_50_epochs" / "weights" / "best.pt"
]

best_pt = None

for candidate in best_candidates:
    if candidate.exists():
        best_pt = candidate
        break

if best_pt is None:
    matches = list(Path("/kaggle/working").rglob("best.pt"))

    if matches:
        best_pt = matches[-1]

if best_pt is None:
    raise FileNotFoundError("best.pt was not found.")

print("\nBest model:")
print(best_pt)


# ------------------------------------------------------------
# 9. COPY BEST / LAST MODEL
# ------------------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

final_best = MODEL_DIR / "best.pt"

shutil.copy2(best_pt, final_best)

last_pt = best_pt.parent / "last.pt"

if last_pt.exists():
    shutil.copy2(last_pt, MODEL_DIR / "last.pt")

print("\nSaved:")
print(final_best)


# ------------------------------------------------------------
# 10. VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

best_model = YOLO(str(final_best))

validation_start = time.perf_counter()

val_results = best_model.val(
    data=str(yaml_path),
    split="val",
    imgsz=640,
    batch=16,
    device=device,
    conf=0.25,
    iou=0.50,
    plots=True,
    save_json=True,
    project=str(EVIDENCE_DIR),
    name="validation"
)

validation_time = time.perf_counter() - validation_start

print("\nValidation completed.")


# ------------------------------------------------------------
# 11. TEST EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_start = time.perf_counter()

test_results = best_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=640,
    batch=16,
    device=device,
    conf=0.25,
    iou=0.50,
    plots=True,
    save_json=True,
    project=str(EVIDENCE_DIR),
    name="test"
)

test_wall_time = time.perf_counter() - test_start

print("\nTest completed.")
print(f"Test wall time: {test_wall_time:.2f} seconds")


# ------------------------------------------------------------
# 12. EXTRACT ULTRALYTICS METRICS
# ------------------------------------------------------------

try:
    precision = float(test_results.box.mp)
except:
    precision = float(test_results.results_dict.get("metrics/precision(B)", 0))

try:
    recall = float(test_results.box.mr)
except:
    recall = float(test_results.results_dict.get("metrics/recall(B)", 0))

try:
    map50 = float(test_results.box.map50)
except:
    map50 = float(test_results.results_dict.get("metrics/mAP50(B)", 0))

try:
    map50_95 = float(test_results.box.map)
except:
    map50_95 = float(test_results.results_dict.get("metrics/mAP50-95(B)", 0))

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0.0

print("\nTest metrics")
print("-" * 50)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map50_95:.4f}")


# ------------------------------------------------------------
# 13. FIXED-THRESHOLD DETECTION EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIXED-THRESHOLD EVALUATION")
print("=" * 70)

print("Confidence threshold : 0.25")
print("IoU threshold        : 0.50")

TP = 0
FP = 0
FN = 0

# ------------------------------------------------------------
# Helper: IoU
# ------------------------------------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)

    intersection = intersection_w * intersection_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ------------------------------------------------------------
# NOTE:
# The following computes image-level detection counts from
# YOLO predictions and ground-truth labels.
# ------------------------------------------------------------

test_files = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    test_files.extend(test_images.glob(ext))

print("Test images:", len(test_files))

for index, image_path in enumerate(test_files):

    label_path = test_labels / f"{image_path.stem}.txt"

    ground_truths = []

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Ground truth will be converted later
                ground_truths.append(
                    [cls, xc, yc, w, h]
                )

    results = best_model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.50,
        imgsz=640,
        device=device,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes_pred = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(
            boxes,
            classes_pred,
            confidences
        ):

            predictions.append(
                {
                    "box": box.tolist(),
                    "class": int(cls),
                    "confidence": float(conf)
                }
            )

    # --------------------------------------------------------
    # Convert GT normalized coordinates to pixel coordinates
    # --------------------------------------------------------

    image_height, image_width = result.orig_shape

    gt_boxes = []

    for gt in ground_truths:

        cls, xc, yc, w, h = gt

        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height
        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height

        gt_boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "class": cls
            }
        )

    matched_gt = set()

    # Highest confidence first
    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    for pred in predictions:

        best_iou = 0
        best_gt = None

        for gt_index, gt in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            iou_value = calculate_iou(
                pred["box"],
                gt["box"]
            )

            if iou_value > best_iou:
                best_iou = iou_value
                best_gt = gt_index

        if best_gt is not None and best_iou >= 0.50:

            if (
                pred["class"]
                == gt_boxes[best_gt]["class"]
            ):
                TP += 1
                matched_gt.add(best_gt)

            else:
                # Wrong class prediction
                FP += 1
                FN += 1
                matched_gt.add(best_gt)

        else:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1}/{len(test_files)}")


# ------------------------------------------------------------
# 14. CALCULATE FIXED-THRESHOLD METRICS
# ------------------------------------------------------------

if TP + FP > 0:
    fixed_precision = TP / (TP + FP)
else:
    fixed_precision = 0

if TP + FN > 0:
    fixed_recall = TP / (TP + FN)
else:
    fixed_recall = 0

if fixed_precision + fixed_recall > 0:
    fixed_f1 = (
        2 * fixed_precision * fixed_recall
        / (fixed_precision + fixed_recall)
    )
else:
    fixed_f1 = 0

if TP + FP + FN > 0:
    accuracy_proxy = TP / (TP + FP + FN)
else:
    accuracy_proxy = 0

print("\nFixed-threshold results")
print("-" * 50)
print("TP              :", TP)
print("FP              :", FP)
print("FN              :", FN)
print(f"Precision       : {fixed_precision:.4f}")
print(f"Recall          : {fixed_recall:.4f}")
print(f"F1-score        : {fixed_f1:.4f}")
print(f"Accuracy Proxy  : {accuracy_proxy:.4f}")
print(f"Accuracy Proxy  : {accuracy_proxy * 100:.2f}%")


# ------------------------------------------------------------
# 15. MODEL INFORMATION
# ------------------------------------------------------------

try:
    parameter_count = sum(
        p.numel()
        for p in best_model.model.parameters()
    )
except:
    parameter_count = None

try:
    model_size_mb = final_best.stat().st_size / (1024 * 1024)
except:
    model_size_mb = None


# ------------------------------------------------------------
# 16. SAVE FINAL METRICS JSON
# ------------------------------------------------------------

final_metrics = {

    "project":
        "AI-Based Railway Animal Intrusion Detection "
        "with Animal-Specific False Alarm Filtering",

    "component":
        "Animal Detection",

    "model":
        "YOLOv8n",

    "checkpoint":
        str(final_best),

    "pretrained_weights":
        "yolov8n.pt",

    "ultralytics":
        ultralytics.__version__,

    "torch":
        torch.__version__,

    "gpu":
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",

    "dataset":
        str(DATASET),

    "classes":
        classes,

    "training": {

        "epochs": 50,

        "imgsz": 640,

        "batch": 16,

        "optimizer": "AdamW",

        "lr0": 0.000625,

        "seed": 0,

        "deterministic": True,

        "device": str(device),

        "amp": True,

        "training_wallclock_seconds":
            round(training_wall_time, 2),

        "training_wallclock_minutes":
            round(training_wall_time / 60, 2)
    },

    "test_evaluation": {

        "images_evaluated":
            len(test_files),

        "confidence_threshold":
            0.25,

        "iou_threshold":
            0.50,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "mAP50":
            map50,

        "mAP50_95":
            map50_95,

        "test_wallclock_seconds":
            round(test_wall_time, 2)
    },

    "fixed_threshold_evaluation": {

        "TP": TP,

        "FP": FP,

        "FN": FN,

        "precision":
            fixed_precision,

        "recall":
            fixed_recall,

        "f1":
            fixed_f1,

        "accuracy_proxy":
            accuracy_proxy,

        "accuracy_proxy_percent":
            accuracy_proxy * 100
    },

    "model_information": {

        "parameters":
            parameter_count,

        "best_pt_size_mb":
            round(model_size_mb, 2)
            if model_size_mb
            else None
    },

    "experimental_status":
        "YOLOv8n selected as the final animal detection model."
}

metrics_json = METRICS_DIR / "YOLOv8n_final_metrics.json"

with open(metrics_json, "w") as f:
    json.dump(final_metrics, f, indent=4)

print("\nMetrics saved:")
print(metrics_json)


# ------------------------------------------------------------
# 17. SAVE CSV
# ------------------------------------------------------------

metrics_csv = METRICS_DIR / "YOLOv8n_final_metrics.csv"

csv_data = {

    "model": "YOLOv8n",

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "accuracy_proxy": accuracy_proxy,

    "mAP50": map50,

    "mAP50_95": map50_95,

    "TP": TP,

    "FP": FP,

    "FN": FN,

    "training_time_minutes":
        training_wall_time / 60,

    "test_time_seconds":
        test_wall_time,

    "parameters":
        parameter_count,

    "best_pt_size_mb":
        model_size_mb
}

with open(metrics_csv, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_data.keys()
    )

    writer.writeheader()
    writer.writerow(csv_data)

print("CSV saved:")
print(metrics_csv)


# ------------------------------------------------------------
# 18. COPY TRAINING GRAPHS
# ------------------------------------------------------------

training_results_dir = (
    TRAIN_DIR /
    "yolov8n_50_epochs"
)

if training_results_dir.exists():

    for file in training_results_dir.iterdir():

        if file.suffix.lower() in [
            ".png",
            ".csv",
            ".jpg",
            ".jpeg"
        ]:

            destination = GRAPHS_DIR / file.name

            try:
                shutil.copy2(file, destination)
            except:
                pass


# ------------------------------------------------------------
# 19. SAVE EXPERIMENT SUMMARY
# ------------------------------------------------------------

summary = f"""
============================================================
YOLOv8n FINAL EXPERIMENTAL EVIDENCE
============================================================

Project:
AI-Based Railway Animal Intrusion Detection with
Animal-Specific False Alarm Filtering

Selected Model:
YOLOv8n

Training:
Epochs              : 50
Image Size          : 640
Batch Size          : 16
Optimizer           : AdamW
Learning Rate       : 0.000625
Seed                : 0
GPU                 : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

Training Time:
{training_wall_time:.2f} seconds
{training_wall_time / 60:.2f} minutes

TEST RESULTS
------------------------------------------------------------
Precision           : {precision:.4f}
Recall              : {recall:.4f}
F1-score            : {f1:.4f}
Accuracy Proxy      : {accuracy_proxy:.4f}
Accuracy Proxy (%)  : {accuracy_proxy * 100:.2f}%
mAP50               : {map50:.4f}
mAP50-95            : {map50_95:.4f}

FIXED THRESHOLD
------------------------------------------------------------
Confidence          : 0.25
IoU                 : 0.50

TP                  : {TP}
FP                  : {FP}
FN                  : {FN}

Fixed Precision     : {fixed_precision:.4f}
Fixed Recall        : {fixed_recall:.4f}
Fixed F1            : {fixed_f1:.4f}

TEST TIME
------------------------------------------------------------
{test_wall_time:.2f} seconds

MODEL
------------------------------------------------------------
Parameters          : {parameter_count}
Best.pt size        : {model_size_mb:.2f} MB

FINAL DECISION
------------------------------------------------------------
YOLOv8n was selected as the final animal detection model
for the railway animal intrusion detection system.

NEXT PIPELINE
------------------------------------------------------------
Railway Video
      |
      v
YOLOv8n Animal Detection
      |
      v
Track Detection
      |
      v
Distance / Movement Analysis
      |
      v
Animal-Specific False Alarm Filtering
      |
      v
Risk Decision
      |
      v
Alert
============================================================
"""

summary_file = EVIDENCE_DIR / "YOLOv8n_experimental_summary.txt"

with open(summary_file, "w") as f:
    f.write(summary)

print(summary)


# ------------------------------------------------------------
# 20. FINAL DIRECTORY SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENTAL EVIDENCE COMPLETED")
print("=" * 70)

print("\nEvidence directory:")
print(EVIDENCE_DIR)

print("\nImportant files:")

print("1. Best model:")
print(final_best)

print("\n2. Metrics JSON:")
print(metrics_json)

print("\n3. Metrics CSV:")
print(metrics_csv)

print("\n4. Experiment summary:")
print(summary_file)

print("\n5. Training graphs:")
print(GRAPHS_DIR)

print("\n6. Confusion/validation/test results:")
print(EVIDENCE_DIR)

print("\n" + "=" * 70)
print("YOLOv8n EXPERIMENT FINISHED")
print("=" * 70)