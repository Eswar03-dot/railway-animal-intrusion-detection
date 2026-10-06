# ============================================================
# YOLOv8n - TRAINING EXPERIMENTAL EVIDENCE
# Railway Animal Intrusion Detection
# ============================================================

import os
import json
import time
import shutil
from pathlib import Path

import yaml
import torch
import ultralytics
from ultralytics import YOLO

# ============================================================
# 1. PROJECT / DATASET PATHS
# ============================================================

DATASET = Path(
    "/kaggle/input/datasets/eswar03/railway-animal-intrusion"
)

PROJECT = Path(
    "/kaggle/working/Railway_Animal_Project"
)

EVIDENCE = PROJECT / "experimental_evidence" / "YOLOv8n"

TRAIN_EVIDENCE = EVIDENCE / "training"

TRAIN_EVIDENCE.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. DATASET YAML
# ============================================================

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

yaml_path = Path(
    "/kaggle/working/data_local.yaml"
)

data_config = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 12,
    "names": classes
}

with open(yaml_path, "w") as f:
    yaml.dump(
        data_config,
        f,
        sort_keys=False
    )

# ============================================================
# 3. ENVIRONMENT INFORMATION
# ============================================================

print("=" * 70)
print("YOLOv8n TRAINING EXPERIMENTAL EVIDENCE")
print("=" * 70)

print("\nEnvironment")
print("-" * 70)

print("Ultralytics :", ultralytics.__version__)
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.cuda.is_available())

if torch.cuda.is_available():

    gpu_name = torch.cuda.get_device_name(0)

    print("GPU         :", gpu_name)

    try:
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU Memory  : {gpu_memory:.2f} GB"
        )

    except:
        gpu_memory = None

    device = 0

else:

    gpu_name = "CPU"
    gpu_memory = None
    device = "cpu"


# ============================================================
# 4. TRAINING CONFIGURATION
# ============================================================

training_config = {

    "project":
        "AI-Based Railway Animal Intrusion Detection "
        "with Animal-Specific False Alarm Filtering",

    "component":
        "Animal Detection",

    "model":
        "YOLOv8n",

    "pretrained_weights":
        "yolov8n.pt",

    "dataset":
        str(DATASET),

    "dataset_yaml":
        str(yaml_path),

    "epochs": 50,

    "image_size": 640,

    "batch_size": 16,

    "optimizer":
        "AdamW",

    "learning_rate":
        0.000625,

    "seed": 0,

    "deterministic": True,

    "patience": 50,

    "close_mosaic": 10,

    "workers": 4,

    "device":
        device,

    "amp": True,

    "classes":
        classes
}

config_file = (
    TRAIN_EVIDENCE /
    "training_configuration.json"
)

with open(config_file, "w") as f:

    json.dump(
        training_config,
        f,
        indent=4
    )

print("\nTraining configuration saved:")
print(config_file)


# ============================================================
# 5. LOAD YOLOv8n
# ============================================================

print("\n" + "=" * 70)
print("LOADING YOLOv8n")
print("=" * 70)

model = YOLO("yolov8n.pt")

print("Model loaded successfully.")


# ============================================================
# 6. START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("Model       : YOLOv8n")
print("Epochs      : 50")
print("Image size  : 640")
print("Batch size  : 16")
print("Optimizer   : AdamW")
print("Learning rate:", 0.000625)
print("Device      :", device)

training_start = time.perf_counter()

results = model.train(

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

    project=str(TRAIN_EVIDENCE),

    name="YOLOv8n_50_epochs",

    save=True,

    plots=True,

    verbose=True
)

training_end = time.perf_counter()

training_time_seconds = (
    training_end -
    training_start
)

training_time_minutes = (
    training_time_seconds / 60
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Training time: "
    f"{training_time_seconds:.2f} seconds"
)

print(
    f"Training time: "
    f"{training_time_minutes:.2f} minutes"
)


# ============================================================
# 7. LOCATE TRAINING DIRECTORY
# ============================================================

run_directory = (
    TRAIN_EVIDENCE /
    "YOLOv8n_50_epochs"
)

print("\nTraining directory:")
print(run_directory)


# ============================================================
# 8. LOCATE BEST AND LAST WEIGHTS
# ============================================================

best_weight = (
    run_directory /
    "weights" /
    "best.pt"
)

last_weight = (
    run_directory /
    "weights" /
    "last.pt"
)

print("\nWeights")

if best_weight.exists():

    print(
        "best.pt found:",
        best_weight
    )

else:

    print("WARNING: best.pt not found")


if last_weight.exists():

    print(
        "last.pt found:",
        last_weight
    )

else:

    print("WARNING: last.pt not found")


# ============================================================
# 9. COPY FINAL WEIGHTS TO CLEAN LOCATION
# ============================================================

FINAL_MODEL_DIR = (
    PROJECT /
    "results" /
    "YOLOv8n"
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if best_weight.exists():

    shutil.copy2(
        best_weight,
        FINAL_MODEL_DIR /
        "best.pt"
    )

if last_weight.exists():

    shutil.copy2(
        last_weight,
        FINAL_MODEL_DIR /
        "last.pt"
    )

print("\nFinal model files saved to:")
print(FINAL_MODEL_DIR)


# ============================================================
# 10. READ TRAINING RESULTS CSV
# ============================================================

results_csv = (
    run_directory /
    "results.csv"
)

training_history = []

if results_csv.exists():

    import csv

    with open(
        results_csv,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            training_history.append(row)

    print("\nTraining history found.")
    print(
        "Epoch records:",
        len(training_history)
    )

else:

    print(
        "\nWARNING: results.csv not found."
    )


# ============================================================
# 11. FIND BEST EPOCH
# ============================================================

best_epoch = None
best_map50_95 = None

if training_history:

    possible_columns = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95"
    ]

    metric_column = None

    for column in possible_columns:

        if column in training_history[0]:

            metric_column = column
            break

    if metric_column:

        values = []

        for row in training_history:

            try:

                values.append(
                    float(
                        row[metric_column]
                    )
                )

            except:

                values.append(
                    float("-inf")
                )

        best_index = int(
            max(
                range(len(values)),
                key=lambda i: values[i]
            )
        )

        best_epoch = (
            best_index + 1
        )

        best_map50_95 = values[
            best_index
        ]

        print("\nBest epoch:")
        print(best_epoch)

        print(
            "Best validation mAP50-95:",
            round(
                best_map50_95,
                4
            )
        )


# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================

if results_csv.exists():

    shutil.copy2(
        results_csv,
        TRAIN_EVIDENCE /
        "YOLOv8n_training_history.csv"
    )

    print(
        "\nTraining history copied."
    )


# ============================================================
# 13. COPY TRAINING GRAPHS
# ============================================================

GRAPH_DIR = (
    EVIDENCE /
    "training_graphs"
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

graph_extensions = [
    ".png",
    ".jpg",
    ".jpeg"
]

graph_count = 0

if run_directory.exists():

    for file in run_directory.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in graph_extensions
        ):

            try:

                shutil.copy2(
                    file,
                    GRAPH_DIR /
                    file.name
                )

                graph_count += 1

            except:
                pass

print(
    "\nTraining graphs copied:",
    graph_count
)


# ============================================================
# 14. MODEL INFORMATION
# ============================================================

final_best = (
    FINAL_MODEL_DIR /
    "best.pt"
)

if final_best.exists():

    model_size_mb = (
        final_best.stat().st_size
        / (1024 * 1024)
    )

else:

    model_size_mb = None


try:

    trained_model = YOLO(
        str(final_best)
    )

    parameters = sum(
        p.numel()
        for p in
        trained_model.model.parameters()
    )

except:

    parameters = None


# ============================================================
# 15. SAVE TRAINING EVIDENCE JSON
# ============================================================

training_evidence = {

    "project":
        "AI-Based Railway Animal Intrusion Detection "
        "with Animal-Specific False Alarm Filtering",

    "component":
        "Animal Detection",

    "model":
        "YOLOv8n",

    "pretrained_weights":
        "yolov8n.pt",

    "dataset":
        str(DATASET),

    "dataset_yaml":
        str(yaml_path),

    "training": {

        "epochs": 50,

        "epochs_completed":
            len(training_history)
            if training_history
            else None,

        "image_size": 640,

        "batch_size": 16,

        "optimizer":
            "AdamW",

        "learning_rate":
            0.000625,

        "seed": 0,

        "deterministic":
            True,

        "patience":
            50,

        "close_mosaic":
            10,

        "workers":
            4,

        "device":
            device,

        "amp":
            True,

        "best_epoch":
            best_epoch,

        "best_validation_mAP50_95":
            best_map50_95,

        "training_wallclock_seconds":
            round(
                training_time_seconds,
                2
            ),

        "training_wallclock_minutes":
            round(
                training_time_minutes,
                2
            )
    },

    "environment": {

        "ultralytics":
            ultralytics.__version__,

        "pytorch":
            torch.__version__,

        "cuda":
            torch.cuda.is_available(),

        "gpu":
            gpu_name,

        "gpu_memory_gb":
            gpu_memory
    },

    "model_information": {

        "best_pt":
            str(final_best),

        "parameters":
            parameters,

        "best_pt_size_mb":
            round(
                model_size_mb,
                2
            )
            if model_size_mb
            else None
    },

    "evidence_files": {

        "training_history":
            str(
                TRAIN_EVIDENCE /
                "YOLOv8n_training_history.csv"
            ),

        "configuration":
            str(config_file),

        "training_graphs":
            str(GRAPH_DIR),

        "best_model":
            str(final_best)
    },

    "status":
        "YOLOv8n 50-epoch training completed."
}

training_json = (
    TRAIN_EVIDENCE /
    "YOLOv8n_training_evidence.json"
)

with open(
    training_json,
    "w"
) as f:

    json.dump(
        training_evidence,
        f,
        indent=4
    )

print(
    "\nTraining evidence JSON saved:"
)

print(training_json)


# ============================================================
# 16. HUMAN-READABLE TRAINING REPORT
# ============================================================

report = f"""
============================================================
YOLOv8n TRAINING EXPERIMENTAL EVIDENCE
============================================================

PROJECT
AI-Based Railway Animal Intrusion Detection with
Animal-Specific False Alarm Filtering

COMPONENT
Animal Detection

MODEL
YOLOv8n

PRETRAINED WEIGHTS
yolov8n.pt

TRAINING CONFIGURATION
------------------------------------------------------------
Epochs              : 50
Image Size          : 640
Batch Size          : 16
Optimizer           : AdamW
Learning Rate       : 0.000625
Seed                : 0
Deterministic       : True
Patience            : 50
Close Mosaic        : 10
Workers             : 4
AMP                 : True

ENVIRONMENT
------------------------------------------------------------
Ultralytics         : {ultralytics.__version__}
PyTorch             : {torch.__version__}
CUDA                : {torch.cuda.is_available()}
GPU                 : {gpu_name}

TRAINING RESULT
------------------------------------------------------------
Epochs Completed    : {len(training_history) if training_history else "N/A"}
Best Epoch          : {best_epoch}
Best mAP50-95       : {best_map50_95}

Training Time
Seconds             : {training_time_seconds:.2f}
Minutes             : {training_time_minutes:.2f}

MODEL
------------------------------------------------------------
Parameters          : {parameters}
Best.pt Size        : {model_size_mb:.2f} MB

MODEL LOCATION
------------------------------------------------------------
{final_best}

EVIDENCE LOCATION
------------------------------------------------------------
{EVIDENCE}

FILES GENERATED
------------------------------------------------------------
training_configuration.json
YOLOv8n_training_history.csv
YOLOv8n_training_evidence.json
training graphs
best.pt
last.pt

============================================================
"""

report_file = (
    TRAIN_EVIDENCE /
    "YOLOv8n_training_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(report)

print(report)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("=" * 70)
print("TRAINING EVIDENCE COMPLETE")
print("=" * 70)

print("\nEvidence folder:")
print(EVIDENCE)

print("\nBest model:")
print(final_best)

print("\nTraining history:")
print(
    TRAIN_EVIDENCE /
    "YOLOv8n_training_history.csv"
)

print("\nTraining evidence JSON:")
print(training_json)

print("\nTraining report:")
print(report_file)

print("\nTraining graphs:")
print(GRAPH_DIR)

print("\n" + "=" * 70)