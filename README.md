# AI-Based Railway Animal Intrusion Detection with Animal-Specific False Alarm Filtering

> A computer-vision-based railway safety system designed to detect animals near railway tracks, analyze their movement and proximity to the track, filter animal-specific false alarms, and generate alerts when an animal is likely to enter a dangerous railway zone.

**Project Status:** 🚧 In Development

**Current Completed Stage:** Animal Detection Model Development, Experimental Evaluation, and YOLOv8n Model Selection

---

# 📌 Project Overview

The **AI-Based Railway Animal Intrusion Detection with Animal-Specific False Alarm Filtering** project is an AI-based railway safety system designed to detect animals near railway tracks and determine whether they represent a potential intrusion risk.

The system uses computer vision and deep-learning-based object detection to identify animals from railway surveillance video.

Instead of generating an alert whenever an animal is detected, the proposed system aims to analyze:

- Animal location
- Distance from the railway track
- Animal movement
- Direction of movement
- Proximity to the danger zone
- Animal-specific behavior
- Probability of entering the railway track

The project combines:

- Computer Vision
- Object Detection
- YOLO-based Deep Learning
- Animal Detection
- Railway Track Detection
- Object Tracking
- Distance Analysis
- Movement Analysis
- False Alarm Filtering
- Risk Assessment
- Alert Generation

The current development stage focuses on the **animal detection component**, where multiple YOLO models were fine-tuned and experimentally evaluated.

---

# 🎯 Problem Statement

Animals such as cattle, elephants, deer, dogs, and other wildlife can enter or approach railway tracks.

Such incidents can create serious safety risks including:

- Train-animal collisions
- Animal injuries or deaths
- Train delays
- Railway operational disruption
- Potential damage to railway infrastructure
- Passenger and railway safety risks

Traditional surveillance systems may rely on manual monitoring or simple detection mechanisms.

A basic animal detector also has an important limitation:

> **Detecting an animal does not necessarily mean that the animal is going to enter the railway track.**

For example, an animal may be:

- Standing far away from the track
- Moving away from the track
- Outside the danger zone
- Moving parallel to the railway track
- Present near the railway area without an immediate intrusion risk

Generating an alert in all these cases can create excessive false alarms.

## Proposed Problem

The project aims to develop an intelligent railway surveillance system that can:

1. Detect animals.
2. Detect the railway track.
3. Determine the animal's position relative to the track.
4. Track animal movement.
5. Estimate distance and movement direction.
6. Apply animal-specific false-alarm filtering.
7. Determine the level of intrusion risk.
8. Generate an alert when necessary.

The main research direction is:

> **Can an AI-based computer vision system detect animals near railway tracks and distinguish genuine track-intrusion risks from harmless animal presence using movement, distance, and animal-specific false-alarm filtering?**

---

# 💡 Proposed Solution

The proposed system processes railway surveillance video and analyzes animals in relation to the railway track.

## Possible Inputs

- Railway surveillance video
- Recorded railway camera footage
- Railway track video
- Future live camera streams

## Processing

The system performs:

1. Animal detection
2. Railway track detection
3. Animal tracking
4. Distance estimation
5. Movement analysis
6. Danger-zone analysis
7. False-alarm filtering
8. Risk classification

## Output

The system is designed to produce:

- Detected animal
- Animal class
- Detection confidence
- Animal position
- Distance from railway track
- Movement direction
- Risk level
- Intrusion warning or alert

---

# 🏗️ High-Level Architecture

```text
                         RAILWAY VIDEO
                               |
                               v
                    +---------------------+
                    |    Video Input      |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |   Animal Detection  |
                    |      YOLOv8n        |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |   Track Detection   |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |   Animal Tracking   |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |  Distance Analysis  |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |  Movement Analysis  |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |   False Alarm       |
                    |     Filtering       |
                    +---------------------+
                               |
                               v
                    +---------------------+
                    |    Risk Decision    |
                    +---------------------+
                               |
                    +----------+----------+
                    |                     |
                 LOW RISK              HIGH RISK
                    |                     |
                 No Alert               ALERT
```

---

# 🧠 AI/ML Approach

The project is designed as a multi-stage computer-vision system.

## 1. Animal Detection

The first major component is automatic animal detection from railway video.

YOLO-based object detection models were used because they provide efficient object detection suitable for video-processing applications.

The following candidate models were fine-tuned and experimentally evaluated:

- YOLOv8n
- YOLOv10n
- YOLO11n

All three models were trained using the same railway animal dataset and a common experimental configuration.

After comparison, **YOLOv8n was selected as the final animal detection model**.

---

## 2. Railway Track Detection

The second component is railway track detection.

The purpose of this component is to identify the railway track or relevant track region so that the system can determine the animal's position relative to the track.

The project uses a separate track-detection approach rather than depending only on manually defined danger zones.

The detected track information is intended to support:

- Distance calculation
- Track proximity analysis
- Danger-zone assessment
- Intrusion-risk calculation

---

## 3. Animal Tracking

After an animal is detected, the system needs to determine how its position changes across video frames.

Tracking allows the system to analyze whether an animal is:

- Stationary
- Moving toward the track
- Moving away from the track
- Moving parallel to the track

Tracking is important because the system should analyze animal movement over time rather than treating every frame as an independent detection.

---

## 4. Distance and Movement Analysis

The system analyzes the relationship between the detected animal and railway track.

Important information includes:

- Animal position
- Track position
- Approximate distance
- Change in distance over time
- Movement direction
- Track proximity

This information is intended to help determine whether the animal is approaching a dangerous railway area.

---

## 5. Animal-Specific False Alarm Filtering

Animal-specific false alarm filtering is an important part of the proposed system.

The system should not simply follow:

```text
Animal detected
      ↓
Alert
```

Instead, the intended logic is:

```text
Animal detected
      ↓
Track proximity
      ↓
Movement analysis
      ↓
Risk assessment
      ↓
Animal-specific false-alarm filtering
      ↓
Alert only when intrusion risk is significant
```

Different animals may have different movement patterns and risk characteristics.

Therefore, the proposed system is designed to incorporate animal-specific filtering into the final risk-decision stage.

---

# 📊 Railway Animal Detection Dataset

The project uses the **Railway Animal Intrusion Detection dataset**.

| Property | Value |
|---|---:|
| Animal classes | **12** |
| Training images | **5,016** |
| Validation images | **358** |
| Test images | **352** |
| Total images | **5,726** |
| Training instances | **6,921** |
| Validation instances | **498** |
| Test instances | **467** |

## Animal Classes

| ID | Animal |
|---:|---|
| 0 | Bear |
| 1 | Buffalo |
| 2 | Camel |
| 3 | Cat |
| 4 | Cow |
| 5 | Deer |
| 6 | Dog |
| 7 | Donkey |
| 8 | Elephant |
| 9 | Leopard |
| 10 | Lion |
| 11 | Tiger |

The dataset contains bounding-box annotations suitable for YOLO-based object detection.

---

# 🔬 Current Experimental Results

Three YOLO models were fine-tuned and evaluated using a common experimental protocol.

| Model | Precision | Recall | F1 | Accuracy Proxy | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| **YOLOv8n** | **90.02%** | **85.87%** | **87.90%** | **78.41%** | 84.33% | 62.35% |
| YOLOv10n | 88.79% | 85.26% | 84.48% | 73.13% | **85.33%** | 64.30% |
| YOLO11n | 88.79% | 84.75% | 84.48% | 73.13% | 84.31% | **64.80%** |

## Accuracy Proxy

For this object-detection project, **Accuracy Proxy** is used instead of conventional classification accuracy.

```text
Accuracy Proxy = TP / (TP + FP + FN)
```

This metric is explicitly treated as an **object-detection accuracy proxy**.

---

# 🏆 Selected Model: YOLOv8n

**YOLOv8n** was selected as the final animal detection model.

The main reasons for selection are:

- Highest Precision: **90.02%**
- Highest Recall: **85.87%**
- Highest F1-score: **87.90%**
- Highest Accuracy Proxy: **78.41%**
- Strong mAP50: **84.33%**
- Strong overall detection performance

YOLOv10n and YOLO11n achieved higher mAP50-95 and faster evaluation times in some measurements. However, YOLOv8n provided the strongest overall Precision, Recall, F1-score, and Accuracy Proxy among the evaluated candidate models.

---

# 🧪 Final YOLOv8n Test Evaluation

The selected YOLOv8n model was evaluated on the test dataset.

| Metric | YOLOv8n |
|---|---:|
| Test images | **352** |
| Test instances | **467** |
| Precision | **90.02%** |
| Recall | **85.87%** |
| F1-score | **87.90%** |
| Accuracy Proxy | **78.41%** |
| mAP50 | **84.33%** |
| mAP50-95 | **62.35%** |
| Macro ROC-AUC | **96.82%** |
| Test wall time | **15.60 sec** |
| Training time | **69.10 min** |

## Fixed-Threshold Evaluation

The project uses the following fixed evaluation thresholds:

```text
Confidence threshold = 0.25
IoU threshold        = 0.50
```

Recorded detection counts:

```text
TP = 401
FP = 44.44
FN = 66
```

These values are used for the project's fixed-threshold Precision, Recall, F1-score, and Accuracy Proxy evaluation.

---

# ⚙️ Experimental Protocol

The YOLO models were evaluated using a common experimental configuration.

| Parameter | Value |
|---|---|
| Models | YOLOv8n, YOLOv10n, YOLO11n |
| Maximum epochs | **50** |
| Image size | **640 × 640** |
| Batch size | **16** |
| Optimizer | **AdamW** |
| Learning rate | **0.000625** |
| Seed | **0** |
| GPU | **NVIDIA Tesla T4** |
| Device | **GPU 0** |
| AMP | **Enabled** |
| Workers | **4** |
| Confidence threshold | **0.25** |
| IoU threshold | **0.50** |

The models were fine-tuned using the training split and evaluated using validation and test data.

---

# 🔬 Experimental Evidence

The repository contains experimental evidence generated during animal-detection model development.

## Training Evidence

The candidate models were trained for 50 epochs:

- YOLOv8n
- YOLOv10n
- YOLO11n

The training evidence includes:

- Training configuration
- Epoch-level training results
- Validation metrics
- Training duration
- Model weights
- Training curves
- Validation plots
- Confusion matrices

For the selected YOLOv8n model, the repository contains the training-evidence generation code:

```text
results/create_training_evidence.py
```

The experimental training configuration and evidence are intended to provide reproducibility of the model-development process.

---

# 📈 Training and Evaluation Graphs

The project generates visual evidence for model training and evaluation.

The evidence includes:

- Training loss curves
- Validation loss curves
- Precision curves
- Recall curves
- mAP50 curves
- mAP50-95 curves
- Confusion matrix
- Normalized confusion matrix

These visualizations are used to understand model learning behavior and evaluation performance.

---

# 📦 Consolidated Experimental Evidence

Machine-readable experimental results are stored inside the `results/` directory.

Important files include:

```text
results/
├── final_model_results.json
├── model_comparison.csv
├── model_comparision.json
├── yolov8n_test_clean_eval_summary.json
├── create_experimental_evidence.py
└── create_training_evidence.py
```

## Experimental Evidence Contains

- Training configuration
- Model comparison
- Precision
- Recall
- F1-score
- Accuracy Proxy
- mAP50
- mAP50-95
- ROC-AUC
- Training time
- Test time
- Model parameters
- Final model selection
- Test evaluation information

---

# 🧪 Evaluation Structure

The model-development process was divided into three major stages.

## 1. Training Evidence

Each candidate YOLO model was fine-tuned using the railway animal training dataset.

## 2. Model Comparison

The candidate models were compared using:

- Precision
- Recall
- F1-score
- Accuracy Proxy
- mAP50
- mAP50-95
- ROC-AUC
- Training time
- Test time
- Model complexity

## 3. Final Model Selection

Based on the recorded comparison, **YOLOv8n was selected as the final animal detection model**.

---

# 📋 Model Comparison

| Metric | YOLOv8n | YOLOv10n | YOLO11n |
|---|---:|---:|---:|
| Precision | **90.02%** | 88.79% | 88.79% |
| Recall | **85.87%** | 85.26% | 84.75% |
| F1 | **87.90%** | 84.48% | 84.48% |
| Accuracy Proxy | **78.41%** | 73.13% | 73.13% |
| mAP50 | 84.33% | **85.33%** | 84.31% |
| mAP50-95 | 62.35% | 64.30% | **64.80%** |
| ROC-AUC | 96.82% | 98.10% | **98.18%** |
| Training time | 69.10 min | 40.41 min | **34.90 min** |
| Test time | 15.60 sec | 9.37 sec | **9.19 sec** |
| Parameters | N/A | **2.27M** | 2.58M |
| GFLOPs | N/A | 6.6 | **6.4** |

## Final Decision

**Selected Model: YOLOv8n**

The final model selection prioritizes the strongest overall detection performance, particularly Precision, Recall, F1-score, and Accuracy Proxy, because reliable animal detection is important for the subsequent railway intrusion-risk analysis.

---

# 📈 Current Project Progress

## Completed

- ✅ Project architecture defined
- ✅ Railway animal detection dataset prepared
- ✅ Dataset classes verified
- ✅ YOLOv8n fine-tuned
- ✅ YOLOv10n fine-tuned
- ✅ YOLO11n fine-tuned
- ✅ Three-model comparison completed
- ✅ YOLOv8n selected as final animal detector
- ✅ Test evaluation completed
- ✅ Training evidence generation implemented
- ✅ Experimental metrics documented
- ✅ Model comparison CSV/JSON created
- ✅ Project source code organized
- ✅ Project pushed to GitHub

## In Progress

- 🚧 Integrating YOLOv8n into the railway video pipeline
- 🚧 Railway track detection
- 🚧 Animal tracking
- 🚧 Distance calculation
- 🚧 Movement analysis
- 🚧 Animal-specific false-alarm filtering
- 🚧 Risk-level decision system
- 🚧 Alert generation
- 🚧 Frontend/UI development
- 🚧 Complete end-to-end testing

---

# 🔬 Research Direction

The project investigates an integrated approach combining:

**Animal Detection + Track Detection + Tracking + Distance Analysis + Movement Analysis + Animal-Specific False Alarm Filtering + Risk Assessment**

The main research question is:

> **Can an AI-based computer vision system distinguish genuine railway animal intrusion risks from harmless animal presence near railway tracks by combining animal detection, track proximity, movement analysis, and animal-specific false-alarm filtering?**

The research contribution is therefore not limited to detecting an animal.

The main objective is to determine:

> **Whether the detected animal actually represents an intrusion risk.**

---

# 🚀 Planned Pipeline

The selected YOLOv8n model will be integrated into the broader railway safety system.

```text
Railway Video
      |
      v
YOLOv8n Animal Detection
      |
      v
Detected Animal + Confidence
      |
      v
Railway Track Detection
      |
      v
Animal Tracking
      |
      v
Distance Calculation
      |
      v
Movement / Direction Analysis
      |
      v
Animal-Specific False Alarm Filtering
      |
      v
Risk Assessment
      |
      +-----------------------+
      |                       |
      v                       v
   LOW RISK                HIGH RISK
      |                       |
      v                       v
   NO ALERT               ALERT
```

---

# 🗂️ Repository Structure

The current GitHub project is organized as follows:

```text
railway-animal-intrusion-detection/
│
├── README.md
├── requirements.txt
│
├── app.py
├── detection.py
├── tracking.py
├── risk_analysis.py
├── danger_zone.py
├── zone_setup.py
├── utils.py
│
├── test_danger_zone.py
├── test_pretrained_animal.py
│
├── models/
│
├── results/
│   ├── final_model_results.json
│   ├── model_comparison.csv
│   ├── model_comparision.json
│   ├── yolov8n_test_clean_eval_summary.json
│   ├── create_experimental_evidence.py
│   └── create_training_evidence.py
│
├── merged_results/
│   └── merge_zip_files.py
│
└── .gitignore
```

Large files such as trained `.pt` models, videos, virtual environments, and runtime outputs are intentionally excluded from the Git repository.

---

# 🛠️ Technology Stack

## AI / ML

- Python
- Ultralytics YOLO
- YOLOv8
- YOLOv10
- YOLO11
- PyTorch
- Computer Vision
- Object Detection
- Object Tracking
- Machine Learning

## Video Processing

- OpenCV
- FFmpeg

## Development

- VS Code
- Git
- GitHub
- Kaggle
- Python Virtual Environment

## Evaluation

- Precision
- Recall
- F1-score
- Accuracy Proxy
- mAP@50
- mAP@50-95
- ROC-AUC
- Confusion Matrix
- Training Curves

---

# ⚠️ Current Limitations

The complete railway intrusion-detection system is still under development.

Current limitations include:

- Animal detection is limited to the 12 classes represented in the dataset.
- Detection performance can vary with lighting and camera angle.
- Occluded animals may be difficult to detect.
- Visually similar animals may occasionally be misclassified.
- The current YOLOv8n model has an Accuracy Proxy of **78.41%**, which is below the project's desired 80% target.
- Distance estimation from ordinary video is an approximation unless suitable camera calibration information is available.
- Railway track detection still requires integration with the complete animal-detection pipeline.
- Animal tracking requires further testing under crowded or partially occluded scenes.
- Animal-specific false-alarm filtering is not yet fully integrated.
- Real-time live-camera deployment has not yet been completed.
- The complete end-to-end system has not yet been evaluated.

---

# 📌 Current Status

## Animal Detection Model

**Completed ✅**

Selected model:

> **YOLOv8n**

Final recorded performance:

- Precision: **90.02%**
- Recall: **85.87%**
- F1-score: **87.90%**
- Accuracy Proxy: **78.41%**
- mAP50: **84.33%**
- mAP50-95: **62.35%**
- Macro ROC-AUC: **96.82%**

## Overall Railway Animal Intrusion Detection System

**In Development 🚧**

The next major milestone is integrating the validated **YOLOv8n animal detector** with:

1. Railway track detection
2. Animal tracking
3. Distance analysis
4. Movement analysis
5. Animal-specific false-alarm filtering
6. Risk assessment
7. Alert generation
8. End-to-end video evaluation

---

# 📚 Project Evidence

The repository contains the code and machine-readable results required to document the animal-detection experiments.

Key evidence files include:

```text
results/create_training_evidence.py
results/create_experimental_evidence.py
results/final_model_results.json
results/model_comparison.csv
results/model_comparision.json
results/yolov8n_test_clean_eval_summary.json
```

The project also maintains the trained-model and video files separately from GitHub because of their large file sizes.

---

# 🔗 Project Repository

GitHub Repository:

**https://github.com/Eswar03-dot/railway-animal-intrusion-detection**

---

# 👨‍💻 Development

This project is being developed as an academic AI/ML project focused on railway safety, computer vision, animal detection, and intelligent false-alarm filtering.

The final goal is to develop a system that does not simply answer:

> **"Is there an animal?"**

but instead answers:

> **"Is this animal likely to enter the railway track, and should an alert be generated?"**
