import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "models/yolo11n.pt"
VIDEO_PATH = "videos/test_video.mp4"
OUTPUT_PATH = "results/detected_video.mp4"

ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

os.makedirs("results", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        device="cpu",
        conf=0.35,
        verbose=False
    )

    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in ANIMAL_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            animal_name = ANIMAL_CLASSES[class_id]

            label = f"{animal_name} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out.write(frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print()
print("Detection completed.")
print(f"Output saved to: {OUTPUT_PATH}")