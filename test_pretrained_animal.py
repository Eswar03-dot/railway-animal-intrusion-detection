from ultralytics import YOLO

# Load pretrained YOLO11 model
model = YOLO("models/yolo11n.pt")

# Test on railway video
results = model.predict(
    source="videos/test_video.mp4",
    save=True,
    conf=0.40
)

print("Animal detection test completed.")