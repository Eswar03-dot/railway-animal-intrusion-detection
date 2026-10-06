import cv2

from danger_zone import DANGER_ZONE


VIDEO_PATH = "videos/test_video.mp4"
OUTPUT_PATH = "results/danger_zone_test.mp4"


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Draw danger-zone polygon
    cv2.polylines(
        frame,
        [DANGER_ZONE],
        True,
        (0, 0, 255),
        3
    )

    # Add label
    cv2.putText(
        frame,
        "RAILWAY DANGER ZONE",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    out.write(frame)


cap.release()
out.release()


print("Danger-zone video created.")
print(f"Saved to: {OUTPUT_PATH}")