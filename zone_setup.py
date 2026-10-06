import cv2

VIDEO_PATH = "videos/test_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video width:", width)
print("Video height:", height)
print("Total frames:", total_frames)

# Move to a frame around the middle of the video
middle_frame = total_frames // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

ret, frame = cap.read()

cap.release()

if not ret:
    print("Error: Could not read video frame.")
    exit()

print("Displaying frame:", middle_frame)

# -----------------------------------------
# Resize frame to fit the screen
# -----------------------------------------

display_width = 1200

scale = display_width / width

display_height = int(height * scale)

display_frame = cv2.resize(
    frame,
    (display_width, display_height)
)

print("Display size:", display_width, "x", display_height)
print("Scale:", scale)

# -----------------------------------------
# Store clicked points
# -----------------------------------------

points = []

window_name = "Select Railway Danger Zone"


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        # Convert displayed coordinates
        # back to original video coordinates

        original_x = int(x / scale)
        original_y = int(y / scale)

        points.append((original_x, original_y))

        print(
            f"Point {len(points)}: "
            f"({original_x}, {original_y})"
        )


# Create window
cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    display_width,
    display_height
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# -----------------------------------------
# Display loop
# -----------------------------------------

while True:

    display = display_frame.copy()

    # Draw points
    for point in points:

        display_x = int(point[0] * scale)
        display_y = int(point[1] * scale)

        cv2.circle(
            display,
            (display_x, display_y),
            7,
            (0, 0, 255),
            -1
        )

    # Draw lines
    if len(points) >= 2:

        for i in range(len(points) - 1):

            x1 = int(points[i][0] * scale)
            y1 = int(points[i][1] * scale)

            x2 = int(points[i + 1][0] * scale)
            y2 = int(points[i + 1][1] * scale)

            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # Close polygon
    if len(points) >= 3:

        x1 = int(points[-1][0] * scale)
        y1 = int(points[-1][1] * scale)

        x2 = int(points[0][0] * scale)
        y2 = int(points[0][1] * scale)

        cv2.line(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # Show instructions
    cv2.putText(
        display,
        "Click points | R = Reset | ENTER = Finish | ESC = Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        window_name,
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # ENTER
    if key == 13:

        if len(points) >= 3:

            print("\n==============================")
            print("Danger zone points:")
            print(points)
            print("==============================")

            break

        else:

            print("Please select at least 3 points.")

    # R = Reset
    elif key == ord("r"):

        points.clear()

        print("Points reset.")

    # ESC
    elif key == 27:

        print("Cancelled.")

        points.clear()

        break


cv2.destroyAllWindows()