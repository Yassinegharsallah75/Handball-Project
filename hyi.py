import cv2
import math

current_point = 1
current_points = []
all_measurements = []
paused = True
playback_speed = 30
known_distance = None
current_frame_number = 0
calibration_done = False


def click_event(event, x, y, flags, param):
    """Handle mouse clicks to select points."""
    global current_point, current_points, all_measurements, known_distance

    if event == cv2.EVENT_LBUTTONDOWN and paused:
        if not calibration_done:
            current_points.append((x, y, current_frame_number))
            print(f"Calibration point {len(current_points)} selected at frame {current_frame_number}")
            if len(current_points) == 2:
                calibrate_distance()
                current_points = []
            return

        current_points.append((x, y, current_frame_number))
        print(f"Point {current_point} selected at frame {current_frame_number}")
        if current_point == 2:
            calculate_speed()
            current_point = 1
            current_points = []
        else:
            current_point = 2


def calibrate_distance():
    """Calculate pixel-to-meter ratio based on known distance."""
    global known_distance, calibration_done

    if len(current_points) != 2:
        return

    (x1, y1, _), (x2, y2, _) = current_points
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    while True:
        try:
            real_distance = float(input("Enter the REAL distance between these two points in meters: "))
            if real_distance <= 0:
                print("Distance must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    known_distance = pixel_distance / real_distance
    calibration_done = True

    print(f"\nCalibration complete! {pixel_distance:.1f} pixels = {real_distance} meters")
    print(f"Ratio: {known_distance:.2f} pixels/meter\n")
    print("Now you can measure shot speeds. Select point 1 (start) and point 2 (end)")


def calculate_speed():
    """Calculate and display speed between two points."""
    if len(current_points) != 2 or not known_distance:
        return

    (x1, y1, frame1), (x2, y2, frame2) = current_points
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_distance = pixel_distance / known_distance

    frame_diff = abs(frame2 - frame1)
    time_sec = frame_diff / fps
    speed_mps = real_distance / time_sec
    speed_kmph = speed_mps * 3.6

    measurement = {
        'points': current_points.copy(),
        'speed_mps': speed_mps,
        'speed_kmph': speed_kmph,
        'time_sec': time_sec,
        'distance': real_distance
    }
    all_measurements.append(measurement)

    print("\n=== NEW SHOT MEASUREMENT ===")
    print(f"Frame {frame1} to {frame2}")
    print(f"Distance: {real_distance:.2f} meters")
    print(f"Time: {time_sec:.3f} seconds")
    print(f"Speed: {speed_mps:.2f} m/s")
    print(f"Speed: {speed_kmph:.2f} km/h")
    print("----------------------------\n")


video_path = r"C:\Users\ghars\Downloads\Andersson.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
print(f"Video FPS: {fps}")

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame from video")
    cap.release()
    exit()

cv2.namedWindow("Handball Speed")
cv2.setMouseCallback("Handball Speed", click_event)

print("\n=== CALIBRATION NEEDED ===")
print("1. Pause video (SPACE)")
print("2. Click on two points that you know the REAL distance between")
print("3. You'll be asked to enter the actual distance in meters")
print("4. After calibration, you can measure speeds\n")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    display_frame = frame.copy()

    for i, (x, y, frame_num) in enumerate(current_points):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(display_frame, (x, y), 8, color, -1)
        cv2.putText(display_frame, str(i + 1), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    status = "PAUSED" if paused else "PLAYING"
    cv2.putText(display_frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame: {current_frame_number}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not calibration_done:
        cv2.putText(display_frame, "CALIBRATION: Click 2 known points", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display_frame, f"Next point: {current_point}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(display_frame, "SPACE=pause/play, Q=quit",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Handball Speed", display_frame)

    key = cv2.waitKey(playback_speed if not paused else 0) & 0xFF
    if key == ord(' '):
        paused = not paused
    elif key == ord('q'):
        break

if calibration_done:
    print("\n=== ALL MEASUREMENTS ===")
    for i, m in enumerate(all_measurements, 1):
        (x1, y1, f1), (x2, y2, f2) = m['points']
        print(f"\nMeasurement {i}:")
        print(f"Frame {f1} to {f2}")
        print(f"Distance: {m['distance']:.2f} m")
        print(f"Time: {m['time_sec']:.3f} s")
        print(f"Speed: {m['speed_mps']:.2f} m/s")
        print(f"Speed: {m['speed_kmph']:.2f} km/h")

cap.release()
cv2.destroyAllWindows()
print("Program ended")
