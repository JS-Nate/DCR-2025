import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime

# === Prompt for scenario name ===
scenario_name = input("Enter the scenario name for this posture detection session: ").strip().replace(" ", "_")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"posture_detection-{scenario_name}-{timestamp}.csv"

# === Initialize MediaPipe pose ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# === Setup CSV writer ===
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame", "left_shoulder_x", "left_shoulder_y",
    "right_shoulder_x", "right_shoulder_y",
    "nose_x", "nose_y", "nose_z",
    "shoulder_diff", "head_forward", "posture",
    "bad_posture_streak", "depth_change", "movement_status"
])

# === Configurable parameters ===
MOVEMENT_DEPTH_THRESHOLD = 0.1  # Significant z-depth change
POSTURE_STREAK_THRESHOLD = 30   # Frames of continuous bad posture
MOVEMENT_SMOOTHING_FRAMES = 5

# === Tracking variables ===
frame_count = 0
prev_nose_z = None
bad_posture_streak = 0
recent_nose_z = []

def evaluate_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    ls = np.array([left_shoulder.x, left_shoulder.y])
    rs = np.array([right_shoulder.x, right_shoulder.y])
    nose_np = np.array([nose.x, nose.y])
    nose_z = nose.z  # Depth from camera (more negative = farther away)

    shoulder_diff = abs(ls[1] - rs[1])
    midpoint = (ls + rs) / 2
    head_forward = nose_np[0] - midpoint[0]

    posture = "Good"
    if shoulder_diff > 0.05:
        posture = "Bad - Slouching"
    elif head_forward > 0.05:
        posture = "Bad - Forward Head"

    return posture, ls, rs, nose_np, nose_z, shoulder_diff, head_forward

# === Start webcam and pose tracking ===
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Evaluate posture
            posture, ls, rs, nose_np, nose_z, shoulder_diff, head_forward = evaluate_posture(results.pose_landmarks.landmark)

            # Bad posture streak logic
            if "Bad" in posture:
                bad_posture_streak += 1
            else:
                bad_posture_streak = 0

            # Depth movement detection
            recent_nose_z.append(nose_z)
            if len(recent_nose_z) > MOVEMENT_SMOOTHING_FRAMES:
                recent_nose_z.pop(0)

            avg_z = np.mean(recent_nose_z)
            depth_change = abs(avg_z - prev_nose_z) if prev_nose_z is not None else 0
            prev_nose_z = avg_z

            if depth_change > MOVEMENT_DEPTH_THRESHOLD:
                movement_status = "Significant Depth Movement"
            elif bad_posture_streak >= POSTURE_STREAK_THRESHOLD:
                movement_status = "Prolonged Bad Posture"
            else:
                movement_status = "Stable"

            # Display info
            cv2.putText(image, f"Posture: {posture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if "Good" in posture else (0, 0, 255), 2)

            cv2.putText(image, f"Status: {movement_status}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Log to CSV
            csv_writer.writerow([
                frame_count,
                ls[0], ls[1],
                rs[0], rs[1],
                nose_np[0], nose_np[1], nose_z,
                shoulder_diff, head_forward, posture,
                bad_posture_streak, depth_change, movement_status
            ])

        cv2.imshow('Posture & Movement Monitor', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
