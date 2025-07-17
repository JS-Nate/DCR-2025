import cv2
import mediapipe as mp
import numpy as np
import csv
import os
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
    "nose_x", "nose_y", "shoulder_diff", "head_forward", "posture"
])

# === Define posture evaluation function ===
def evaluate_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    ls = np.array([left_shoulder.x, left_shoulder.y])
    rs = np.array([right_shoulder.x, right_shoulder.y])
    nose_np = np.array([nose.x, nose.y])

    shoulder_diff = abs(ls[1] - rs[1])
    midpoint = (ls + rs) / 2
    head_forward = nose_np[0] - midpoint[0]

    posture = "Good"
    if shoulder_diff > 0.05:
        posture = "Bad - Slouching"
    elif head_forward > 0.05:
        posture = "Bad - Forward Head"

    return posture, ls, rs, nose_np, shoulder_diff, head_forward

# === Start webcam and pose tracking ===
cap = cv2.VideoCapture(0)
frame_count = 0

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

            posture, ls, rs, nose_np, shoulder_diff, head_forward = evaluate_posture(results.pose_landmarks.landmark)

            cv2.putText(image, f"Posture: {posture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Good" in posture else (0, 0, 255), 2)

            csv_writer.writerow([
                frame_count,
                ls[0], ls[1],
                rs[0], rs[1],
                nose_np[0], nose_np[1],
                shoulder_diff, head_forward, posture
            ])

        cv2.imshow('Posture Tracker', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
