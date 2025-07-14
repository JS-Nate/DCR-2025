import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define posture evaluation function
def evaluate_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Convert to numpy for calculations
    ls = np.array([left_shoulder.x, left_shoulder.y])
    rs = np.array([right_shoulder.x, right_shoulder.y])
    nose = np.array([nose.x, nose.y])

    # Check if shoulders are aligned (y-difference small)
    shoulder_diff = abs(ls[1] - rs[1])

    # Check if head is forward (nose too far ahead of shoulders)
    midpoint = (ls + rs) / 2
    head_forward = nose[0] - midpoint[0]

    posture = "Good"
    if shoulder_diff > 0.05:
        posture = "Bad - Slouching"
    elif head_forward > 0.05:
        posture = "Bad - Forward Head"

    return posture

# Start webcam and pose tracking
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw and evaluate
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            posture = evaluate_posture(results.pose_landmarks.landmark)
            cv2.putText(image, f"Posture: {posture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Good" in posture else (0, 0, 255), 2)

        cv2.imshow('Posture Tracker', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
