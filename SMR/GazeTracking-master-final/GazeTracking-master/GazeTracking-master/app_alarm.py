import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import datetime
import pandas as pd
import threading
from playsound import playsound

# ---------------------------
# User Input & Setup
# ---------------------------
Operator_name = input("Please Enter your name Before starting the Scenario: ")
results = {"Eye Detection": [], "Time": []}
dtTm_now = datetime.datetime.now()
dt = dtTm_now.strftime("%Y-%m-%d_%H_%M_%S")

# ---------------------------
# Helper Functions
# ---------------------------
def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio (EAR) to determine eye closure.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_gaze_direction(eye_roi):
    """
    A simple approach to determine gaze direction.
    The eye region is thresholded and then split into left and right halves.
    The ratio of white pixels in each half helps decide the direction.
    """
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    h, w = thresh_eye.shape
    left_side = thresh_eye[0:h, 0:int(w/2)]
    right_side = thresh_eye[0:h, int(w/2):w]
    
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    
    # Avoid division by zero
    if left_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = right_white / left_white

    if gaze_ratio < 0.8:
        return "RIGHT"
    elif gaze_ratio > 1.2:
        return "LEFT"
    else:
        return "CENTER"

def play_alarm_sound():
    """
    Play the alarm sound from a file.
    """
    try:
        playsound("Alarm sound effect.mp3")
    except Exception as e:
        print("Error playing sound:", e)

# ---------------------------
# Initialize Dlib’s Face Detector and Landmark Predictor
# ---------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices for the eyes in dlib's 68-point model
(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

# Blink detection parameters
EYE_AR_THRESH = 0.2      # EAR threshold to indicate eye closure
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames below threshold to count a blink

blink_counter = 0
total_blinks = 0

# Variables for alarm when eyes remain closed for more than 5 seconds
blink_start_time = None
alarm_triggered = False

# ---------------------------
# Start Video Capture
# ---------------------------
cap = cv2.VideoCapture(0)  # Change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize for faster processing
    frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray, 0)
    
    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        
        # Extract eye regions
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        
        # Compute EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Blink detection for counting individual blinks
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0
        
        # ---------------------------
        # Continuous Eye Closure Alarm
        # ---------------------------
        if ear < EYE_AR_THRESH:
            if blink_start_time is None:
                blink_start_time = datetime.datetime.now()
            else:
                elapsed = (datetime.datetime.now() - blink_start_time).total_seconds()
                if elapsed >= 3 and not alarm_triggered:
                    alarm_triggered = True
                    # Play alarm sound in a separate thread
                    threading.Thread(target=play_alarm_sound, daemon=True).start()
                    cv2.putText(frame, "ALARM: Eyes closed > 5 sec!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_start_time = None
            alarm_triggered = False
        
        # ---------------------------
        # Gaze Direction Estimation
        # ---------------------------
        (l_x, l_y, l_w, l_h) = cv2.boundingRect(leftEye)
        left_eye_roi = frame[l_y:l_y+l_h, l_x:l_x+l_w]
        
        (r_x, r_y, r_w, r_h) = cv2.boundingRect(rightEye)
        right_eye_roi = frame[r_y:r_y+r_h, r_x:r_x+r_w]
        
        left_gaze = get_gaze_direction(left_eye_roi)
        right_gaze = get_gaze_direction(right_eye_roi)
        
        if left_gaze == right_gaze:
            gaze_direction = left_gaze
        else:
            gaze_direction = "CENTER"
        
        # ---------------------------
        # Combine Blinking and Gaze Conditions
        # ---------------------------
        if ear < EYE_AR_THRESH:
            current_state = "Blinking"
        else:
            current_state = gaze_direction
        
        # Annotate the eyes:
        # Draw thicker contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
        # Draw circles on each eye landmark
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        # Display the current state and blink count on the frame
        cv2.putText(frame, f"State: {current_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        results["Eye Detection"].append(current_state)
        results["Time"].append(datetime.datetime.now().strftime("%H:%M:%S"))
        print(current_state)
    
    # Show the video feed
    cv2.imshow("Eye Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

results = pd.DataFrame(results)
results.to_csv(f"Results/EyeTracking-{Operator_name}-{dt}.csv", index=False)
cap.release()
cv2.destroyAllWindows()
