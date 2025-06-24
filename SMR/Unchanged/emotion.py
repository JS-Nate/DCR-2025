import cv2
import datetime
import pandas as pd
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Dt_file= datetime.datetime.now()
Dt_file =Dt_file.strftime("%Y%m%d")
# Start capturing video
cap = cv2.VideoCapture(1)

Operator_name = input("Please Enter your name Before start the Scenario:")

results ={"Emotion Detection":[] , "Time":[]}
while True:
    dt_now =  datetime.datetime.now()
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        results["Emotion Detection"].append(emotion)
        results["Time"].append(dt_now)

        print(results)
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == 27:
        break


import os
# Define the file path



# Save the results

results =  pd.DataFrame(results)
print(results)
dt_now=dt_now.strftime("%Y-%m-%d_%H-%M-%S")
results.to_csv(f"Results\\emotion_detection-{Operator_name}-{dt_now}.csv")

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

