import cv2
import datetime
import pandas as pd
from deepface import DeepFace
import threading
from playsound import playsound

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Dt_file = datetime.datetime.now()
Dt_file = Dt_file.strftime("%Y%m%d")

# Start capturing video
cap = cv2.VideoCapture(1)  # Change index if needed

Operator_name = input("Please Enter your name Before start the Scenario:")

results = {"Emotion Detection": [], "Time": []}

# Alarm variables for continuous "angry" emotion
alarm_start_time = None
alarm_triggered = False

def play_alarm_sound():
    """
    Play an alarm sound from a local file.
    """
    try:
        playsound("Alarm sound effect.mp3")
    except Exception as e:
        print("Error playing sound:", e)

while True:
    dt_now = datetime.datetime.now()
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and then to RGB for DeepFace processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI and perform emotion analysis
        face_roi = rgb_frame[y:y+h, x:x+w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        
        emotion = result[0]['dominant_emotion']


        if emotion.lower() == "angry" or emotion.lower() == "fear" or  emotion.lower() == "sad":
            emotion = "high stress"
        elif emotion.lower() == "Disgust" or emotion.lower() == "surprise":
            emotion = "low stress"
        else:
            emotion = "non stress"
        
        
        # Draw rectangle around face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        results["Emotion Detection"].append(emotion)
        results["Time"].append(dt_now.strftime("%H:%M:%S"))
       
        print(results)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break

# Save the results
results = pd.DataFrame(results)
dt_now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# results.to_csv(f"Results/Stress-{Operator_name}-{dt_now_str}.csv", index=False)
results.to_csv(f"Results\\Stress-{Operator_name}-{dt_now_str}.csv", index=False)

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
