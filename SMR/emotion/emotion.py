import cv2
import datetime
import pandas as pd
from deepface import DeepFace
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import os

# === GLOBAL SETUP ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(3)

Operator_name = input("Please Enter your name Before start the Scenario:")

results = {"Emotion Detection": [], "Time": []}
results_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=1)

# Output file setup
os.makedirs("Results", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = f"Results/emotion_detection-{Operator_name}-{timestamp}.csv"

SAVE_EVERY_N = 10  # Save results every N detections

# === EMOTION ANALYSIS FUNCTION ===
def analyze_emotion(face_roi, x, y, w, h, frame, dt_now):
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        emotion = "error"
        print(f"Emotion detection error: {e}")

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    with results_lock:
        results["Emotion Detection"].append(emotion)
        results["Time"].append(dt_now.strftime("%Y-%m-%d %H:%M:%S"))

        # Save periodically to CSV
        if len(results["Emotion Detection"]) % SAVE_EVERY_N == 0:
            pd.DataFrame(results).to_csv(csv_path, index=False)

# === FRAME CAPTURE THREAD ===
def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

# === FRAME PROCESSING THREAD ===
def process_frames():
    executor = ThreadPoolExecutor(max_workers=2)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        dt_now = datetime.datetime.now()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        futures = []
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            futures.append(executor.submit(analyze_emotion, face_roi, x, y, w, h, frame, dt_now))

        for future in futures:
            future.result()

        cv2.imshow('Real-time Emotion Detection', frame)
        if cv2.waitKey(1) == 27:
            frame_queue.put(None)  # Stop signal
            break

# === START THREADS ===
t1 = threading.Thread(target=capture_frames, daemon=True)
t2 = threading.Thread(target=process_frames)

t1.start()
t2.start()

t2.join()

# === FINAL CSV SAVE ===
with results_lock:
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nFinal results saved to: {csv_path}")

cap.release()
cv2.destroyAllWindows()
