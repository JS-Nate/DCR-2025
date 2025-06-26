import csv
import time
import os
from datetime import datetime

# Load and process CSV data
def load_data(file_path):
    data = []
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    try:
        with open(full_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({"emotion": row["Emotion Detection"], "time": datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")})
    except FileNotFoundError:
        print(f"Error: Could not find {full_path}. Please ensure it's in the same directory as this script or adjust the path.")
        return []
    return data

# Generate feedback based on emotion and shift
def get_feedback(current, last):
    shift = "no shift" if current == last else f"shift from {last} to {current}"
    if current == "happy":
        return f"Great! You're happy. {shift} might indicate a positive event."
    elif current == "neutral":
        return f"Stable mood. {shift} could suggest focus or a baseline return."
    else:  # Assuming sad for demo
        return f"Caution: Possible sad mood. {shift} may indicate stress."

# Main processing loop
def monitor_emotions(data):
    last_emotion = None
    for entry in data:
        current_emotion = entry["emotion"]
        if last_emotion is not None:
            feedback = get_feedback(current_emotion, last_emotion)
            print(f"Time: {entry['time'].strftime('%H:%M:%S')} | Mood: {current_emotion} | Feedback: {feedback}")
        last_emotion = current_emotion
        time.sleep(1)  # Simulate real-time check every second

# Run the script
if __name__ == "__main__":
    file_path = "emotion_detection-Test1-2025-06-24_13-52-29.csv"  # Relative to script location
    emotion_data = load_data(file_path)
    if emotion_data:
        print("Starting emotion monitoring...")
        monitor_emotions(emotion_data)
    else:
        print("Monitoring aborted due to data loading issues.")