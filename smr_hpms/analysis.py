# analysis.py
SAFETY_MARGINS = {
    "1.1_heart_rate": {"min": 60, "max": 100},
    "1.2_eeg_signal": {
        "safe": ["alpha-dominant", "beta-dominant"],
        "warning": ["theta-dominant"],
        "danger": ["alpha-suppressed"]
    },
    "1.3_face_emotion": {
        "safe": ["happy", "neutral"],
        "warning": ["surprised"],
        "danger": ["sad", "fear", "angry"]
    },
    "2.1_room_temp": {"min": 21, "max": 28}
}

def calculate_performance_score(hr, eeg_signal, room_temp, task, face_emotion):
    if hr < 90:
        hr_score = 0.1
    elif hr < 110:
        hr_score = 0.5
    else:
        hr_score = 0.9

    eeg_score = {
        "alpha-dominant": 0.1,
        "beta-dominant": 0.3,
        "theta-dominant": 0.7,
        "alpha-suppressed": 0.9
    }.get(eeg_signal.lower(), 0.5)

    if room_temp < 21 or room_temp > 28:
        temp_score = 0.9
    elif 26 < room_temp <= 28:
        temp_score = 0.6
    else:
        temp_score = 0.2

    task_difficulty = {
        "startup": 0.4,
        "load_change": 0.6,
        "shutdown": 0.8,
        "emergency_shutdown": 1.0
    }.get(task.lower(), 0.5)

    emotion_score = {
        "happy": 0.1,
        "neutral": 0.2,
        "sad": 0.7,
        "fear": 0.8,
        "angry": 0.9,
        "surprised": 0.6
    }.get(face_emotion.lower(), 0.5)

    score = (
        0.25 * hr_score +
        0.25 * eeg_score +
        0.15 * temp_score +
        0.15 * task_difficulty +
        0.2 * emotion_score
    )
    return round(score, 2)

def check_safety_margins(hr, eeg_signal, temp, emotion):
    result = []

    if hr < SAFETY_MARGINS["1.1_heart_rate"]["min"] or hr > SAFETY_MARGINS["1.1_heart_rate"]["max"]:
        status = "🚨 Danger"
    else:
        status = "✅ OK"
    result.append(("1.1", "Heart Rate", hr, status))

    eeg = eeg_signal.lower()
    if eeg in SAFETY_MARGINS["1.2_eeg_signal"]["danger"]:
        status = "🚨 Danger"
    elif eeg in SAFETY_MARGINS["1.2_eeg_signal"]["warning"]:
        status = "⚠️ Warning"
    else:
        status = "✅ OK"
    result.append(("1.2", "EEG Signal", eeg_signal, status))

    emo = emotion.lower()
    if emo in SAFETY_MARGINS["1.3_face_emotion"]["danger"]:
        status = "🚨 Danger"
    elif emo in SAFETY_MARGINS["1.3_face_emotion"]["warning"]:
        status = "⚠️ Warning"
    else:
        status = "✅ OK"
    result.append(("1.3", "Facial Emotion", emotion, status))

    if temp < SAFETY_MARGINS["2.1_room_temp"]["min"] or temp > SAFETY_MARGINS["2.1_room_temp"]["max"]:
        status = "🚨 Danger"
    else:
        status = "✅ OK"
    result.append(("2.1", "Room Temperature", temp, status))

    return result
