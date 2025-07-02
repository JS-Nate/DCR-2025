# Safety margin thresholds for different input parameters
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
    # Heart rate score
    if hr < 90:
        hr_score = 0.1
    elif hr < 110:
        hr_score = 0.5
    else:
        hr_score = 0.9

    # EEG score
    eeg_score = {
        "alpha-dominant": 0.1,
        "beta-dominant": 0.3,
        "theta-dominant": 0.7,
        "alpha-suppressed": 0.9
    }.get(eeg_signal.lower(), 0.5)

    # Room temperature stress
    if room_temp < 21 or room_temp > 28:
        temp_score = 0.9
    elif 26 < room_temp <= 28:
        temp_score = 0.6
    else:
        temp_score = 0.2

    # Task difficulty
    task_difficulty = {
        "startup": 0.4,
        "load_change": 0.6,
        "shutdown": 0.8,
        "emergency_shutdown": 1.0
    }.get(task.lower(), 0.5)

    # Face emotion stress mapping
    emotion_score = {
        "happy": 0.1,
        "neutral": 0.2,
        "sad": 0.7,
        "fear": 0.8,
        "angry": 0.9,
        "surprised": 0.6
    }.get(face_emotion.lower(), 0.5)

    # Final score (can tweak weights)
    score = (
        0.25 * hr_score +
        0.25 * eeg_score +
        0.15 * temp_score +
        0.15 * task_difficulty +
        0.2 * emotion_score
    )
    return round(score, 2)


def detect_coupling(hr, eeg, temp):
    coupling_issues = []

    if hr > 110 and "alpha-suppressed" in eeg.lower():
        coupling_issues.append("High HR and alpha-suppressed EEG (possible overload)")

    if temp > 28 and "theta" in eeg.lower():
        coupling_issues.append("Elevated room temperature + low focus EEG (fatigue risk)")

    if not coupling_issues:
        coupling_issues.append("No critical coupling detected")

    return coupling_issues


def get_alert_level(score):
    if score < 0.4:
        return "🟢 GREEN", "Performance is within normal bounds."
    elif score < 0.7:
        return "⚠️ YELLOW", "Operator or environment stress detected — monitor closely."
    else:
        return "🚨 RED", "High performance risk due to multiple stressors."



def check_safety_margins(hr, eeg_signal, temp, emotion):
    result = []

    # 1.1 Heart Rate
    if hr < SAFETY_MARGINS["1.1_heart_rate"]["min"] or hr > SAFETY_MARGINS["1.1_heart_rate"]["max"]:
        status = "🚨 Danger"
    else:
        status = "✅ OK"
    result.append(("1.1", "Heart Rate", hr, status))

    # 1.2 EEG
    eeg = eeg_signal.lower()
    if eeg in SAFETY_MARGINS["1.2_eeg_signal"]["danger"]:
        status = "🚨 Danger"
    elif eeg in SAFETY_MARGINS["1.2_eeg_signal"]["warning"]:
        status = "⚠️ Warning"
    else:
        status = "✅ OK"
    result.append(("1.2", "EEG Signal", eeg_signal, status))

    # 1.3 Face Emotion
    emo = emotion.lower()
    if emo in SAFETY_MARGINS["1.3_face_emotion"]["danger"]:
        status = "🚨 Danger"
    elif emo in SAFETY_MARGINS["1.3_face_emotion"]["warning"]:
        status = "⚠️ Warning"
    else:
        status = "✅ OK"
    result.append(("1.3", "Facial Emotion", emotion, status))

    # 2.1 Room Temp
    if temp < SAFETY_MARGINS["2.1_room_temp"]["min"] or temp > SAFETY_MARGINS["2.1_room_temp"]["max"]:
        status = "🚨 Danger"
    else:
        status = "✅ OK"
    result.append(("2.1", "Room Temperature", temp, status))

    return result
