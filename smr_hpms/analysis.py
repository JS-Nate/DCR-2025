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




def calculate_performance_score(hr, eeg, temp, task, emotion,
                                light_temp, light_intensity, humidity, pressure):
    score = 1.0

    # Physiological adjustments
    if hr > 110 or hr < 50:
        score -= 0.15
    if eeg.lower() in ["alpha-suppressed", "theta-dominant"]:
        score -= 0.1
    if emotion.lower() in ["sad", "fear", "angry"]:
        score -= 0.05

    # Environmental factors
    if temp > 28 or temp < 20:
        score -= 0.05
    if not (4500 <= light_temp <= 5500):
        score -= 0.05
    if light_intensity < 300 or light_intensity > 1000:
        score -= 0.05
    if humidity < 30 or humidity > 60:
        score -= 0.05
    if pressure < 980 or pressure > 1020:
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 2)



def check_safety_margins(hr, eeg, temp, emotion, light_temp, light_intensity, humidity, pressure):
    status = []

    def margin(value, low, high, param, id):
        if value < low:
            return (id, param, value, f"⚠️ Below threshold (< {low})")
        elif value > high:
            return (id, param, value, f"🚨 Above threshold (> {high})")
        else:
            return (id, param, value, "✅ Normal")

    status.append(margin(hr, 60, 100, "Heart Rate", "1.1"))
    status.append(margin(temp, 20, 27, "Room Temperature", "1.2"))

    if eeg.lower() in ["alpha-suppressed", "theta-dominant"]:
        status.append(("1.3", "EEG", eeg, "⚠️ Suboptimal"))
    else:
        status.append(("1.3", "EEG", eeg, "✅ Normal"))

    if emotion.lower() in ["sad", "fear", "angry"]:
        status.append(("1.4", "Emotion", emotion, "⚠️ Elevated Stress"))
    else:
        status.append(("1.4", "Emotion", emotion, "✅ Normal"))

    status.append(margin(light_temp, 4500, 5500, "Light Temperature (K)", "2.1"))
    status.append(margin(light_intensity, 300, 1000, "Light Intensity (lux)", "2.2"))
    status.append(margin(humidity, 30, 60, "Humidity (%)", "2.3"))
    status.append(margin(pressure, 980, 1020, "Pressure (hPa)", "2.4"))

    return status
