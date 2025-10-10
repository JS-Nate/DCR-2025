# analysis.py (updated with physiological feedback fallback generator)
SAFETY_MARGINS = {
    "1.1_heart_rate": {"min": 60, "max": 100},
    "1.2_skin_temp": {"min": 32.0, "max": 36.0},
    "1.3_face_emotion": {
        "safe": ["happy", "neutral"],
        "warning": ["surprised"],
        "danger": ["sad", "fear", "angry"]
    },
    "1.4_face_stress": {
        "safe": ["low"],
        "warning": ["medium"],
        "danger": ["high"]
    },
    "2.1_room_temp": {"min": 21, "max": 28},
    "2.2_cct_temp": {"min": 4500, "max": 5500},
    "2.3_light_intensity": {"min": 300, "max": 1000},
    "2.4_humidity": {"min": 30, "max": 60},
    "2.5_pressure": {"min": 980, "max": 1020}
}

# PSF weights
PSF_WEIGHTS = {
    "physiological": 0.30,
    "behavioral":    0.20,
    "interaction":   0.20,
    "environmental": 0.20,
    "task_system":   0.10,
}

OUTPUT_WEIGHTS = {
    "task_success":          0.40,  # 0..1
    "inv_completion_time":   0.20,  # uses 1/seconds
    "decision_accuracy":     0.20,  # 0..1
    "operational_stability": 0.10,  # 0..1
    "situational_awareness": 0.10,  # 0..1
}

def _clamp(x, lo, hi):
    try:
        x = float(x)
    except:
        return lo
    return max(lo, min(hi, x))

def _norm(x, lo, hi):
    x = _clamp(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-9)

def _dev_from_target(x, target, span):
    # 0 at target, ->1 when |x-target| >= span
    try:
        x = float(x)
    except:
        return 1.0
    return min(abs(x - target) / (span + 1e-9), 1.0)

def compute_psf_loads(hr, skin_temp, posture, eye_tracking, voice,
                      emotion, stress, room_temp, cct_temp, light_intensity, humidity, pressure,
                      task, task_duration, reactor_status="normal"):
    # Physiological (0..1 higher = more load)
    physiological = {
        "heart_rate": _norm(hr, 60, 120),
        "facial_stress": 1.0 if str(stress).lower()=="high" else 0.5 if str(stress).lower()=="medium" else 0.0,
        "emotion_negative": 1.0 if str(emotion).lower() in ["sad","fear","angry"] else 0.5 if str(emotion).lower()=="surprised" else 0.0,
        "skin_temp_dev": _dev_from_target(skin_temp, 34.0, 2.0),
    }

    # Behavioral
    behavioral = {
        "eye_tracking_anomaly": 0.5 if str(eye_tracking).lower() not in ["steady","normal","focused"] else 0.0,
        "posture_instability": 0.5 if str(posture).lower() not in ["neutral","stable"] else 0.0,
    }

    # Interaction
    interaction = {
        "voice_stress": 0.5 if str(voice).lower() in ["strained","shaky","loud"] else 0.0,
    }

    # Environmental
    environmental = {
        "temp_dev": _dev_from_target(room_temp, 24, 4),
        "cct_dev": _dev_from_target(cct_temp, 5000, 1000),
        "light_dev": 0.0 if 300 <= float(light_intensity) <= 1000 else 1.0,
        "humidity_dev": _dev_from_target(humidity, 45, 15),
        "pressure_dev": _dev_from_target(pressure, 1010, 20),
    }

    # Task/System
    task_system = {
        "task_complexity": 0.6 if str(task).lower() in ["emergency_shutdown","diagnostic_fault","abnormal_ops"] else 0.3,
        "task_duration_norm": _norm(task_duration, 30, 900),
        "reactor_malfunction": 1.0 if str(reactor_status).lower()!="normal" else 0.0,
    }

    per_group = {
        "physiological": sum(physiological.values()) / len(physiological),
        "behavioral":    sum(behavioral.values())    / max(len(behavioral),1),
        "interaction":   sum(interaction.values())   / max(len(interaction),1),
        "environmental": sum(environmental.values()) / len(environmental),
        "task_system":   sum(task_system.values())   / len(task_system),
    }

    input_load_score = sum(per_group[g]*PSF_WEIGHTS[g] for g in PSF_WEIGHTS)
    return per_group, input_load_score

def compute_output_score(task_success=None, completion_time_s=None, decision_accuracy=None,
                         operational_stability=None, situational_awareness=None, fallback_task_duration=None):
    ts  = float(task_success) if task_success is not None else 1.0
    cts = float(completion_time_s) if completion_time_s is not None else float(fallback_task_duration or 120.0)
    da  = float(decision_accuracy) if decision_accuracy is not None else 0.8
    os_ = float(operational_stability) if operational_stability is not None else 0.8
    sa  = float(situational_awareness) if situational_awareness is not None else 0.8
    inv_time = 1.0 / max(cts, 1.0)
    score = (
        ts  * OUTPUT_WEIGHTS["task_success"] +
        inv_time * OUTPUT_WEIGHTS["inv_completion_time"] +
        da  * OUTPUT_WEIGHTS["decision_accuracy"] +
        os_ * OUTPUT_WEIGHTS["operational_stability"] +
        sa  * OUTPUT_WEIGHTS["situational_awareness"]
    )
    return max(0.0, min(1.0, score))

def compute_performance(per_group_loads, input_load_score, output_score, eps=1e-6):
    perf = output_score / max(input_load_score, eps)
    state = "normal" if perf >= 0.85 else "warning" if perf >= 0.70 else "critical"
    return round(perf, 3), state

def calculate_performance_score_legacy(hr, skin_temp, temp, task, emotion, stress,
                                cct_temp, light_intensity, humidity, pressure):
    score = 1.0
    if hr > 110 or hr < 50:
        score -= 0.15
    if skin_temp < 32.0 or skin_temp > 36.0:
        score -= 0.05
    if emotion.lower() in ["sad", "fear", "angry"]:
        score -= 0.05
    if stress.lower() == "high":
        score -= 0.1
    elif stress.lower() == "medium":
        score -= 0.05
    if temp > 28 or temp < 20:
        score -= 0.05
    if not (4500 <= cct_temp <= 5500):
        score -= 0.05
    if light_intensity < 300 or light_intensity > 1000:
        score -= 0.05
    if humidity < 30 or humidity > 60:
        score -= 0.05
    if pressure < 980 or pressure > 1020:
        score -= 0.05
    return round(max(0.0, min(score, 1.0)), 2)

def check_safety_margins(hr, skin_temp, temp, emotion, stress, cct_temp, light_intensity, humidity, pressure):
    status = []
    def margin(value, low, high, param, id):
        if value < low:
            return (id, param, value, f"⚠️ Below threshold (< {low})")
        elif value > high:
            return (id, param, value, f"🚨 Above threshold (> {high})")
        else:
            return (id, param, value, "✅ Normal")
    status.append(margin(hr, 60, 100, "Heart Rate", "1.1"))
    status.append(margin(skin_temp, 32.0, 36.0, "Skin Temperature", "1.2"))
    if emotion.lower() in SAFETY_MARGINS["1.3_face_emotion"]["danger"]:
        status.append(("1.3", "Emotion", emotion, "⚠️ Elevated Stress"))
    elif emotion.lower() in SAFETY_MARGINS["1.3_face_emotion"]["warning"]:
        status.append(("1.3", "Emotion", emotion, "⚠️ Caution"))
    else:
        status.append(("1.3", "Emotion", emotion, "✅ Normal"))
    if stress.lower() in SAFETY_MARGINS["1.4_face_stress"]["danger"]:
        status.append(("1.4", "Facial Stress", stress, "🚨 High Stress"))
    elif stress.lower() in SAFETY_MARGINS["1.4_face_stress"]["warning"]:
        status.append(("1.4", "Facial Stress", stress, "⚠️ Medium Stress"))
    else:
        status.append(("1.4", "Facial Stress", stress, "✅ Normal"))
    status.append(margin(temp, 21, 28, "Room Temperature", "2.1"))
    status.append(margin(cct_temp, 4500, 5500, "CCT Temperature", "2.2"))
    status.append(margin(light_intensity, 300, 1000, "Light Intensity (lux)", "2.3"))
    status.append(margin(humidity, 30, 60, "Humidity (%)", "2.4"))
    status.append(margin(pressure, 980, 1020, "Pressure (hPa)", "2.5"))
    return status

def generate_environment_commands(cct_temp=None, light_intensity=None, humidity=None, pressure=None):
    cmds = []
    if cct_temp is not None:
        if cct_temp < 4500:
            cmds.append("adjust_cct_temperature(5000)  # raise CCT")
        elif cct_temp > 5500:
            cmds.append("adjust_cct_temperature(5000)  # lower CCT")
    if light_intensity is not None:
        if light_intensity < 300:
            cmds.append("increase_light_intensity(600)")
        elif light_intensity > 1000:
            cmds.append("decrease_light_intensity(800)")
    if humidity is not None:
        if humidity < 30:
            cmds.append("humidifier_on(target=40)")
        elif humidity > 60:
            cmds.append("dehumidifier_on(target=50)")
    if pressure is not None:
        if pressure < 980:
            cmds.append("adjust_pressure(1000)  # low pressure detected")
        elif pressure > 1020:
            cmds.append("adjust_pressure(1000)  # high pressure detected")
    return cmds

# ---------------- Physiological recommendations (fallback) ----------------
def build_local_physiological_feedback(
    hr: float,
    skin_temp: float,
    stress: str,
    emotion: str,
    posture: str,
    eye_tracking: str,
    voice: str,
    humidity: float,
) -> list[dict]:
    """Rule-based suggestions if LLaMA is unavailable or returns invalid JSON."""
    def _ls(x): return str(x).lower()

    recs: list[dict] = []

    # HR / stress management
    if (isinstance(hr, (int, float)) and hr > 100) or _ls(stress) in ("medium", "high"):
        recs.append({
            "label": "paced_breathing_60s",
            "why": "Heart rate elevated and/or stress medium/high",
            "how_to": "Inhale 4s, exhale 6s for 60s, seated, shoulders relaxed.",
            "duration_s": 60,
            "expected_effect": "Lower sympathetic arousal; HR trend down",
            "monitor": "Recheck HR and stress label after 5–10 min",
            "constraints": "Stop if dizzy; notify supervisor if symptoms persist"
        })

    # Eye fatigue
    if _ls(eye_tracking) in ("blinking frequently", "distracted", "unsteady"):
        recs.append({
            "label": "visual_reset_20_20_20",
            "why": "Eye fatigue indicators (blink/focus issues)",
            "how_to": "Every 20 min, look 20 ft away for 20 s; blink deliberately.",
            "duration_s": 20,
            "expected_effect": "Reduce visual fatigue",
            "monitor": "Blink comfort; fewer eye-tracking anomalies",
            "constraints": "n/a"
        })

    # Posture / voice strain
    if _ls(posture) not in ("neutral", "stable") or _ls(voice) in ("strained", "shaky", "loud"):
        recs.append({
            "label": "microbreak_posture_reset",
            "why": "Posture/voice strain detected",
            "how_to": "Stand 60s, shoulder rolls x6, gentle neck stretch.",
            "duration_s": 60,
            "expected_effect": "Reduce muscle tension, improve voice control",
            "monitor": "Perceived effort ↓; voice strain ↓",
            "constraints": "Avoid if balance is impaired"
        })

    # Hydration
    if (isinstance(humidity, (int, float)) and humidity < 35) or _ls(voice) in ("strained",):
        recs.append({
            "label": "sip_water",
            "why": "Dry air or voice strain",
            "how_to": "Sip 50–100 ml water.",
            "duration_s": 30,
            "expected_effect": "Support thermoregulation and vocal clarity",
            "monitor": "Voice strain ↓",
            "constraints": "Follow site hydration rules"
        })

    # Skin temperature out of band
    if isinstance(skin_temp, (int, float)) and (skin_temp < 32.0 or skin_temp > 36.0):
        recs.append({
            "label": "thermal_comfort_adjust",
            "why": f"Skin temp {skin_temp:.1f} °C outside 32–36 °C",
            "how_to": "If safe, adjust local airflow; relax tight PPE briefly.",
            "duration_s": 60,
            "expected_effect": "Move toward thermal comfort range",
            "monitor": "Skin temp closer to 32–36 °C",
            "constraints": "Only when operationally safe"
        })

    return recs
