# analysis.py  (updated to align with streamlit_app.py + HPSN)
from __future__ import annotations
from typing import Dict, Tuple, List, Any

# ----------------------------- Constants -----------------------------

SAFETY_MARGINS: Dict[str, Any] = {
    "1.1_heart_rate": {"min": 60, "max": 100},
    "1.2_skin_temp": {"min": 32.0, "max": 36.0},
    "1.3_face_emotion": {
        "safe":    ["happy", "neutral"],
        "warning": ["surprised"],
        "danger":  ["sad", "fear", "angry"],
    },
    "1.4_face_stress": {
        "safe":    ["low"],
        "warning": ["medium"],
        "danger":  ["high"],
    },
    "2.1_room_temp":       {"min": 21, "max": 28},
    "2.2_cct_temp":        {"min": 4500, "max": 5500},
    "2.3_light_intensity": {"min": 300, "max": 1000},
    "2.4_humidity":        {"min": 30, "max": 60},
    "2.5_pressure":        {"min": 980, "max": 1020},
}

# Group weights used by compute_psf_loads()
PSF_WEIGHTS: Dict[str, float] = {
    "physiological": 0.30,
    "behavioral":    0.20,
    "interaction":   0.20,
    "environmental": 0.20,
    "task_system":   0.10,
}

# Output components used by compute_output_score()
OUTPUT_WEIGHTS: Dict[str, float] = {
    "task_success":          0.40,  # 0..1
    "inv_completion_time":   0.20,  # uses 1/seconds
    "decision_accuracy":     0.20,  # 0..1
    "operational_stability": 0.10,  # 0..1
    "situational_awareness": 0.10,  # 0..1
}

# ----------------------------- Helpers ------------------------------

_EMOTION_CODE_MAP = {
    1: "happy", 2: "neutral", 3: "surprised",
    4: "sad",   5: "fear",    6: "angry",
}

def _clamp(x: Any, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi], coercing to float when possible."""
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))

def _norm(x: Any, lo: float, hi: float) -> float:
    """Normalize x to [0,1] given bounds [lo, hi]."""
    x = _clamp(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-9)

def _dev_from_target(x: Any, target: float, span: float) -> float:
    """0 at target; →1 when |x - target| >= span."""
    try:
        x = float(x)
    except Exception:
        return 1.0
    return min(abs(x - target) / (span + 1e-9), 1.0)

def _to_lower_str(x: Any, default: str = "unknown") -> str:
    """Convert value to lower-case string safely."""
    try:
        s = str(x).strip().lower()
        return s if s else default
    except Exception:
        return default

def _normalize_emotion(x: Any) -> str:
    """Accepts string or int code (1..6) and returns canonical emotion string."""
    try:
        if isinstance(x, (int, float)) and int(x) in _EMOTION_CODE_MAP:
            return _EMOTION_CODE_MAP[int(x)]
    except Exception:
        pass
    return _to_lower_str(x, "neutral")

# ------------------------- Core Calculations -------------------------

def compute_psf_loads(
    hr: Any,
    skin_temp: Any,
    posture: Any,
    eye_tracking: Any,
    voice: Any,
    emotion: Any,
    stress: Any,
    room_temp: Any,
    cct_temp: Any,
    light_intensity: Any,
    humidity: Any,
    pressure: Any,
    task: Any,
    task_duration: Any,
    reactor_status: Any = "normal",
) -> Tuple[Dict[str, float], float]:
    """
    Compute per-group PSF load (0..1, higher = more load) and an overall input load score.
    Returns: (per_group_loads: dict, input_load_score: float)
    """
    emotion_s = _normalize_emotion(emotion)
    stress_s  = _to_lower_str(stress, "low")
    posture_s = _to_lower_str(posture, "unknown")
    eyes_s    = _to_lower_str(eye_tracking, "unknown")
    voice_s   = _to_lower_str(voice, "unknown")
    task_s    = _to_lower_str(task, "unknown")
    reactor_s = _to_lower_str(reactor_status, "normal")

    # Physiological
    physiological = {
        "heart_rate":       _norm(hr, 60, 120),  # tune ranges as you collect data
        "facial_stress":    1.0 if stress_s == "high" else 0.5 if stress_s == "medium" else 0.0,
        "emotion_negative": 1.0 if emotion_s in ["sad", "fear", "angry"]
                              else 0.5 if emotion_s == "surprised" else 0.0,
        "skin_temp_dev":    _dev_from_target(skin_temp, 34.0, 2.0),  # ~34±2 °C comfort
    }

    # Behavioral (boolean-ish proxies for now)
    behavioral = {
        "eye_tracking_anomaly": 0.5 if eyes_s not in ["steady", "normal", "focused"] else 0.0,
        "posture_instability":  0.5 if posture_s not in ["neutral", "stable"] else 0.0,
    }

    # Interaction
    interaction = {
        "voice_stress": 0.5 if voice_s in ["strained", "shaky", "loud"] else 0.0,
    }

    # Environmental
    environmental = {
        "temp_dev":     _dev_from_target(room_temp, 24, 4),       # 20–28
        "cct_dev":      _dev_from_target(cct_temp, 5000, 1000),   # 4k–6k
        "light_dev":    0.0 if 300 <= float(_clamp(light_intensity, 0, 10_000)) <= 1000 else 1.0,
        "humidity_dev": _dev_from_target(humidity, 45, 15),       # 30–60
        "pressure_dev": _dev_from_target(pressure, 1010, 20),
    }

    # Task/System
    task_system = {
        "task_complexity":   0.6 if task_s in ["emergency_shutdown", "diagnostic_fault", "abnormal_ops"] else 0.3,
        "task_duration_norm": _norm(task_duration, 30, 900),      # 0 at 30s, 1 at 15m
        "reactor_malfunction": 1.0 if reactor_s != "normal" else 0.0,
    }

    # Group averages
    per_group: Dict[str, float] = {
        "physiological": sum(physiological.values()) / max(len(physiological), 1),
        "behavioral":    sum(behavioral.values())    / max(len(behavioral),    1),
        "interaction":   sum(interaction.values())   / max(len(interaction),   1),
        "environmental": sum(environmental.values()) / max(len(environmental), 1),
        "task_system":   sum(task_system.values())   / max(len(task_system),   1),
    }

    input_load_score = sum(per_group[g] * PSF_WEIGHTS[g] for g in PSF_WEIGHTS)
    return per_group, float(input_load_score)

def compute_output_score(
    task_success: Any = None,
    completion_time_s: Any = None,
    decision_accuracy: Any = None,
    operational_stability: Any = None,
    situational_awareness: Any = None,
    fallback_task_duration: Any = None,
) -> float:
    """
    Compute an output-side score in [0,1] from optional fields (robust to missing data).
    """
    ts  = float(task_success) if task_success is not None else 1.0
    cts = float(completion_time_s) if completion_time_s is not None else float(fallback_task_duration or 120.0)
    da  = float(decision_accuracy) if decision_accuracy is not None else 0.8
    os_ = float(operational_stability) if operational_stability is not None else 0.8
    sa  = float(situational_awareness) if situational_awareness is not None else 0.8

    inv_time = 1.0 / max(cts, 1.0)

    score = (
        ts       * OUTPUT_WEIGHTS["task_success"] +
        inv_time * OUTPUT_WEIGHTS["inv_completion_time"] +
        da       * OUTPUT_WEIGHTS["decision_accuracy"] +
        os_      * OUTPUT_WEIGHTS["operational_stability"] +
        sa       * OUTPUT_WEIGHTS["situational_awareness"]
    )
    return float(max(0.0, min(1.0, score)))

def compute_performance(
    per_group_loads: Dict[str, float],
    input_load_score: float,
    output_score: float,
    eps: float = 1e-6,
) -> Tuple[float, str]:
    """
    Performance = output_score / input_load_score (higher is better).
    Returns: (perf_score, perf_state) where state ∈ {"normal","warning","critical"}.
    """
    perf = output_score / max(input_load_score, eps)
    state = "normal" if perf >= 0.85 else "warning" if perf >= 0.70 else "critical"
    return round(float(perf), 3), state

# --------------------------- Safety & Commands -----------------------

def calculate_performance_score_legacy(
    hr: Any, skin_temp: Any, temp: Any, task: Any, emotion: Any, stress: Any,
    cct_temp: Any, light_intensity: Any, humidity: Any, pressure: Any
) -> float:
    """
    Legacy heuristic score in [0,1]. Kept for backwards-compat testing.
    """
    score = 1.0
    try:
        if float(hr) > 110 or float(hr) < 50:
            score -= 0.15
    except Exception:
        pass

    try:
        stf = float(skin_temp)
        if stf < 32.0 or stf > 36.0:
            score -= 0.05
    except Exception:
        score -= 0.05

    emotion_s = _normalize_emotion(emotion)
    if emotion_s in ["sad", "fear", "angry"]:
        score -= 0.05

    stress_s = _to_lower_str(stress, "low")
    if stress_s == "high":
        score -= 0.1
    elif stress_s == "medium":
        score -= 0.05

    try:
        tf = float(temp)
        if tf > 28 or tf < 20:
            score -= 0.05
    except Exception:
        score -= 0.05

    try:
        cctf = float(cct_temp)
        if not (4500 <= cctf <= 5500):
            score -= 0.05
    except Exception:
        score -= 0.05

    try:
        li = float(light_intensity)
        if li < 300 or li > 1000:
            score -= 0.05
    except Exception:
        score -= 0.05

    try:
        hum = float(humidity)
        if hum < 30 or hum > 60:
            score -= 0.05
    except Exception:
        score -= 0.05

    try:
        p = float(pressure)
        if p < 980 or p > 1020:
            score -= 0.05
    except Exception:
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 2)

def check_safety_margins(
    hr: Any,
    skin_temp: Any,
    temp: Any,
    emotion: Any,
    stress: Any,
    cct_temp: Any,
    light_intensity: Any,
    humidity: Any,
    pressure: Any,
) -> List[Tuple[str, str, Any, str]]:
    """
    Returns a list of tuples: (ID, Parameter, Value, StatusText)
    Signature matches the updated streamlit_app.py call order.
    """
    status: List[Tuple[str, str, Any, str]] = []

    def margin(value: Any, low: float, high: float, param: str, id_: str):
        try:
            v = float(value)
        except Exception:
            return (id_, param, value, f"⚠️ Invalid")
        if v < low:
            return (id_, param, v, f"⚠️ Below threshold (< {low})")
        elif v > high:
            return (id_, param, v, f"🚨 Above threshold (> {high})")
        else:
            return (id_, param, v, "✅ Normal")

    # Physiological
    status.append(margin(hr,         SAFETY_MARGINS["1.1_heart_rate"]["min"], SAFETY_MARGINS["1.1_heart_rate"]["max"], "Heart Rate",          "1.1"))
    status.append(margin(skin_temp,  SAFETY_MARGINS["1.2_skin_temp"]["min"],  SAFETY_MARGINS["1.2_skin_temp"]["max"],  "Skin Temperature",     "1.2"))

    # Emotion (categorical)
    emotion_s = _normalize_emotion(emotion)
    if emotion_s in SAFETY_MARGINS["1.3_face_emotion"]["danger"]:
        status.append(("1.3", "Emotion", emotion_s, "🚨 High-Risk Emotion"))
    elif emotion_s in SAFETY_MARGINS["1.3_face_emotion"]["warning"]:
        status.append(("1.3", "Emotion", emotion_s, "⚠️ Elevated Emotion"))
    else:
        status.append(("1.3", "Emotion", emotion_s, "✅ Normal"))

    # Facial stress (categorical)
    stress_s = _to_lower_str(stress, "low")
    if stress_s in SAFETY_MARGINS["1.4_face_stress"]["danger"]:
        status.append(("1.4", "Facial Stress", stress_s, "🚨 High Stress"))
    elif stress_s in SAFETY_MARGINS["1.4_face_stress"]["warning"]:
        status.append(("1.4", "Facial Stress", stress_s, "⚠️ Medium Stress"))
    else:
        status.append(("1.4", "Facial Stress", stress_s, "✅ Normal"))

    # Environmental
    status.append(margin(temp,           SAFETY_MARGINS["2.1_room_temp"]["min"],       SAFETY_MARGINS["2.1_room_temp"]["max"],       "Room Temperature",     "2.1"))
    status.append(margin(cct_temp,       SAFETY_MARGINS["2.2_cct_temp"]["min"],        SAFETY_MARGINS["2.2_cct_temp"]["max"],        "CCT Temperature",      "2.2"))
    status.append(margin(light_intensity,SAFETY_MARGINS["2.3_light_intensity"]["min"], SAFETY_MARGINS["2.3_light_intensity"]["max"], "Light Intensity (lux)","2.3"))
    status.append(margin(humidity,       SAFETY_MARGINS["2.4_humidity"]["min"],        SAFETY_MARGINS["2.4_humidity"]["max"],        "Humidity (%)",         "2.4"))
    status.append(margin(pressure,       SAFETY_MARGINS["2.5_pressure"]["min"],        SAFETY_MARGINS["2.5_pressure"]["max"],        "Pressure (hPa)",       "2.5"))

    return status



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

    # Hydration (dry air or voice strain)
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


def generate_environment_commands(
    cct_temp: Any = None,
    light_intensity: Any = None,
    humidity: Any = None,
    pressure: Any = None,
) -> List[str]:
    """
    Produce simple environment adjustment suggestions based on current readings.
    """
    cmds: List[str] = []

    try:
        if cct_temp is not None:
            cct = float(cct_temp)
            if cct < SAFETY_MARGINS["2.2_cct_temp"]["min"]:
                cmds.append("adjust_cct_temperature(5000)  # raise CCT")
            elif cct > SAFETY_MARGINS["2.2_cct_temp"]["max"]:
                cmds.append("adjust_cct_temperature(5000)  # lower CCT")
    except Exception:
        pass

    try:
        if light_intensity is not None:
            lux = float(light_intensity)
            if lux < SAFETY_MARGINS["2.3_light_intensity"]["min"]:
                cmds.append("increase_light_intensity(600)")
            elif lux > SAFETY_MARGINS["2.3_light_intensity"]["max"]:
                cmds.append("decrease_light_intensity(800)")
    except Exception:
        pass

    try:
        if humidity is not None:
            hum = float(humidity)
            if hum < SAFETY_MARGINS["2.4_humidity"]["min"]:
                cmds.append("humidifier_on(target=40)")
            elif hum > SAFETY_MARGINS["2.4_humidity"]["max"]:
                cmds.append("dehumidifier_on(target=50)")
    except Exception:
        pass

    try:
        if pressure is not None:
            p = float(pressure)
            if p < SAFETY_MARGINS["2.5_pressure"]["min"]:
                cmds.append("adjust_pressure(1000)  # low pressure detected")
            elif p > SAFETY_MARGINS["2.5_pressure"]["max"]:
                cmds.append("adjust_pressure(1000)  # high pressure detected")
    except Exception:
        pass

    return cmds
