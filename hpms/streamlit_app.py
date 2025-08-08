import streamlit as st
import pandas as pd
import datetime
import requests

# NEW imports that match the updated analysis.py
from analysis import (
    compute_psf_loads,        # NEW
    compute_output_score,     # NEW
    compute_performance,      # NEW
    check_safety_margins,     # existing
    generate_environment_commands  # existing
)

# Your existing HPSN utilities (unchanged)
from hpsn import load_hpsn, query_hpsn, update_hpsn, map_inputs_to_nodes, save_hpsn

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="SMR HPMS Dashboard", layout="wide")
st.title("SMR Human Performance Management System (HPMS)")
st.caption("Real-time Output/Input performance with PSF loads, alerts, and HPSN integration.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Session logs
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# ---------------- HELPERS ----------------
def safe_str(x, default="unknown"):
    try:
        s = str(x).strip()
        return s if s else default
    except:
        return default

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def safe_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default

# ---------------- MAIN ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Required inputs (outputs are optional and will fall back if missing)
    required_cols = [
        'timestamp', 'heart_rate', 'skin_temp', 'posture', 'eye_tracking', 'voice',
        'face_emotion', 'face_stress', 'task', 'task_duration',
        'room_temp', 'cct_temp', 'light_intensity', 'humidity', 'pressure'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    # Optional outputs & reactor flag
    optional_cols = [
        'task_success', 'completion_time_s', 'decision_accuracy',
        'operational_stability', 'situational_awareness', 'reactor_status'
    ]
    for c in optional_cols:
        if c not in df.columns:
            df[c] = None  # so row.get(...) works

    # Prepare time and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values(by='timestamp')

    # Use last record as "current"
    row = df.iloc[-1]

    # ------------- Extract inputs -------------
    hr = safe_float(row.get('heart_rate'), 0)
    skin_temp = safe_float(row.get('skin_temp'), 0)
    posture = safe_str(row.get('posture'), 'unknown')
    eye_tracking = safe_str(row.get('eye_tracking'), 'unknown')
    voice = safe_str(row.get('voice'), 'unknown')
    task = safe_str(row.get('task'), 'unknown')
    task_duration = safe_int(row.get('task_duration'), 0)

    emotion = safe_str(row.get('face_emotion'), 'neutral').lower()
    stress = safe_str(row.get('face_stress'), 'low').lower()

    temp = safe_float(row.get('room_temp'), 0)
    cct_temp = safe_float(row.get('cct_temp'), 0)
    light_intensity = safe_float(row.get('light_intensity'), 0)
    humidity = safe_float(row.get('humidity'), 0)
    pressure = safe_float(row.get('pressure'), 0)

    reactor_status = safe_str(row.get('reactor_status'), 'normal').lower()

    # ------------- Extract optional outputs -------------
    # These may be None; compute_output_score has fallbacks.
    task_success = row.get('task_success', None)
    completion_time_s = row.get('completion_time_s', None)
    decision_accuracy = row.get('decision_accuracy', None)
    operational_stability = row.get('operational_stability', None)
    situational_awareness = row.get('situational_awareness', None)

    # ------------- HPSN (existing behavior) -------------
    load_hpsn()
    input_nodes = map_inputs_to_nodes(row)
    hpsn_suggestions = query_hpsn(input_nodes)
    hpsn_suggestions_text = "\n".join([f"- {action} (weight: {weight:.2f})" for action, weight in hpsn_suggestions]) if hpsn_suggestions else "—"

    # ------------- SAFETY MARGINS (existing) -------------
    safety_data = check_safety_margins(
        hr, skin_temp, temp, emotion, stress, cct_temp, light_intensity, humidity, pressure
    )

    # ------------- NEW: PSF Loads → Input Load -------------
    per_group_loads, input_load_score = compute_psf_loads(
        hr, skin_temp, posture, eye_tracking, voice,
        emotion, stress, temp, cct_temp, light_intensity, humidity, pressure,
        task, task_duration, reactor_status
    )

    # ------------- NEW: Output Score -------------
    output_score = compute_output_score(
        task_success=task_success,
        completion_time_s=completion_time_s,
        decision_accuracy=decision_accuracy,
        operational_stability=operational_stability,
        situational_awareness=situational_awareness,
        fallback_task_duration=task_duration
    )

    # ------------- NEW: Performance = Output / Input -------------
    perf_score, perf_state = compute_performance(per_group_loads, input_load_score, output_score)

    # ----------------- TOP KPIs -----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Performance Score (Output / Input)", f"{perf_score:.3f}", help="Higher = more efficient output given the current load.")
    c2.metric("Output Score (0–1)", f"{output_score:.2f}")
    c3.metric("Input Load (0–1)", f"{input_load_score:.2f}")

    # State banner
    if perf_state == "normal":
        st.success("State: NORMAL – Performance within expected range.")
    elif perf_state == "warning":
        st.warning("State: WARNING – Performance trending down. Review load drivers.")
    else:
        st.error("State: CRITICAL – Performance degraded. Immediate attention required.")

    # ----------------- DRIVERS TABLE -----------------
    st.markdown("### PSF Load Breakdown (by category)")
    drivers_df = pd.DataFrame.from_dict(per_group_loads, orient="index", columns=["load"]).sort_values("load", ascending=False)
    st.dataframe(drivers_df.round(3), use_container_width=True)

    # Log alert entries for recent display
    top_driver = drivers_df.index[0] if not drivers_df.empty else "unknown"
    if perf_state in ["warning", "critical"]:
        st.session_state.alert_log.append({
            "timestamp": row['timestamp'],
            "state": perf_state,
            "perf_score": round(perf_score, 3),
            "output_score": round(output_score, 2),
            "input_load": round(input_load_score, 2),
            "top_driver": safe_str(top_driver)
        })

    # ----------------- PERFORMANCE HIERARCHY (Safety Margins) -----------------
    st.markdown("### Performance Hierarchy Status (Safety Margins)")
    st.dataframe(pd.DataFrame(safety_data, columns=["ID", "Parameter", "Value", "Status"]).astype(str),
                 hide_index=True, use_container_width=True)

    # ----------------- RECENT ALERT LOG -----------------
    st.markdown("### Recent Alert Log")
    if st.session_state.alert_log:
        df_log = pd.DataFrame(st.session_state.alert_log[-10:])
        st.dataframe(df_log.astype(str), hide_index=True, use_container_width=True)
    else:
        st.success("No recent performance alerts.")

    # ----------------- TRENDS -----------------
    st.markdown("### Operator Trend Plots")
    emotion_map = {"happy": 1, "neutral": 2, "surprised": 3, "sad": 4, "fear": 5, "angry": 6}
    df['emotion_numeric'] = df['face_emotion'].str.lower().map(emotion_map)

    st.line_chart(df.set_index('timestamp')[['heart_rate', 'room_temp']])
    st.line_chart(df.set_index('timestamp')[['emotion_numeric']].rename(columns={'emotion_numeric': 'Emotion (coded)'}))

    with st.expander("ℹ Emotion Code Legend"):
        st.markdown("1=happy, 2=neutral, 3=surprised, 4=sad, 5=fear, 6=angry")

    # ----------------- LLaMA ANALYSIS (robust) -----------------
    st.markdown("### LLaMA Analysis")
    use_llama = st.toggle("Enable LLaMA analysis", value=False, help="Turn on if your local model server is running on :11434")

    llama_response = "[LLaMA disabled]" if not use_llama else "[Contacting local LLaMA…]"
    if use_llama:
        prompt = f"""You are an HPMS assistant in a nuclear control room. Analyze the operator's current state.

Heart rate: {hr} bpm
Skin Temp: {skin_temp} °C
Posture: {posture}
Eye Tracking: {eye_tracking}
Voice: {voice}
Task: {task}
Task Duration (s): {task_duration}
Emotion: {emotion}
Stress Level: {stress}
Room Temp: {temp} °C
CCT Temperature: {cct_temp} K
Light Intensity: {light_intensity} lux
Humidity: {humidity} %
Pressure: {pressure} hPa
Reactor Status: {reactor_status}

Performance:
- Output Score: {output_score:.2f}
- Input Load: {input_load_score:.2f}
- Performance (Output/Input): {perf_score:.3f} → {perf_state.upper()}
Top PSF driver: {top_driver}

HPSN Suggestions:
{hpsn_suggestions_text}

Provide a concise summary:
1) Performance risks (why)
2) Coupling concerns (human ↔ plant)
3) Suggested actions (operator + plant)
"""

        def call_llama(prompt_text, tries=2, timeout=25):
            last_err = None
            for i in range(tries):
                try:
                    r = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "llama2", "prompt": prompt_text, "stream": False},
                        timeout=timeout
                    )
                    r.raise_for_status()
                    return r.json().get("response", "[No response field]")
                except Exception as e:
                    last_err = e
            raise last_err

        with st.spinner("Contacting local LLaMA…"):
            try:
                llama_response = call_llama(prompt)
            except Exception as e:
                llama_response = f"[LLaMA call failed: {e}]"

    st.write(llama_response)

    # ----------------- HPSN UPDATE -----------------
    llama_output = {
        "operator_actions": [],
        "plant_actions": [],
        "confidence": 0.9,
        "related_concepts": input_nodes
    }
    if isinstance(llama_response, str) and not llama_response.startswith("[LLaMA call failed"):
        for line in llama_response.split("\n"):
            L = line.strip().lower()
            if L.startswith("- take ") or L.startswith("- practice ") or L.startswith("- perform "):
                llama_output["operator_actions"].append(line.strip("- ").strip())
            elif L.startswith("- lower ") or L.startswith("- increase ") or L.startswith("- adjust ") or L.startswith("- reassign "):
                llama_output["plant_actions"].append(line.strip("- ").strip())

    update_hpsn(llama_output)

    # ----------------- OPERATOR FEEDBACK -----------------
    st.markdown("### Operator Feedback")
    feedback = st.radio("Do you agree with the AI's analysis?", [
        "✅ Acknowledged and will take action",
        "🕒 Acknowledged but defer action",
        "❌ Disagree with assessment"
    ])
    notes = st.text_area("Optional Notes (e.g., context or follow-up actions)")

    if st.button("Submit Feedback"):
        effectiveness = {
            "✅ Acknowledged and will take action": 0.8,
            "🕒 Acknowledged but defer action": 0.5,
            "❌ Disagree with assessment": 0.2
        }
        st.session_state.feedback_log.append({
            "timestamp": datetime.datetime.now(),
            "response": feedback,
            "notes": notes,
            "perf_score": perf_score
        })
        update_hpsn(llama_output, {"effectiveness": effectiveness[feedback]})
        save_hpsn()
        st.success("Feedback submitted.")

    if st.session_state.feedback_log:
        st.markdown("### Feedback Log")
        st.dataframe(pd.DataFrame(st.session_state.feedback_log), use_container_width=True)

    # ----------------- HPSN SUGGESTIONS -----------------
    st.markdown("### HPSN Suggestions")
    st.write(hpsn_suggestions_text if hpsn_suggestions else "No HPSN suggestions available.")

    # ----------------- REPORT -----------------
    st.markdown("### Scenario Summary Report")
    report_text = f"""
Scenario Report: {task}
Timestamp: {row['timestamp']}

Inputs:
- Heart Rate: {hr} bpm
- Skin Temperature: {skin_temp} °C
- Posture: {posture}
- Eye Tracking: {eye_tracking}
- Voice: {voice}
- Emotion: {emotion}
- Facial Stress: {stress}
- Room Temp: {temp} °C
- CCT Temp: {cct_temp} K
- Light Intensity: {light_intensity} lux
- Humidity: {humidity} %
- Pressure: {pressure} hPa
- Reactor Status: {reactor_status}

Outputs (if provided / inferred):
- Task Success: {task_success}
- Completion Time (s): {completion_time_s if completion_time_s is not None else task_duration}
- Decision Accuracy: {decision_accuracy}
- Operational Stability: {operational_stability}
- Situational Awareness: {situational_awareness}

Scores:
- Output Score: {output_score:.2f}
- Input Load: {input_load_score:.2f}
- Performance (Output/Input): {perf_score:.3f} → {perf_state.upper()}
- Top PSF Driver: {top_driver}

HPSN Suggestions:
{hpsn_suggestions_text}

AI Summary (LLaMA):
{llama_response}
"""
    st.text_area("Generated Report:", report_text, height=320)
    st.download_button(
        label="📥 Download Scenario Report",
        data=str(report_text),
        file_name=f"scenario_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # ----------------- ENVIRONMENT COMMANDS -----------------
    st.markdown("### ⚙️ Environment Adjustment Commands")
    env_cmds = generate_environment_commands(cct_temp, light_intensity, humidity, pressure)
    if env_cmds:
        for cmd in env_cmds:
            st.code(cmd, language="python")
    else:
        st.caption("Environment within nominal ranges. No commands generated.")
else:
    st.info("Upload a CSV to begin. Required columns are:")
    st.code(", ".join([
        'timestamp','heart_rate','skin_temp','posture','eye_tracking','voice',
        'face_emotion','face_stress','task','task_duration',
        'room_temp','cct_temp','light_intensity','humidity','pressure'
    ]))
    st.caption("Optional outputs (recommended): task_success, completion_time_s, decision_accuracy, operational_stability, situational_awareness, reactor_status.")
