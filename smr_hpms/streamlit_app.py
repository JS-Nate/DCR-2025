import streamlit as st
import pandas as pd
import time
from datetime import datetime
from analysis import calculate_performance_score, detect_coupling, get_alert_level
from llama_utils import query_llama
from analysis import check_safety_margins


st.set_page_config(page_title="SMR HPMS Dashboard", layout="centered")

st.title("SMR Human Performance Management System")

# Auto-refresh every N seconds
AUTO_REFRESH_SEC = 5
st.markdown(f"🔄 Auto-refreshing every **{AUTO_REFRESH_SEC} seconds**...")
time.sleep(AUTO_REFRESH_SEC)


st.markdown("Upload your CSV data to assess operator performance in a control room environment.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data (Most Recent Row):")
    st.dataframe(df.tail(1))

    row = df.iloc[-1]
    hr = row['heart_rate']
    eeg = row['eeg_signal']
    temp = row['room_temp']
    task = row['task']
    emotion = row['face_emotion']


    # score = calculate_performance_score(hr, eeg, temp, task)
    score = calculate_performance_score(hr, eeg, temp, task, emotion)
    safety_alerts = check_safety_margins(hr, eeg, temp, emotion)
    coupling = detect_coupling(hr, eeg, temp)
    alert_level, alert_reason = get_alert_level(score)

    st.markdown(f"### Performance Score: `{score}`")
    st.markdown(f"### Alert Level: `{alert_level}`")
    st.markdown(f"**Face Emotion:** {emotion}")
    st.info(alert_reason)
    st.markdown("### 📏 Safety Margin Check")
    if not safety_alerts:
        st.success("✅ All safety margins within range.")
    else:
        for alert in safety_alerts:
            if "🚨" in alert:
                st.error(alert)
            elif "⚠️" in alert:
                st.warning(alert)

    st.markdown("### 📐 Performance Hierarchy Status")
    df_margin = pd.DataFrame(safety_alerts, columns=["ID", "Parameter", "Value", "Status"])
    st.dataframe(df_margin, hide_index=True, use_container_width=True)


    st.markdown("### Coupling Issues Detected:")
    for issue in coupling:
        st.write(f"- {issue}")





    # Alert log section
    st.markdown("### 📝 Recent Alert Log")

    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []

    # Append only the violated margins
    for item in safety_alerts:
        if "⚠️" in item[3] or "🚨" in item[3]:
            st.session_state.alert_log.append({
                "timestamp": row['timestamp'],
                "id": item[0],
                "parameter": item[1],
                "value": item[2],
                "status": item[3]
            })

    # Show last 10 alerts
    if st.session_state.alert_log:
        df_log = pd.DataFrame(st.session_state.alert_log[-10:])
        st.dataframe(df_log, hide_index=True, use_container_width=True)
    else:
        st.success("No recent margin violations.")












    prompt = f"""
You're acting as a Human Performance Management System (HPMS) assistant in an SMR control room.

Inputs:
- Heart Rate: {hr} bpm
- EEG Signal: {eeg}
- Room Temp: {temp} °C
- Task: {task}
- Observed Face Emotion: {emotion}
- Performance Score (0-1): {score}
- Detected Coupling Issues: {', '.join(coupling)}
- Alert Level: {alert_level}

Instructions:
1. Briefly justify the alert level.
2. Analyze how facial emotion and physiological state impact performance.
3. Recommend human, environmental, or operational interventions.
"""


    with st.spinner("Analyzing with LLaMA..."):
        llama_output = query_llama(prompt)

    st.markdown("### LLaMA Recommendations")
    st.write(llama_output)
