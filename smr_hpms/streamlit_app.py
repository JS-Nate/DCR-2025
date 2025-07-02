# streamlit_app.py
import streamlit as st
import pandas as pd
import time
from analysis import calculate_performance_score, check_safety_margins

st.set_page_config(page_title="SMR HPMS Dashboard", layout="wide")
st.title("🧠 SMR Human Performance Management System (HPMS)")

AUTO_REFRESH_SEC = 5
st.markdown(f"🔄 Auto-refreshing every **{AUTO_REFRESH_SEC} seconds**...")
time.sleep(AUTO_REFRESH_SEC)

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp")

    row = df.iloc[-1]
    hr = row['heart_rate']
    eeg = row['eeg_signal']
    temp = row['room_temp']
    task = row['task']
    emotion = row['face_emotion']

    score = calculate_performance_score(hr, eeg, temp, task, emotion)
    safety_data = check_safety_margins(hr, eeg, temp, emotion)

    st.metric("Performance Score", score)

    st.markdown("### 📐 Performance Hierarchy Status")
    df_margin = pd.DataFrame(safety_data, columns=["ID", "Parameter", "Value", "Status"])
    # st.dataframe(df_margin, hide_index=True, use_container_width=True)
    # Force all values to string to prevent Arrow conversion crash
    df_margin_display = df_margin.astype(str)
    st.dataframe(df_margin_display, hide_index=True, use_container_width=True)




    st.markdown("### 📝 Recent Alert Log")
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []

    for item in safety_data:
        if "⚠️" in item[3] or "🚨" in item[3]:
            st.session_state.alert_log.append({
                "timestamp": row['timestamp'],
                "id": item[0],
                "parameter": item[1],
                "value": item[2],
                "status": item[3]
            })

    if st.session_state.alert_log:
        # df_log = pd.DataFrame(st.session_state.alert_log[-10:])
        # st.dataframe(df_log, hide_index=True, use_container_width=True)

        df_log = pd.DataFrame(st.session_state.alert_log[-10:])
        df_log_display = df_log.astype(str)
        st.dataframe(df_log_display, hide_index=True, use_container_width=True)


    else:
        st.success("No recent margin violations.")

    st.markdown("### 📈 Operator Trend Plots")

    eeg_map = {
        "alpha-dominant": 1,
        "beta-dominant": 2,
        "theta-dominant": 3,
        "alpha-suppressed": 4
    }
    emotion_map = {
        "happy": 1,
        "neutral": 2,
        "surprised": 3,
        "sad": 4,
        "fear": 5,
        "angry": 6
    }

    df['eeg_numeric'] = df['eeg_signal'].str.lower().map(eeg_map)
    df['emotion_numeric'] = df['face_emotion'].str.lower().map(emotion_map)

    st.line_chart(df.set_index('timestamp')[['heart_rate', 'room_temp']])
    st.line_chart(df.set_index('timestamp')[['eeg_numeric']].rename(columns={'eeg_numeric': 'EEG (coded)'}))
    st.line_chart(df.set_index('timestamp')[['emotion_numeric']].rename(columns={'emotion_numeric': 'Emotion (coded)'}))

    with st.expander("ℹ️ Coded Value Legend"):
        st.markdown("""
        **EEG Codes**  
        1 = alpha-dominant  
        2 = beta-dominant  
        3 = theta-dominant  
        4 = alpha-suppressed  

        **Emotion Codes**  
        1 = happy  
        2 = neutral  
        3 = surprised  
        4 = sad  
        5 = fear  
        6 = angry
        """)
