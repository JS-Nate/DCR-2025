import streamlit as st
import pandas as pd
import time
import requests
import datetime
from analysis import calculate_performance_score, check_safety_margins
from hpsn import load_hpsn, query_hpsn, update_hpsn, map_inputs_to_nodes, save_hpsn

# Scenario Report Generator
def create_scenario_report(df: pd.DataFrame, llama_response: str, hpsn_suggestions: list) -> str:
    latest = df.iloc[-1]
    hpsn_text = "\n".join([f"- {action} (weight: {weight:.2f})" for action, weight in hpsn_suggestions])
    return f"""
Scenario Report: {latest['task']}
Timestamp: {latest['timestamp']}

- Heart Rate: {latest['heart_rate']} bpm
- EEG: {latest['eeg_signal']}
- Emotion: {latest['face_emotion']}
- Task: {latest['task']}
- Room Temp: {latest['room_temp']}°C
- Light Temp: {latest.get('light_temp', 'N/A')}
- Light Intensity: {latest.get('light_intensity', 'N/A')}
- Humidity: {latest.get('humidity', 'N/A')}%
- Pressure: {latest.get('pressure', 'N/A')} kPa

AI Summary (LLaMA):
{llama_response}

HPSN Suggestions:
{hpsn_text}

Performance Insights:
This scenario captures the operator executing the "{latest['task']}" task under the measured physiological and environmental conditions.
The emotion and EEG suggest a state of "{latest['face_emotion']}" and "{latest['eeg_signal']}" brainwave dominance.
""".strip()

st.set_page_config(page_title="SMR HPMS Dashboard", layout="wide")
st.title("SMR Human Performance Management System (HPMS)")

# Auto-refresh simulation
AUTO_REFRESH_SEC = 5
st.markdown(f"Auto-refreshing every **{AUTO_REFRESH_SEC} seconds**...")
time.sleep(AUTO_REFRESH_SEC)

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp")

    # Most recent row
    row = df.iloc[-1]
    hr = row['heart_rate']
    eeg = row['eeg_signal']
    temp = row['room_temp']
    task = row['task']
    emotion = row['face_emotion']
    light_temp = row.get('light_temp', 'unknown')
    light_intensity = row.get('light_intensity', 'unknown')
    humidity = row.get('humidity', 'unknown')
    pressure = row.get('pressure', 'unknown')

    # Load HPSN
    load_hpsn()

    # Map inputs to HPSN nodes
    input_nodes = map_inputs_to_nodes(row)
    
    # Query HPSN for suggestions
    hpsn_suggestions = query_hpsn(input_nodes)
    hpsn_suggestions_text = "\n".join([f"- {action} (weight: {weight:.2f})" for action, weight in hpsn_suggestions])

    # Calculate performance and check margins
    score = calculate_performance_score(hr, eeg, temp, task, emotion, light_temp, light_intensity, humidity, pressure)
    safety_data = check_safety_margins(hr, eeg, temp, emotion, light_temp, light_intensity, humidity, pressure)

    st.metric("Performance Score", score)

    # Performance Hierarchy
    st.markdown("### Performance Hierarchy Status")
    df_margin = pd.DataFrame(safety_data, columns=["ID", "Parameter", "Value", "Status"])
    st.dataframe(df_margin.astype(str), hide_index=True, use_container_width=True)

    # Alert Log
    st.markdown("### Recent Alert Log")
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
        df_log = pd.DataFrame(st.session_state.alert_log[-10:])
        st.dataframe(df_log.astype(str), hide_index=True, use_container_width=True)
    else:
        st.success("No recent margin violations.")

    # Trend Plots
    st.markdown("### Operator Trend Plots")
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

    with st.expander("ℹCoded Value Legend"):
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

    # LLaMA Response
    st.markdown("### LLaMA Response")
    prompt = f"""You are an HPMS assistant in a nuclear control room. Analyze the operator's current state.

    Heart rate: {hr} bpm
    EEG: {eeg}
    Room temp: {temp}°C
    Emotion: {emotion}
    Task: {task}
    Light temperature: {light_temp}
    Light intensity: {light_intensity}
    Humidity: {humidity}%
    Pressure: {pressure} kPa

    HPSN Suggestions:
    {hpsn_suggestions_text}

    Provide a short summary on:
    1. Performance risks
    2. Coupling concerns
    3. Suggested actions (operator and plant)
    """

    llama_response = "[No response received]"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False}
        )
        result = response.json()
        llama_response = result.get("response", "[No response received from LLaMA]")
        st.write(llama_response)
    except Exception as e:
        st.error(f"LLaMA API call failed: {e}")

    # Parse Llama response for structured output (simplified, assumes Llama outputs structured text)
    # In practice, use NLP or regex to extract actions and concepts
    llama_output = {
        "operator_actions": [],
        "plant_actions": [],
        "confidence": 0.9,
        "related_concepts": input_nodes
    }
    if "suggested actions" in llama_response.lower():
        # Example parsing (adjust based on actual Llama output format)
        lines = llama_response.split("\n")
        for line in lines:
            line = line.lower().strip()
            if line.startswith("- take ") or line.startswith("- practice "):
                llama_output["operator_actions"].append(line[2:].strip())
            elif line.startswith("- lower ") or line.startswith("- increase ") or line.startswith("- adjust "):
                llama_output["plant_actions"].append(line[2:].strip())

    # Update HPSN with Llama's recommendations
    update_hpsn(llama_output)

    # Operator Feedback
    st.markdown("### Operator Feedback")
    feedback = st.radio("Do you agree with the AI's analysis?", [
        "✅ Acknowledged and will take action",
        "🕒 Acknowledged but defer action",
        "❌ Disagree with assessment"
    ])
    notes = st.text_area("Optional Notes (e.g., context or follow-up actions)")

    if st.button("Submit Feedback"):
        effectiveness = {"✅ Acknowledged and will take action": 0.8, "🕒 Acknowledged but defer action": 0.5, "❌ Disagree with assessment": 0.2}
        st.session_state.feedback_log.append({
            "timestamp": datetime.datetime.now(),
            "response": feedback,
            "notes": notes,
            "score": score
        })
        # Update HPSN with feedback
        update_hpsn(llama_output, {"effectiveness": effectiveness[feedback]})
        save_hpsn()
        st.success("Feedback submitted.")

    if st.session_state.feedback_log:
        st.markdown("### Feedback Log")
        st.dataframe(pd.DataFrame(st.session_state.feedback_log), use_container_width=True)

    # Display HPSN Suggestions
    st.markdown("### HPSN Suggestions")
    if hpsn_suggestions:
        st.write(hpsn_suggestions_text)
    else:
        st.write("No HPSN suggestions available.")

    # Scenario Report Generator
    st.markdown("### Scenario Summary Report")
    report_text = create_scenario_report(df, llama_response, hpsn_suggestions)
    st.text_area("Generated Report:", report_text, height=300)
    st.download_button(
        label="📥 Download Scenario Report",
        data=report_text,
        file_name=f"scenario_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )