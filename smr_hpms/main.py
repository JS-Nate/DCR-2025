import pandas as pd
from analysis import calculate_performance_score, detect_coupling, get_alert_level
import subprocess
import os

def read_inputs(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, "inputs", file_path)
    return pd.read_csv(full_path)

def query_llama(prompt: str):
    process = subprocess.Popen(['ollama', 'run', 'llama2'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               encoding='utf-8')  # Fix UnicodeDecodeError
    response, _ = process.communicate(prompt)
    return response.strip()


if __name__ == "__main__":
    df = read_inputs("sample_data.csv")
    row = df.iloc[-1]

    # Extract input parameters
    hr = row['heart_rate']
    eeg = row['eeg_signal']
    temp = row['room_temp']
    task = row['task']

    # Performance analysis
    perf_score = calculate_performance_score(hr, eeg, temp, task)
    coupling = detect_coupling(hr, eeg, temp)
    alert_level, alert_reason = get_alert_level(perf_score)

    # Structured LLaMA prompt
    prompt = f"""
You're acting as an HPMS assistant in a Small Modular Reactor (SMR) control room.

Inputs:
- Heart Rate: {hr} bpm
- EEG Signal: {eeg}
- Room Temp: {temp} °C
- Task: {task}
- Performance Score (0-1): {perf_score}
- Detected Coupling Issues: {', '.join(coupling)}
- Alert Level: {alert_level}

Instructions:
1. Briefly justify the alert level.
2. Summarize performance risk.
3. Recommend human, environmental, or operational actions.
"""

    # LLaMA output
    llama_response = query_llama(prompt)

    # Print all results
    print("\n=== LLaMA RESPONSE ===")
    print(llama_response)

    print("\n=== SUMMARY ===")
    print(f"Performance Score: {perf_score}")
    print(f"Alert Level: {alert_level} — {alert_reason}")
    print("Detected Coupling:", "; ".join(coupling))
