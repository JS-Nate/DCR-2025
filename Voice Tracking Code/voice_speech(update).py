import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import opensmile
import numpy as np
import time
import os
import pandas as pd
import spacy
import re
import tempfile

# --- Setup ---
operator = input("Operator name: ")
samplerate = 16000
chunk_duration = 5  # seconds
recognizer = sr.Recognizer()
nlp = spacy.load("en_core_web_sm")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# --- Select Input Device ---
print("\nAvailable input devices:")
input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
for i, device in enumerate(input_devices):
    print(f"{i}: {device['name']}")

device_index = int(input("\nSelect the input device index: "))
selected_device = input_devices[device_index]
print(f"\nUsing input device: {selected_device['name']}")

# --- Helper Functions ---
def classify_voice(pitch, jitter, shimmer):
    if pitch > 216.9 and jitter > 1.18 and shimmer > 6.82:
        return "Stressed"
    elif pitch < 140 and shimmer > 0.5:
        return "Fatigue"
    return "Normal"

def classify_gender(pitch_hz):
    return "Male" if pitch_hz < 165 else "Female" if pitch_hz > 180 else "Uncertain"

def estimate_clarity(text, duration):
    words = text.split()
    wpm = len(words) / (duration / 60)  # Words per minute
    if wpm < 100:
        return "Too slow - possible hesitation or mental fatigue"
    elif 100 <= wpm <= 200:
        return "Normal pace - good clarity"
    elif wpm > 200:
        return "Too fast - may indicate possible stress or anxiety"

def analyze_chunk(audio_data, chunk_index):
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_data, samplerate)
        path = tmpfile.name

    # Speech recognition
    with sr.AudioFile(path) as source:
        try:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        except:
            text = ""

    # Voice features
    features = smile.process_file(path)
    pitch_semitone = features["F0semitoneFrom27.5Hz_sma3nz_amean"].values[0]
    pitch_hz = 27.5 * (2 ** (pitch_semitone / 12))
    jitter = features["jitterLocal_sma3nz_amean"].values[0]
    shimmer = features["shimmerLocaldB_sma3nz_amean"].values[0]
    gender = classify_gender(pitch_hz)
    state = classify_voice(pitch_hz, jitter, shimmer)

    # Clarity metrics
    duration = len(audio_data) / samplerate
    words = text.split()
    wpm = len(words) / (duration / 60) if duration > 0 else 0
    sentence_lengths = [len(s.split()) for s in text.split('.') if s.strip()]
    avg_sent_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    stutters = len(re.findall(r"\b(\w+)(\s+\1)+\b", text.lower()))
    clarity_estimate = estimate_clarity(text, duration)

    # Print output
    print(f"\n[Chunk {chunk_index}] Text: {text}")
    print(f"Pitch: {pitch_hz:.2f} Hz | Jitter: {jitter:.4f} | Shimmer: {shimmer:.4f}")
    print(f"Gender: {gender} | Voice State: {state}")
    print(f"WPM: {wpm:.2f} | Avg Sentence Length: {avg_sent_len:.2f} | Stutters: {stutters}")
    print(f"Clarity Estimate: {clarity_estimate}")

    # Save to CSV
    df = pd.DataFrame([{
        "Chunk": chunk_index,
        "Operator": operator,
        "Text": text,
        "Pitch (Hz)": pitch_hz,
        "Jitter": jitter,
        "Shimmer": shimmer,
        "Gender": gender,
        "Voice State": state,
        "WPM": wpm,
        "Avg Sentence Length": avg_sent_len,
        "Stutters": stutters,
        "Clarity": clarity_estimate,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    csv_file = f"voice_speech_tracking_{operator}.csv"
    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

    os.remove(path)

# --- Interval Recording Loop ---
chunk_index = 0
print("\nRecording ... Press Ctrl+C to stop.\n")

try:
    while True:
        print(f"\n[Chunk {chunk_index}] Recording...")
        audio = sd.rec(int(chunk_duration * samplerate),
                       samplerate=samplerate,
                       channels=1,
                       device=selected_device['index'])
        sd.wait()
        audio = audio.flatten()
        if np.max(np.abs(audio)) < 0.01:
            print("Silence detected, skipping...")
        else:
            analyze_chunk(audio, chunk_index)
        chunk_index += 1
except KeyboardInterrupt:
    print("\nRecording stopped.")
