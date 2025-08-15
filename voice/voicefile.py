import os
import csv
import datetime
import math
import parselmouth
import numpy as np
import whisper
from pydub import AudioSegment
from pydub.utils import make_chunks

# Load whisper model for speech-to-text
model = whisper.load_model("base")

def analyze_chunk(audio_chunk_path):
    snd = parselmouth.Sound(audio_chunk_path)
    
    pitch = snd.to_pitch()
    mean_pitch = pitch.selected_array['frequency']
    mean_pitch = mean_pitch[mean_pitch > 0]
    pitch_hz = np.mean(mean_pitch) if len(mean_pitch) > 0 else 0
    
    # Jitter & shimmer estimates using Praat algorithms
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Gender heuristic
    gender = "Female" if pitch_hz > 160 else "Male" if pitch_hz > 50 else "Unknown"
    
    # Run Whisper speech recognition
    result = model.transcribe(audio_chunk_path)
    text = result["text"].strip()
    
    # Calculate WPM (words per minute)
    duration_seconds = snd.get_total_duration()
    word_count = len(text.split())
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    # Avg sentence length (words per sentence)
    sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    # Stutters (basic heuristic: count repeated words)
    words = text.lower().split()
    stutters = sum(words[i] == words[i+1] for i in range(len(words)-1))
    
    # Clarity heuristic
    clarity = "Normal pace - good clarity" if wpm >= 100 else "Too slow - possible hesitation or mental fatigue"
    
    # Voice state heuristic based on jitter/shimmer thresholds
    voice_state = "Fatigue" if jitter_local > 0.03 or shimmer_local > 1.2 else "Normal"
    
    return {
        "Pitch (Hz)": round(pitch_hz, 5),
        "Jitter": round(jitter_local, 8),
        "Shimmer": round(shimmer_local, 7),
        "Gender": gender,
        "Voice State": voice_state,
        "WPM": round(wpm, 1),
        "Avg Sentence Length": round(avg_sentence_length, 1),
        "Stutters": stutters,
        "Clarity": clarity,
        "Text": text
    }

def process_audio(input_file, chunk_length_ms=6000, report_name="report1v2"):
    audio = AudioSegment.from_file(input_file)
    chunks = make_chunks(audio, chunk_length_ms)
    
    # Prepare CSV output
    output_csv = report_name + ".csv"
    fieldnames = ["Chunk", "Operator", "Text", "Pitch (Hz)", "Jitter", "Shimmer", "Gender", 
                  "Voice State", "WPM", "Avg Sentence Length", "Stutters", "Clarity", "Timestamp"]
    
    start_time = datetime.datetime.now()
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            
            try:
                analysis = analyze_chunk(chunk_file)
            except Exception as e:
                # If analysis fails, fill with defaults
                analysis = {
                    "Pitch (Hz)": 27.5,
                    "Jitter": 0.0,
                    "Shimmer": 0.0,
                    "Gender": "Male",
                    "Voice State": "Normal",
                    "WPM": 0.0,
                    "Avg Sentence Length": 0,
                    "Stutters": 0,
                    "Clarity": "Too slow - possible hesitation or mental fatigue",
                    "Text": ""
                }
            
            timestamp = (start_time + datetime.timedelta(seconds=(i * chunk_length_ms / 1000))).strftime("%Y-%m-%d %H:%M:%S")
            
            row = {
                "Chunk": i,
                "Operator": report_name,
                **analysis,
                "Timestamp": timestamp
            }
            writer.writerow(row)
            
            os.remove(chunk_file)
            print(f"Processed chunk {i}")
    
    print(f"Analysis complete. CSV saved to {output_csv}")

# Usage
import os

def choose_audio_file():
    # Supported audio extensions
    audio_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    files = [f for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1].lower() in audio_exts]

    if not files:
        print("No audio files found in the current directory.")
        return None

    print("Available audio files:")
    for i, file in enumerate(files, 1):
        print(f"{i}: {file}")

    while True:
        choice = input(f"Choose a file (1-{len(files)}): ")
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        print("Invalid choice. Please try again.")

# Use it in your script like this:
chosen_file = choose_audio_file()
if chosen_file:
    process_audio(chosen_file, chunk_length_ms=6000, report_name="report1v2")
else:
    print("No file selected. Exiting.")
