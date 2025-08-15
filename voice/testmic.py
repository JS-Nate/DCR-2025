import speech_recognition as sr
import pyttsx3
import os
import re

def list_microphones():
    print("Available Microphones:\n")
    mics = sr.Microphone.list_microphone_names()
    for idx, mic in enumerate(mics):
        print(f"{idx}: {mic}")
    return mics

def choose_microphone(mics):
    while True:
        try:
            index = int(input("\nEnter the number of the microphone you want to use: "))
            if 0 <= index < len(mics):
                print(f"Using microphone: {mics[index]}")
                return index, mics[index]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

def sanitize_filename(name):
    # Remove characters that are unsafe for filenames and replace spaces with underscores
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def recognize_and_speak(mic_index, mic_name):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    recording_counter = 1
    save_folder = "recordings"
    os.makedirs(save_folder, exist_ok=True)

    clean_mic_name = sanitize_filename(mic_name)

    with sr.Microphone(device_index=mic_index) as source:
        print("\nSpeak something! (Press Ctrl+C to exit)")
        try:
            while True:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source)

                # Save audio to WAV file with mic name included
                filename = os.path.join(
                    save_folder,
                    f"recording_{clean_mic_name}_{recording_counter}.wav"
                )
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())
                print(f"Saved audio to {filename}")

                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")

                engine.say(text)
                engine.runAndWait()

                recording_counter += 1
        except KeyboardInterrupt:
            print("\nExiting program. Goodbye!")

if __name__ == "__main__":
    microphones = list_microphones()
    selected_index, selected_name = choose_microphone(microphones)
    recognize_and_speak(selected_index, selected_name)
