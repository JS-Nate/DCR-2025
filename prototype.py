import pygame
import tkinter as tk
from tkinter import messagebox
import random
import asyncio
import platform
import numpy as np
import threading
import time
import pandas as pd
import os

# Initialize Pygame for alarm task
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("SMR Alarm Task")
clock = pygame.time.Clock()
FPS = 60

alarms = ["red", "green"]
current_alarm = None
alarm_time = 0
response_correct = 0
response_errors = 0
alarm_freq = 5  

hrv_data = 50  
eeg_beta = 10  
gsr_data = 3 
stress_score = 0
monitoring_aware = True  

log_file = "stress_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Timestamp", "HRV", "EEG_Beta", "GSR", "Stress_Score"]).to_csv(log_file, index=False)

root = tk.Tk()
root.title("HPMS Dashboard")
stress_label = tk.Label(root, text="Stress Score: 0")
stress_label.pack()
alert_label = tk.Label(root, text="Alerts: None")
alert_label.pack()
monitoring_label = tk.Label(root, text="Monitoring: " + ("On" if monitoring_aware else "Off"))
monitoring_label.pack()
simplify_button = tk.Button(root, text="Simplify Task", command=lambda: globals().update(alarm_freq=10))
simplify_button.pack()

def simulate_sensors():
    global hrv_data, eeg_beta, gsr_data, stress_score
    while True:
        hrv_data = max(20, min(80, hrv_data + random.uniform(-5, 5)))  # Simulate HRV
        eeg_beta = max(5, min(20, eeg_beta + random.uniform(-2, 2)))  # Simulate EEG
        gsr_data = max(1, min(5, gsr_data + random.uniform(-0.5, 0.5)))  # Simulate GSR
        stress_score = int(100 * ((1 - (hrv_data - 20) / 60) + eeg_beta / 20 + gsr_data / 5) / 3)
        stress_label.config(text=f"Stress Score: {stress_score}")
        if stress_score > 70:
            alert_label.config(text="Alerts: High Stress Detected!")
            messagebox.showwarning("HPMS Alert", "High stress detected. Simplify task?")
        elif stress_score < 30:
            alert_label.config(text="Alerts: Nice focus!")
            messagebox.showinfo("HPMS Alert", "Well done, staying calm!")
        else:
            alert_label.config(text="Alerts: None")
        log_data = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "HRV": hrv_data,
            "EEG_Beta": eeg_beta,
            "GSR": gsr_data,
            "Stress_Score": stress_score
        }
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
        time.sleep(1)

threading.Thread(target=simulate_sensors, daemon=True).start()

async def main():
    global current_alarm, alarm_time, response_correct, response_errors, alarm_freq
    running = True
    while running:
        screen.fill((255, 255, 255))  
        if current_alarm:
            color = (255, 0, 0) if current_alarm == "red" else (0, 255, 0)
            pygame.draw.rect(screen, color, (150, 150, 100, 100))
        pygame.display.flip()

        if time.time() - alarm_time > alarm_freq:
            current_alarm = random.choice(alarms)
            alert_label.config(text=f"Alerts: Respond to {current_alarm.upper()} alarm!")
            alarm_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if current_alarm:
                    if (event.key == pygame.K_r and current_alarm == "red") or \
                       (event.key == pygame.K_g and current_alarm == "green"):
                        response_correct += 1
                        alert_label.config(text="Alerts: Correct response!")
                    else:
                        response_errors += 1
                        alert_label.config(text="Alerts: Error detected!")
                    current_alarm = None

        root.update()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

pygame.quit()
root.destroy()