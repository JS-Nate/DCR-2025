import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# --- Step 1: List available CSV files and select one ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the directory.")

print("Available CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1
if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

# --- Step 2: Load and parse CSV data ---
positions = []
timestamps = []

with open(selected_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["X"] or not row["Y"] or not row["Timestamp"]:
            continue
        try:
            x = int(row["X"])
            y = int(row["Y"])
            # Parse timestamp string to datetime object
            # Adjust the format to match your timestamp string
            t = datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            positions.append((x, y))
            timestamps.append(t)
        except (ValueError, KeyError, TypeError):
            continue

positions = np.array(positions)
timestamps = np.array(timestamps)

# Convert datetime timestamps to elapsed seconds from the first timestamp
elapsed_seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])

# --- Step 3: Calculate movement features ---
dx = np.diff(positions[:, 0])
dy = np.diff(positions[:, 1])
dt = np.diff(elapsed_seconds)
speed = np.hypot(dx, dy) / np.where(dt == 0, 1e-6, dt)

angles = np.arctan2(dy, dx)
d_angle = np.abs(np.diff(angles))
d_angle = np.where(d_angle > np.pi, 2 * np.pi - d_angle, d_angle)

ERRATIC_SPEED = np.percentile(speed, 90)
ERRATIC_ANGLE = np.percentile(d_angle, 90)
SLOW_SPEED = np.percentile(speed, 10)

erratic_speed_idx = np.where(speed > ERRATIC_SPEED)[0]
erratic_angle_idx = np.where(d_angle > ERRATIC_ANGLE)[0]
erratic_idx = np.unique(np.concatenate([erratic_speed_idx, erratic_angle_idx + 1]))

slow_idx = np.where(speed < SLOW_SPEED)[0]

# --- Plot 1: Mouse Path with Stress Indicators ---
plt.figure(figsize=(12, 8))
plt.plot(positions[:, 0], positions[:, 1], color="gray", alpha=0.5, label="Mouse Path")
plt.scatter(positions[erratic_idx, 0], positions[erratic_idx, 1], color="red", label="Erratic Movement", zorder=5)
plt.scatter(positions[slow_idx, 0], positions[slow_idx, 1], color="blue", label="Slow Movement", zorder=5)
plt.title("Mouse Movement with Stress Indicators")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Bar Graph of Movement Types ---
labels = ['Fast', 'Slow', 'Normal']
counts = [
    len(erratic_idx),
    len(slow_idx),
    len(positions) - len(np.unique(np.concatenate([erratic_idx, slow_idx])))
]
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['red', 'blue', 'gray'])
plt.title("Counts of Movement Types")
plt.ylabel("Number of Points")
plt.tight_layout()
plt.show()

# --- Plot 3: Line Graph of Speed Over Time ---
plt.figure(figsize=(12, 5))
plt.plot(elapsed_seconds[1:], speed, label="Speed (pixels/sec)", color="green")
plt.axhline(ERRATIC_SPEED, color='red', linestyle='--', label="Erratic Threshold")
plt.axhline(SLOW_SPEED, color='blue', linestyle='--', label="Slow Threshold")
plt.title("Mouse Speed Over Time")
plt.xlabel("Time (seconds from start)")
plt.ylabel("Speed (pixels/sec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Pie Chart of Movement Types ---
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=labels,
    colors=['red', 'blue', 'gray'],
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Movement Type Distribution")
plt.tight_layout()
plt.show()

# --- Plot 5: Line Progression Graph of Cumulative Distance ---
cumulative_distance = np.cumsum(np.hypot(dx, dy))
timestamps_mid = (elapsed_seconds[1:] + elapsed_seconds[:-1]) / 2

plt.figure(figsize=(12, 5))
plt.plot(timestamps_mid, cumulative_distance, label="Cumulative Distance", color="purple")
plt.title("Cumulative Mouse Movement Over Time")
plt.xlabel("Time (seconds from start)")
plt.ylabel("Total Distance Moved (pixels)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
