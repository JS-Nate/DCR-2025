import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the CSV
positions = []
timestamps = []

with open("mouse/mouse_stress_data.csv", "r") as f:
    for line in f:
        match = re.match(r"Moved to \(([-\d]+), ([-\d]+)\) at ([\d\.]+)", line)
        if match:
            x, y, t = int(match.group(1)), int(match.group(2)), float(match.group(3))
            positions.append((x, y))
            timestamps.append(t)

positions = np.array(positions)
timestamps = np.array(timestamps)

# Calculate speed (pixels/sec) and direction change (radians)
dx = np.diff(positions[:, 0])
dy = np.diff(positions[:, 1])
dt = np.diff(timestamps)
speed = np.hypot(dx, dy) / np.where(dt == 0, 1e-6, dt)  # avoid division by zero

# Direction (angle in radians)
angles = np.arctan2(dy, dx)
d_angle = np.abs(np.diff(angles))
d_angle = np.where(d_angle > np.pi, 2 * np.pi - d_angle, d_angle)  # wrap-around

# Heuristics for stress markers
ERRATIC_SPEED = np.percentile(speed, 90)  # top 10% speed
ERRATIC_ANGLE = np.percentile(d_angle, 90)  # top 10% direction change
SLOW_SPEED = np.percentile(speed, 10)  # bottom 10% speed

# Find erratic movement (either high speed or high direction change)
erratic_speed_idx = np.where(speed > ERRATIC_SPEED)[0]
erratic_angle_idx = np.where(d_angle > ERRATIC_ANGLE)[0]
erratic_idx = np.unique(np.concatenate([erratic_speed_idx, erratic_angle_idx + 1]))  # shift angle index to match position

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
plt.plot(timestamps[1:], speed, label="Speed (pixels/sec)", color="green")
plt.axhline(ERRATIC_SPEED, color='red', linestyle='--', label="Erratic Threshold")
plt.axhline(SLOW_SPEED, color='blue', linestyle='--', label="Slow Threshold")
plt.title("Mouse Speed Over Time")
plt.xlabel("Timestamp")
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
timestamps_mid = (timestamps[1:] + timestamps[:-1]) / 2  # midpoint for each segment

plt.figure(figsize=(12, 5))
plt.plot(timestamps_mid, cumulative_distance, label="Cumulative Distance", color="purple")
plt.title("Cumulative Mouse Movement Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Total Distance Moved (pixels)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
