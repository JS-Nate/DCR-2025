import time
import math
from pynput.mouse import Listener
import matplotlib.pyplot as plt

mouse_data = []

def on_move(x, y):
    timestamp = time.time()
    mouse_data.append((timestamp, x, y))
    print(f"Moved to ({x}, {y}) at {timestamp:.2f}")

def on_click(x, y, button, pressed):
    if not pressed:
        return False 

def compute_features(data):
    speeds = []
    times = []
    for i in range(1, len(data)):
        t1, x1, y1 = data[i - 1]
        t2, x2, y2 = data[i]
        dt = t2 - t1
        if dt == 0:
            continue
        dist = math.hypot(x2 - x1, y2 - y1)
        speed = dist / dt
        speeds.append(speed)
        times.append(t2)
    return speeds, times

def plot_results(data, speeds, times):
    x_vals = [x for _, x, _ in data]
    y_vals = [y for _, _, y in data]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', alpha=0.6)
    plt.title('Mouse Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    if speeds:
        plt.plot(times[1:], speeds[1:], color='orange')
        plt.title('Mouse Speed Over Time')
        plt.xlabel('Time')
        plt.ylabel('Speed (pixels/sec)')

    plt.tight_layout()
    plt.show()

print("Tracking mouse movement... (Click mouse to stop)")
with Listener(on_move=on_move, on_click=on_click) as listener:
    listener.join()

if len(mouse_data) > 2:
    speeds, times = compute_features(mouse_data)
    plot_results(mouse_data, speeds, times)
else:
    print("Not enough data to analyze.")
