import time
import math
import csv
import threading
from pynput import mouse, keyboard
import matplotlib.pyplot as plt
from datetime import datetime

# === Prompt for scenario name and generate CSV filename ===
scenario_name = input("Enter a name for this mouse tracking session: ").strip().replace(" ", "_")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"mouse_detection-{scenario_name}-{timestamp}.csv"

mouse_data = []
stop_flag = False

def on_move(x, y):
    if stop_flag:
        return False
    timestamp = time.time()
    mouse_data.append((timestamp, x, y))
    print(f"Moved to ({x}, {y}) at {timestamp:.2f}")

def on_click(x, y, button, pressed):
    return not stop_flag

def on_press(key):
    global stop_flag
    if key == keyboard.Key.esc:
        print("\nEscape pressed. Stopping...")
        stop_flag = True
        return False

def compute_speed(data):
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

def compute_jerkiness(data):
    angle_changes = []
    for i in range(2, len(data)):
        _, x0, y0 = data[i - 2]
        _, x1, y1 = data[i - 1]
        _, x2, y2 = data[i]

        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1

        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)

        if mag1 * mag2 == 0:
            continue

        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(min(cos_theta, 1), -1)
        angle = math.acos(cos_theta)
        angle_changes.append(angle)
    return angle_changes

def compute_idle_time(data, threshold=2):
    idle_periods = 0
    for i in range(1, len(data)):
        t1, x1, y1 = data[i - 1]
        t2, x2, y2 = data[i]
        if math.hypot(x2 - x1, y2 - y1) < 1.0 and (t2 - t1) > threshold:
            idle_periods += 1
    return idle_periods

def compute_micro_movements(data, threshold=3):
    small_moves = 0
    for i in range(1, len(data)):
        _, x1, y1 = data[i - 1]
        _, x2, y2 = data[i]
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < threshold:
            small_moves += 1
    return small_moves

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

def save_to_csv(data, speeds, jerkiness, idle_count, micro_moves, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'X', 'Y', 'Speed'])

        for i in range(len(data)):
            timestamp, x, y = data[i]
            speed = speeds[i - 1] if i > 0 and i - 1 < len(speeds) else ''
            writer.writerow([timestamp, x, y, speed])

        writer.writerow([])
        writer.writerow(['Summary Metrics'])
        writer.writerow(['Average Speed', sum(speeds) / len(speeds) if speeds else 0])
        writer.writerow(['Average Jerkiness (radians)', sum(jerkiness) / len(jerkiness) if jerkiness else 0])
        writer.writerow(['Idle Periods (>2s)', idle_count])
        writer.writerow(['Micro Movements (<3px)', micro_moves])

    print(f"\nData saved to '{filename}'")

# === Start Mouse and Keyboard Listeners ===

print("Tracking mouse movement... (Press ESC to stop)")

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

with mouse.Listener(on_move=on_move, on_click=on_click) as mouse_listener:
    while not stop_flag:
        time.sleep(0.1)
    mouse_listener.stop()

keyboard_listener.join()

# === After Collection: Analyze and Save ===

if len(mouse_data) > 2:
    speeds, times = compute_speed(mouse_data)
    jerkiness = compute_jerkiness(mouse_data)
    idle_count = compute_idle_time(mouse_data)
    micro_moves = compute_micro_movements(mouse_data)

    print("\n--- Stress-Related Metrics ---")
    print(f"Average Speed: {sum(speeds)/len(speeds):.2f} px/sec")
    print(f"Average Jerkiness: {sum(jerkiness)/len(jerkiness):.3f} radians")
    print(f"Idle Periods (>2s): {idle_count}")
    print(f"Micro Movements (<3px): {micro_moves}")

    plot_results(mouse_data, speeds, times)
    save_to_csv(mouse_data, speeds, jerkiness, idle_count, micro_moves, csv_filename)

else:
    print("Not enough data to analyze.")
