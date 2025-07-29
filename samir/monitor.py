import cv2
import pytesseract
import json
import pyautogui
import numpy as np
import time
from tkinter import Tk, messagebox

# === Config ===
SCREEN_REGION = (0, 0, 1920, 1080)
ROI_FILE = "rois_config.json"
RULES_FILE = "malfunction_rules.json"
DELAY = 1.0  # seconds

# === Load ROIs ===
with open(ROI_FILE, "r") as f:
    ROIS = json.load(f)
print(f"[INFO] Loaded {len(ROIS)} existing ROIs.")

# === Load Malfunction Rules ===
with open(RULES_FILE, "r") as f:
    raw_rules = json.load(f)
MALFUNCTION_RULES = {k: set(v) for k, v in raw_rules.items()}

# === Init Variables ===
previous_values = {}
active_alerts = set()
shown_malfunctions = set()

# === Hide tkinter root window for popup use ===
root = Tk()
root.withdraw()

def average_color(img):
    return tuple(map(int, img.mean(axis=0).mean(axis=0)))

def detect_change(name, roi_type, image, prev):
    if roi_type == "color":
        avg_color = average_color(image)
        is_white = all(c > 230 for c in avg_color)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
        orange_pixels = cv2.countNonZero(mask_orange)
        total_pixels = image.shape[0] * image.shape[1]
        orange_ratio = orange_pixels / total_pixels

        has_orange = orange_ratio > 0.05
        alert = not is_white and has_orange

        return {"avg_color": avg_color, "orange_ratio": round(orange_ratio, 2)}, alert

    elif roi_type == "text":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_text_img = cv2.bitwise_and(image, image, mask=red_mask)
        gray = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()

        if name not in prev:
            return text, bool(text)
        changed = text != prev[name] and text != ''
        return text, changed

def check_for_malfunctions():
    for malfunction, required_signals in MALFUNCTION_RULES.items():
        if required_signals.issubset(active_alerts) and malfunction not in shown_malfunctions:
            print(f"[⚠️ MALFUNCTION] {malfunction}")
            messagebox.showinfo("⚠️ Malfunction Detected", f"{malfunction} detected based on alerts.")
            shown_malfunctions.add(malfunction)

def monitor():
    print("📡 Monitoring started... Press Ctrl+C to stop.\n")
    try:
        while True:
            screenshot = pyautogui.screenshot(region=SCREEN_REGION)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            for roi in ROIS:
                name = roi["name"]
                x, y, w, h = roi["region"]
                roi_type = roi["type"]

                cropped = frame[y:y+h, x:x+w]
                result, changed = detect_change(name, roi_type, cropped, previous_values)

                if changed:
                    print(f"[🔄 CHANGED] {name}: {result}")
                    active_alerts.add(name)

                previous_values[name] = result

            check_for_malfunctions()
            time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user.")

if __name__ == "__main__":
    monitor()
