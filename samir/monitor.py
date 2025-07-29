import cv2
import pytesseract
import json
import pyautogui
import numpy as np
import time

SCREEN_REGION = (0, 0, 1920, 1080)
ROI_FILE = "rois_config.json"
DELAY = 1.0  # seconds

# Load existing ROIs
with open(ROI_FILE, "r") as f:
    ROIS = json.load(f)
print(f"[INFO] Loaded {len(ROIS)} existing ROIs.")

previous_values = {}

def average_color(img):
    return tuple(map(int, img.mean(axis=0).mean(axis=0)))

def detect_change(name, roi_type, image, prev):
    # if roi_type == "color":
    #     current = average_color(image)
    #     if name not in prev:
    #         return current, False
    #     changed = np.linalg.norm(np.array(current) - np.array(prev[name])) > 10
    #     return current, changed
    

    if roi_type == "color":
        # Define what counts as orange in BGR (since OpenCV uses BGR)
        orange_lower = np.array([0, 100, 200])    # light orange (BGR)
        orange_upper = np.array([100, 180, 255])  # dark orange (BGR)

        # Consider region non-white if average brightness is below threshold
        avg_color = average_color(image)
        is_white = all(c > 230 for c in avg_color)

        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))  # HSV range for orange
        orange_pixels = cv2.countNonZero(mask_orange)
        total_pixels = image.shape[0] * image.shape[1]

        orange_ratio = orange_pixels / total_pixels

        has_orange = orange_ratio > 0.05  # 5% of region is orange
        alert = not is_white and has_orange

        return {"avg_color": avg_color, "orange_ratio": round(orange_ratio, 2)}, alert


    
    # elif roi_type == "text":
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     text = pytesseract.image_to_string(gray).strip()
    #     if name not in prev:
    #         return text, False
    #     changed = text != prev[name]
    #     return text, changed



    elif roi_type == "text":
        # Convert image to HSV for red color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red in HSV appears at both ends of hue spectrum (low and high)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Combine both red masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Apply red mask to original image to isolate red text
        red_text_img = cv2.bitwise_and(image, image, mask=red_mask)

        # Convert red-only image to grayscale for OCR
        gray = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()

        # Check for change from previous text
        if name not in prev:
            return text, bool(text)  # Treat appearance of text as change
        changed = text != prev[name] and text != ''
        return text, changed



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
                previous_values[name] = result

            time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user.")

if __name__ == "__main__":
    monitor()
