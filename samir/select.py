# import cv2
# import pyautogui
# import json
# import numpy as np
# import tkinter as tk
# from tkinter import simpledialog
# import os

# output_json = "rois_config.json"

# # Load existing ROI data if file exists
# output_json = "rois_config.json"
# if os.path.exists(output_json):
#     with open(output_json, "r") as f:
#         roi_data = json.load(f)
#     print(f"[INFO] Loaded {len(roi_data)} existing ROIs.")
# else:
#     roi_data = []

# # Init tkinter
# root = tk.Tk()
# root.withdraw()

# # Globals for mouse drawing
# drawing = False
# ix, iy = -1, -1
# rx, ry, rw, rh = 0, 0, 0, 0
# roi_selected = False
# exit_program = False

# # Take screenshot once
# screenshot = pyautogui.screenshot(region=(0, 0, 1920, 1080))
# base_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
# clone_img = base_img.copy()

# def draw_roi(event, x, y, flags, param):
#     global ix, iy, rx, ry, rw, rh, drawing, roi_selected, clone_img

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         clone_img = base_img.copy()
#         cv2.rectangle(clone_img, (ix, iy), (x, y), (0, 255, 0), 2)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         rx, ry = min(ix, x), min(iy, y)
#         rw, rh = abs(x - ix), abs(y - iy)
#         roi_selected = True

# cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Select ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.setMouseCallback("Select ROI", draw_roi)

# print("[INFO] Click and drag to select ROI, then release.")
# print("       Prompt will appear to name and type.")
# print("       Press ESC to stop.\n")

# while not exit_program:
#     cv2.imshow("Select ROI", clone_img)
#     key = cv2.waitKey(1)

#     if roi_selected:
#         roi_selected = False
#         name = simpledialog.askstring("ROI Name", "Enter a name for this region:")
#         detect_type = None
#         while detect_type not in ("text", "color"):
#             detect_type = simpledialog.askstring("Detection Type", "Enter detection type: 'text' or 'color'").lower()

#         roi_data.append({
#             "name": name,
#             "region": [int(rx), int(ry), int(rw), int(rh)],
#             "type": detect_type
#         })

#         print(f"[ADDED] {name} ({detect_type}) at (x={rx}, y={ry}, w={rw}, h={rh})")
#         clone_img = base_img.copy()

#     if key == 27:  # ESC key
#         exit_program = True

# cv2.destroyAllWindows()

# # Save only once
# with open(output_json, "w") as f:
#     json.dump(roi_data, f, indent=4)

# print(f"\nSaved {len(roi_data)} ROIs to {output_json}")
