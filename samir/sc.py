import pyautogui

# Screenshot of your primary display (right screen)
screenshot = pyautogui.screenshot(region=(0, 0, 1920, 1080))
screenshot.save("primary_monitor_check.png")
print("Screenshot saved as primary_monitor_check.png")
