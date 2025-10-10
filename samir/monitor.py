import cv2
import pytesseract
import json
import pyautogui
import numpy as np
import time
import threading
import platform
from datetime import datetime
from tkinter import Tk, Toplevel, Frame, Label, Text, Button

# === Try to import Rich for pretty console ===
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None

# === Config ===
SCREEN_REGION = (0, 0, 1920, 1080)
ROI_FILE = "rois_config.json"
RULES_FILE = "malfunctions_rules.json"
DELAY = 1.0
MAX_ROWS = 25

# === Load Configuration Files ===
with open(ROI_FILE, "r") as f:
    ROIS = json.load(f)
print(f"[INFO] Loaded {len(ROIS)} existing ROIs.")

with open(RULES_FILE, "r") as f:
    raw_rules = json.load(f)
MALFUNCTION_RULES = {k: set(v) for k, v in raw_rules.items()}

# === State Variables ===
previous_values = {}
active_alerts = set()
shown_malfunctions = set()
last_logged = {}

# === Tkinter Setup ===
root = Tk()
root.withdraw()


# --- Popup Helper ---------------------------------------------------------
def show_malfunction_popup(title_text: str, signals=None):
    """Pretty non-blocking popup for malfunction alerts."""
    signals = signals or []

    def _popup():
        win = Toplevel(root)
        win.title("⚠️ Malfunction Detected")
        win.attributes("-topmost", True)
        win.configure(bg="#2b2b2b")
        win.resizable(False, False)

        pad = 14
        border = Frame(win, bg="#ff4d4f", bd=0)
        border.pack(fill="both", expand=True, padx=2, pady=2)
        container = Frame(border, bg="#2b2b2b")
        container.pack(fill="both", expand=True, padx=pad, pady=pad)

        title = Label(
            container,
            text=f"⚠️  {title_text}",
            font=("Segoe UI", 18, "bold"),
            fg="#ffffff",
            bg="#2b2b2b",
        )
        title.pack(anchor="w", pady=(0, 8))

        desc = Label(
            container,
            text="The following signals are active:",
            font=("Segoe UI", 11),
            fg="#d0d0d0",
            bg="#2b2b2b",
        )
        desc.pack(anchor="w")

        if signals:
            listbox = Text(
                container,
                height=min(10, max(3, len(signals))),
                width=48,
                font=("Consolas", 11),
                fg="#f0f0f0",
                bg="#1e1e1e",
                bd=0,
            )
            listbox.pack(fill="both", expand=True, pady=(6, 10))
            listbox.insert("1.0", "\n".join(f"• {s}" for s in signals))
            listbox.configure(state="disabled")

        btns = Frame(container, bg="#2b2b2b")
        btns.pack(fill="x")

        def close():
            win.destroy()

        def copy_to_clip():
            try:
                root.clipboard_clear()
                root.clipboard_append(
                    f"Malfunction: {title_text}\nSignals:\n"
                    + "\n".join(f"- {s}" for s in signals)
                )
            except Exception:
                pass

        Button(btns, text="Copy details", command=copy_to_clip).pack(side="left")
        Button(btns, text="Acknowledge (Enter)", command=close, default="active").pack(
            side="right"
        )

        win.bind("<Return>", lambda e: close())
        win.bind("<Escape>", lambda e: close())

        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{int((sw - w) / 2)}+{int((sh - h) / 3)}")

        try:
            if platform.system() == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONHAND)
            else:
                print("\a", end="")
        except Exception:
            pass

        win.lift()
        win.focus_force()

    threading.Thread(target=_popup, daemon=True).start()


# --- Detection Functions ---------------------------------------------------
def average_color(img):
    return tuple(map(int, img.mean(axis=0).mean(axis=0)))


def detect_change(name, roi_type, image, prev):
    if roi_type == "color":
        avg_color = average_color(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
        orange_ratio = cv2.countNonZero(mask_orange) / (
            image.shape[0] * image.shape[1]
        )
        is_whiteish = all(c > 230 for c in avg_color)
        alert = (not is_whiteish) and orange_ratio > 0.05
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
        changed = text != prev[name] and text != ""
        return text, changed


# --- Table Display Helpers -------------------------------------------------
def make_table():
    tbl = Table(title="Live Signal Changes", box=box.SIMPLE_HEAVY)
    tbl.add_column("#", justify="right")
    tbl.add_column("Time", no_wrap=True)
    tbl.add_column("Alert", style="bold")
    tbl.add_column("Avg Color (B,G,R)", justify="center")
    tbl.add_column("Orange Ratio", justify="right")
    tbl.add_column("Note")
    return tbl


def add_row(tbl, idx, when, name, result):
    if isinstance(result, dict):
        avg = result.get("avg_color", "")
        ratio = result.get("orange_ratio", "")
        note = "Color → orange-dominant"
    else:
        avg = "-"
        ratio = "-"
        note = f"Text → '{result}'"
    tbl.add_row(str(idx), when, name, str(avg), str(ratio), note)


# --- Malfunction Detection -------------------------------------------------
def pretty_malfunction(mname):
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"⚠️  Malfunction Detected: [bold]{mname}[/bold]", border_style="red"))
    else:
        print(f"[⚠️ MALFUNCTION] {mname}")


def check_for_malfunctions():
    for malfunction, required_signals in MALFUNCTION_RULES.items():
        if required_signals.issubset(active_alerts) and malfunction not in shown_malfunctions:
            pretty_malfunction(malfunction)
            signals_list = sorted(list(required_signals))
            show_malfunction_popup(malfunction, signals=signals_list)
            shown_malfunctions.add(malfunction)


# --- Main Monitoring Loop --------------------------------------------------
def monitor():
    header = "📡 Monitoring started... (Ctrl+C to stop)"
    if RICH_AVAILABLE:
        console.rule(header)
    else:
        print(header)

    row_counter = 0
    rows_buffer = []
    live = Live(refresh_per_second=8, console=console) if RICH_AVAILABLE else None
    if RICH_AVAILABLE:
        tbl = make_table()
        live.start()

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
                previous_values[name] = result

                key = (name, str(result))
                if changed and last_logged.get(name) != str(result):
                    last_logged[name] = str(result)
                    active_alerts.add(name)

                    row_counter += 1
                    now = datetime.now().strftime("%H:%M:%S")
                    rows_buffer.append((row_counter, now, name, result))
                    if len(rows_buffer) > MAX_ROWS:
                        rows_buffer = rows_buffer[-MAX_ROWS:]

                    if RICH_AVAILABLE:
                        tbl = make_table()
                        for idx, when, n, res in rows_buffer:
                            add_row(tbl, idx, when, n, res)
                        live.update(tbl)
                    else:
                        print(
                            f"[{now}] CHANGED | {name:<20} | avg={result.get('avg_color')} | orange={result.get('orange_ratio')}"
                        )

            check_for_malfunctions()
            time.sleep(DELAY)

    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.rule("🛑 Monitoring stopped by user.")
            live.stop()
        else:
            print("\n🛑 Monitoring stopped by user.")


# --- Run -------------------------------------------------------------------
if __name__ == "__main__":
    monitor()
