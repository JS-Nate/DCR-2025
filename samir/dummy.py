# demo_popup_native.py
import platform

MALFUNCTION = "Steam Header Break"
SIGNALS = [
    "Reactor Trip",
    "Turbine Trip",
    "MSS Isolation",
    "FWS Isolation",
    "PDHR Actuation",
    "Gen Breaker Open",
    "Low RCS Pressure",
    "PDHR -> ON",
    "RCS -> Trip",
    "Turbine -> Decelerating",
    "MSS -> Isolated",
    "FWS -> Isolated",
]

def format_message(malfunction: str, signals: list[str]) -> str:
    if not signals:
        return f"{malfunction} detected."
    bullet = "\n".join(f" • {s}" for s in signals)
    return f"{malfunction} detected.\n\nActive signals:\n{bullet}"

def show_native_popup(title: str, message: str) -> None:
    system = platform.system()

    if system == "Windows":
        # True native MessageBox (compact, authentic)
        import ctypes
        from ctypes import wintypes

        MB_OK = 0x00000000
        MB_ICONWARNING = 0x00000030
        MB_TOPMOST = 0x00040000

        flags = MB_OK | MB_ICONWARNING | MB_TOPMOST

        # Set the console window as owner (optional; helps keep box on top of this app)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        GetConsoleWindow = kernel32.GetConsoleWindow
        GetConsoleWindow.restype = wintypes.HWND

        MessageBoxW = user32.MessageBoxW
        MessageBoxW.argtypes = (wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT)
        MessageBoxW.restype = ctypes.c_int

        hwnd_owner = GetConsoleWindow()
        MessageBoxW(hwnd_owner, message, title, flags)
        return

    # Fallback: small, default Tk message box (macOS/Linux)
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()

if __name__ == "__main__":
    title = "⚠️ Malfunction Detected"
    msg = format_message(MALFUNCTION, SIGNALS)
    show_native_popup(title, msg)
    print("Popup shown.")  # optional terminal confirmation
