import pygetwindow as gw
from PIL import ImageGrab
import win32gui
import win32con
import ctypes

# Enable DPI awareness for accurate pixel coordinates
ctypes.windll.shcore.SetProcessDpiAwareness(2)  # DPI_AWARENESS_PER_MONITOR_AWARE

# Define the exact title of the window
exact_title = "Kingdom"

# Get all windows with the substring and filter for the exact title
windows = gw.getWindowsWithTitle(exact_title)
matching_window = None
for window in windows:
    if window.title == exact_title:
        matching_window = window
        break

if matching_window:
    # Get window handle
    hwnd = matching_window._hWnd

    # Bring the window to the foreground
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore if minimized
    win32gui.SetForegroundWindow(hwnd)  # Bring to foreground

    # Adjust for DPI scaling
    scaling_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0  # 0: Primary monitor
    print(f"Scaling factor: {scaling_factor}")

    # Get window coordinates and dimensions
    x, y = int(matching_window.left * scaling_factor), int(matching_window.top * scaling_factor)
    width, height = int(matching_window.width * scaling_factor), int(matching_window.height * scaling_factor)

    # Take a screenshot of the specified region
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))  # bbox=(left, top, right, bottom)

    # Save the screenshot to a file
    screenshot.save("window_screenshot_corrected.png")
    print(f"Screenshot saved as 'window_screenshot_corrected.png'")
else:
    print(f"No window found with the exact title '{exact_title}'.")
