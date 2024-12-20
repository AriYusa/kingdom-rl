import os
import time

import PIL
import pyautogui
import pygetwindow

from src.control import do_random_action
from src.logging_config import logger

def prepare(game_window_title):
    # Obtain screen size
    x, y = pyautogui.size()
    print(f"Screen resolution: width={x}, height={y}")

    windows = pygetwindow.getWindowsWithTitle(game_window_title)
    if not windows:
        raise Exception(f"Window with title '{game_window_title}' not found.")

    # Use the first matching window
    window = windows[1]

    # Resize and position the window
    window.resizeTo(x // 2, y // 2)  # Half of the screen
    window.moveTo(0, 0)  # Move to top-left corner
    window.activate()
    time.sleep(1)  # Allow time for the window to adjust

    # Obtain the window's geometry
    win_left, win_top, win_width, win_height = window.left, window.top, window.width, window.height
    logger.debug(f"Window geometry: Left={win_left}, Top={win_top}, Width={win_width}, Height={win_height}")

    border_width = 10
    title_bar_height = 50

    inner_left = win_left + border_width
    inner_top = win_top + title_bar_height
    inner_width = win_width - 2 * border_width
    inner_height = win_height - title_bar_height - border_width

    logger.debug(f"Inner content area: Left={inner_left}, Top={inner_top}, Width={inner_width}, Height={inner_height}")
    return inner_left, inner_top, inner_width, inner_height

def save_screenshot(output_dir: str, frame: PIL.Image.Image):
    logger.debug("Saving screenshot")
    timestamp = time.strftime("%H%M%S")
    file_path = os.path.join(output_dir, f"{timestamp}.png")
    logger.debug(file_path)
    frame.save(file_path)


def control_and_capture(window_name):
    """
    Control the character and capture screenshots simultaneously.
    """
    inner_left, inner_top, inner_width, inner_height = prepare(window_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screen_output_dir = f"../logs/screenshots/{timestamp}"
    os.makedirs(screen_output_dir)

    # Set up frame capture and action loop
    logger.info("Start capture")
    while True:
        # Capture the screenshot
        screenshot = pyautogui.screenshot()

        # Crop to the inner content area
        inner_bounds = (inner_left, inner_top, inner_left + inner_width, inner_top + inner_height)
        cropped_image = screenshot.crop(inner_bounds)

        # Save the cropped screenshot
        save_screenshot(screen_output_dir, cropped_image)

        # Perform an action (example: move right)
        do_random_action()
        # Wait for a short duration to simulate frame rate
        time.sleep(2)



# Example usage

game_window_title = "Kingdom"
control_and_capture(game_window_title)
