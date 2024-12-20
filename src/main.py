import os
import time

import cv2
import mss
import numpy as np

from src.logging_config import logger
from src.window_resize import (bring_window_to_foreground,
                               get_client_window_coords,
                               set_client_window_size)


def save_screenshot(output_dir: str, frame):
    logger.debug("Saving screenshot")
    timestamp = time.strftime("%H%M%S")
    file_path = os.path.join(output_dir, f"{timestamp}.png")
    logger.debug(file_path)
    cv2.imwrite(file_path, frame)


def control_and_capture(window_name):
    """
    Control the character and capture screenshots simultaneously.
    """
    x, y, width, height = get_client_window_coords(window_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screen_output_dir = f"../logs/screenshots/{timestamp}"
    os.makedirs(screen_output_dir)

    # Set up frame capture and action loop
    with mss.mss() as sct:
        monitor = {
            "top": 175,
            "left": 160,
            "width": width,
            "height": height,
        }
        logger.debug(monitor)

        logger.info("Start capture")
        while True:
            # Capture the game screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            save_screenshot(screen_output_dir, frame)

            # Perform an action (example: move right)
            # do_random_action()
            # Wait for a short duration to simulate frame rate
            time.sleep(2)

            # Break the loop if 'q' is pressed (for testing/debugging)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Stop capture")
                break


# Example usage

game_window_title = "Kingdom"
bring_window_to_foreground(game_window_title)
set_client_window_size(game_window_title, x=100, y=100, width=640, height=310)

control_and_capture(game_window_title)
