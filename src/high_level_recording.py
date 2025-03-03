import os
import time
from typing import List

import keyboard
from PIL import Image

from src.environment import GameEnvironment
from src.logging_config import logger


def capture_continuous(environment, interval=0.07):
    screenshots: List[Image.Image] = []

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"../expert_observations/{int(interval * 1000)}_{run_id}"
    os.makedirs(os.path.join(output_dir, "screenshots"))

    logger.info("Press 'Space' to start recording screenshots.")
    keyboard.wait("space")  # Wait for Space key to be pressed
    logger.info("Recording started. Press 'Esc' to stop.")

    while not keyboard.is_pressed("esc"):  # Stop when Esc is pressed
        try:
            screenshots.append(environment.get_state())
            time.sleep(interval)
        except Exception as e:
            print(f"Error: {e}")
            break

    logger.info(f"Saving {len(screenshots)} screenshots")

    for i, screenshot in enumerate(screenshots):
        screenshot.save(os.path.join(output_dir, "screenshots", f"{i}.png"))
    logger.info("Saved")


env = GameEnvironment()
while True:
    capture_continuous(env, interval=0.07)
    env.reset()
env.close_game()
