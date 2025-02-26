import json
import os
import time
from datetime import datetime
from typing import List

import keyboard
from PIL import Image

from src.control import do_nothing, unhold_all
from src.environment import GameEnvironment
from src.logging_config import logger


class Run:
    max_pause_sec = 1
    run_id: str
    output_dir: str
    action_log: dict
    environment: GameEnvironment

    def __init__(self, environment: GameEnvironment):
        self.environment = environment

        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"../expert_observations/{int(0.07 * 1000)}_{self.run_id}"  # TODO refactor
        os.makedirs(os.path.join(self.output_dir, "screenshots"))

        self.action_log = {}

    def safe_action_log(self):
        log_path = os.path.join(self.output_dir, "action_log.json")
        with open(log_path, "w") as f:
            json.dump(self.action_log, f)
        logger.debug("Action sequence saved")

    def save_frame(self, state_id: int, frame: Image.Image):
        logger.debug("Saving screenshot")

        file_path = os.path.join(self.output_dir, "screenshots", f"{state_id}.png")
        logger.debug(file_path)
        frame.save(file_path)

    def capture_by_action(self):
        """
        Control the character and capture screenshots simultaneously.
        """

        # just in case reset
        self.environment.window.prepare_window()

        logger.info("Start capture")
        saving_timestamp = datetime.now()

        try:
            while True:
                # чтобы мой мозг не нагружать буду играть не по реале, а прокси-действиями
                # потому что ч хз нужно ли логировтаь стейт, когда я отпускаю кнопку?
                # Check for button presses
                pressed_key = keyboard.read_key()
                logger.debug(f"Pressed {pressed_key}")

                # Check for "Q" key press to save log and exit
                if pressed_key == "q":
                    unhold_all()
                    self.safe_action_log()
                    break

                # Make screenshot if action from action space is done
                action = self.environment.button_actions.get(pressed_key)
                if action is not None:
                    saving_timestamp = self.log_and_perform_action(action)
                # TODO fix. this part is never executed since keyboard.read_key() is blocking
                elif (datetime.now() - saving_timestamp).seconds > self.max_pause_sec:
                    action = do_nothing
                    saving_timestamp = self.log_and_perform_action(action)
                else:
                    time.sleep(0.1)

        except Exception as e:
            self.safe_action_log()
            raise e


    def log_and_perform_action(self, action: callable) -> datetime:
        saving_timestamp = datetime.now()
        state_id = self.environment.get_state_id(saving_timestamp)

        # Log state just before action
        state_image = self.environment.get_state()
        self.save_frame(state_id, state_image)

        # Log action
        self.action_log[state_id] = action.__name__
        # Perform an action
        action()
        return saving_timestamp

    def capture_continuous(self, interval=0.07):
        screenshots: List[Image.Image] = []


        logger.info("Press 'Space' to start recording screenshots.")
        keyboard.wait('space')  # Wait for Space key to be pressed
        logger.info("Recording started. Press 'Esc' to stop.")

        while not keyboard.is_pressed('esc'):  # Stop when Esc is pressed
            try:
                screenshots.append(self.environment.get_state())
                time.sleep(interval)
            except Exception as e:
                print(f"Error: {e}")
                break

        logger.info(f"Saving {len(screenshots)} screenshots")

        for i, screenshot in enumerate(screenshots):
            screenshot.save(os.path.join(self.output_dir, "screenshots", f"{i}.png"))
        logger.info("Saved")


env = GameEnvironment()
run = Run(env)
run.capture_continuous()
env.close_game()