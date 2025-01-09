import json
import os
import time
from datetime import datetime

from PIL import Image
import pyautogui
import keyboard
import pygetwindow
from pygetwindow import Win32Window

from src.control import drop_coin, pay, unhold_all, walk_left, walk_right, do_nothing, run_right, run_left
from src.logging_config import logger


class Window:
    inner_height: int
    inner_width: int
    inner_top: int
    inner_left: int

    def __init__(self, window_name):

        self.window_name = window_name

    @staticmethod
    def get_window_with_exact_title(window_name: str) -> Win32Window:
        windows = pygetwindow.getWindowsWithTitle(window_name)
        if not windows:
            raise Exception(f"Window with title '{window_name}' not found.")

        for w in windows:
            if w.title == window_name:
                return w

        raise Exception(f"Window with title '{window_name}' not found.")


    def prepare_window(self):
        # Obtain screen size
        x, y = pyautogui.size()
        print(f"Screen resolution: width={x}, height={y}")

        window = self.get_window_with_exact_title(self.window_name)
        if not window:
            raise Exception(f"Window with title '{self.window_name}' not found.")

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

        self.inner_left = win_left + border_width
        self.inner_top = win_top + title_bar_height
        self.inner_width = win_width - 2 * border_width
        self.inner_height = win_height - title_bar_height - border_width

        logger.debug(
            f"Inner content area: Left={self.inner_left},"
            f" Top={self.inner_top}, Width={self.inner_width}, Height={self.inner_height}")


class GameEnvironment:
    def __init__(self, window_name:str = "Kingdom"):
        self.window = Window(window_name)
        self.window.prepare_window()

    button_actions = {
        "1": run_left,
        "2": walk_left,
        "3": walk_right,
        "4": run_right,
        "8": pay,
        "9": drop_coin,
        "0": unhold_all,
        "-": do_nothing,
    }

    def postprocess_image(self, screenshot: Image.Image):
        inner_bounds = (
            self.window.inner_left,
            self.window.inner_top,
            self.window.inner_left + self.window.inner_width,
            self.window.inner_top + self.window.inner_height)

        cropped_image = screenshot.crop(inner_bounds)
        image = cropped_image.resize((640, 360), Image.Resampling.LANCZOS)
        return image

    def get_state(self) -> Image.Image:
        # Capture the screenshot before action
        screenshot = pyautogui.screenshot()

        # Crop, resize
        image = self.postprocess_image(screenshot)
        return image

    @staticmethod
    def get_state_id(timestamp: datetime) -> int:
        return int(timestamp.strftime('%H%M%S%f')[:-2])


class Run:
    max_pause_sec = 1
    run_id: str
    output_dir: str
    action_log: dict
    environment: GameEnvironment

    def __init__(self, environment: GameEnvironment):
        self.environment = environment

        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"../logs/{self.run_id}"
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

    def capture(self):
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

env = GameEnvironment()
run = Run(env)
run.capture()