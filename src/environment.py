import os
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyautogui
import pygetwindow
from pygetwindow import Win32Window

from src.control import drop_coin, pay, unhold_all, walk_left, walk_right, do_nothing, run_right, run_left, un_pause
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
        self.start_game()
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

    @staticmethod
    def start_game():
        os.startfile(r"D:\Games\KingdomNewLands\Kingdom.exe")
        time.sleep(10)  # Allow time for the game to open
        logger.debug("Start game")

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

    def step(self, action: int):
        """
        Takes an action in the environment and returns the next state.
        """
        action_func = self.button_actions.get(str(action), do_nothing)
        action_func()
        time.sleep(0.1)
        next_state = self.get_state()
        return next_state

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial state.
        """
        # Close the game window
        window = self.window.get_window_with_exact_title(self.window.window_name)
        window.close()

        # Replace the save file
        save_path = os.path.expanduser("~\\AppData\\LocalLow\\noio\\Kingdom\\storage_v34_AUTO.dat")
        initial_state_path = os.path.expanduser("~\\AppData\\LocalLow\\noio\\Kingdom\\initial_state.dat")
        with open(initial_state_path, "rb") as src, open(save_path, "wb") as dst:
            dst.write(src.read())

        # Reopen the game
        self.start_game()

        # Prepare the window again
        self.window.prepare_window()
        initial_state = self.get_state()
        return initial_state

    @staticmethod
    def un_pause(sleep: Optional[int] = None):
        un_pause()
        if sleep:
            time.sleep(sleep)

# env = GameEnvironment()

# next_frame = env.step(1)
# print(next_frame)
# next_frame.save("step_1.png")
# time.sleep(3)
#
# frame = env.reset()
# print(frame)
# frame.save("initial_state.png")
#
# next_frame = env.step(1)
# print(next_frame)
# next_frame.save("step_1.png")
#
# time.sleep(3)
# next_frame = env.step(9)
# next_frame.save("step_9.png")
