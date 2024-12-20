from enum import Enum

import pydirectinput
import time
import win32gui


def set_window_size(window_title, x, y, width, height):
    """
    Sets the size and position of a window based on its title.

    Parameters:
    window_title (str): The title of the target window.
    x (int): The x-coordinate of the top-left corner.
    y (int): The y-coordinate of the top-left corner.
    width (int): The width of the window.
    height (int): The height of the window.
    """
    # Find the window handle (HWND) by its title
    hwnd = win32gui.FindWindow(None, window_title)

    if hwnd:
        # Use MoveWindow to set the window's position and size
        win32gui.MoveWindow(hwnd, x, y, width, height, True)
        print(f"Set window '{window_title}' to position ({x}, {y}) with size ({width}x{height}).")
    else:
        print(f"Window with title '{window_title}' not found.")


def bring_window_to_foreground(window_title):
    # Find the window handle (HWND) by its title
    hwnd = win32gui.FindWindow(None, window_title)

    if hwnd:
        # Bring the window to the foreground
        win32gui.SetForegroundWindow(hwnd)

        print(f"Window '{window_title}' is now in the foreground.")
    else:
        print(f"Window with title '{window_title}' not found.")

SIMPLE_CLICK_DURATION = 0.1


def use_key(key, duration=SIMPLE_CLICK_DURATION):
    pydirectinput.keyDown(key)  # Hold the key
    time.sleep(duration)  # Hold the key for the specified duration
    pydirectinput.keyUp(key)  # Release the key after holding


def hold_key(key, duration=SIMPLE_CLICK_DURATION, is_double_press=False):
    """

    :param key:
    :param duration:
    :param is_double_press:
    :return:
    """

    if is_double_press:
        use_key(key)
        time.sleep(SIMPLE_CLICK_DURATION) # To imitate user action

    use_key(key, duration)


class MoveDirection(Enum):
   LEFT = 'left'
   RIGTH = 'right'
   DOWN = 'down'


def walk(direction: MoveDirection, duration):
    use_key(direction.value, duration)

def run(direction: MoveDirection, duration):
    hold_key(direction.value, duration, is_double_press=True)

def drop_coins(amount:int):
    for i in range(amount):
        use_key(MoveDirection.DOWN.value)
        time.sleep(SIMPLE_CLICK_DURATION) # To imitate user action

def pay(amount: int):
    hold_key(MoveDirection.DOWN.value, duration=0.3 * amount)

def do_nothing(duration = 1):
    time.sleep(duration)

def test_movements():
    walk(MoveDirection.LEFT, duration=1)
    run(MoveDirection.RIGTH, duration=0.5)

    # do_nothing()
    # drop_coins(0)
    # do_nothing()

    pay(5)




# Example usage
game_window_title = "Kingdom"  # Replace with your game window title
set_window_size(game_window_title, x=100, y=100, width=640, height=360)
bring_window_to_foreground(game_window_title)
test_movements()

