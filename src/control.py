import time
from enum import Enum

import pydirectinput

SIMPLE_CLICK_DURATION = 0.1


class ControlKey(Enum):
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"


def tap(key: ControlKey):
    """
    For example: to drop money
    """
    pydirectinput.keyDown(key.value)
    time.sleep(SIMPLE_CLICK_DURATION)
    pydirectinput.keyUp(key.value)


def hold(key: ControlKey):
    """
    For example: to walk, to pay money
    """
    pydirectinput.keyDown(key.value)


def unhold(key: ControlKey):
    pydirectinput.keyUp(key.value)


def tap_hold(key: ControlKey):
    """
    For example: to run
    """
    tap(key)
    hold(key)


def do_nothing():
    pass


def test_movements():
    # Move left
    print("Move left")
    hold(ControlKey.LEFT)
    time.sleep(2)
    unhold(ControlKey.LEFT)

    # Run right
    print("Run right")
    tap_hold(ControlKey.RIGHT)
    time.sleep(2)
    unhold(ControlKey.RIGHT)

    # Drop money
    print("Drop money")
    time.sleep(2)
    tap(ControlKey.DOWN)
    time.sleep(2)

    # Walk left and drop money
    print("Drop money")
    hold(ControlKey.LEFT)
    time.sleep(0.1)
    tap(ControlKey.DOWN)
    time.sleep(0.5)
    tap(ControlKey.DOWN)
    time.sleep(1)
    unhold(ControlKey.LEFT)

    # Pay
    print("Pay money")
    hold(ControlKey.DOWN)
    time.sleep(0.3 * 5)
    unhold(ControlKey.DOWN)
