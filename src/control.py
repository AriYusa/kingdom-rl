import random
import time
from enum import Enum

import pydirectinput

from src.logging_config import logger

SIMPLE_CLICK_DURATION = 0.1


class ControlKey(Enum):
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"


def tap(key: ControlKey):
    """
    For example: to drop money
    """
    logger.debug(f"Performing tap on {key.name}")
    pydirectinput.keyDown(key.value)
    time.sleep(SIMPLE_CLICK_DURATION)
    pydirectinput.keyUp(key.value)


def hold(key: ControlKey):
    """
    For example: to walk, to pay money
    """
    logger.debug(f"Performing hold on {key.name}")
    pydirectinput.keyDown(key.value)


def unhold(key: ControlKey):
    logger.debug(f"Performing unhold on {key.name}")
    pydirectinput.keyUp(key.value)


def tap_hold(key: ControlKey):
    """
    For example: to run
    """
    logger.debug(f"Performing tap_hold on {key.name}")
    tap(key)
    hold(key)


def do_nothing():
    logger.debug("Performing do_nothing")
    pass


actions = [hold, unhold, tap, tap_hold, do_nothing]


def test_actions():
    # Move left
    logger.debug("Move left")
    hold(ControlKey.LEFT)
    time.sleep(2)
    unhold(ControlKey.LEFT)

    # Run right
    logger.debug("Run right")
    tap_hold(ControlKey.RIGHT)
    time.sleep(2)
    unhold(ControlKey.RIGHT)

    # Drop money
    logger.debug("Drop money")
    time.sleep(2)
    tap(ControlKey.DOWN)
    time.sleep(2)

    # Walk left and drop money
    logger.debug("Drop money")
    hold(ControlKey.LEFT)
    time.sleep(0.1)
    tap(ControlKey.DOWN)
    time.sleep(0.5)
    tap(ControlKey.DOWN)
    time.sleep(1)
    unhold(ControlKey.LEFT)

    # Pay
    logger.debug("Pay money")
    hold(ControlKey.DOWN)
    time.sleep(0.3 * 5)
    unhold(ControlKey.DOWN)


def do_random_action():
    keys = [ControlKey.LEFT, ControlKey.RIGHT, ControlKey.DOWN]  # Keys to act on
    actions_list = [tap, hold, unhold, tap_hold, do_nothing]  # Action functions

    while True:
        # Randomly choose an action and a key (if applicable)
        action = random.choice(actions_list)

        if action in [tap, hold, unhold, tap_hold]:
            key = random.choice(keys)  # Select a random key
            action(key)  # Call the action with the key
        else:
            action()  # Call actions like do_nothing with no arguments
