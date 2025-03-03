import random
import time
from enum import Enum
from time import sleep

import pydirectinput

from src.logging_config import logger

SIMPLE_CLICK_DURATION = 0.05


# TODO research more about actions in continuous-time envs
class ControlKey(Enum):
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    ESCAPE = "esc"
    SHIFT = "shift"


def unhold_all():
    logger.info(f"Performing unhold_all")
    pydirectinput.keyUp(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)


def walk_left():
    logger.info(f"Performing walk_left")

    pydirectinput.keyUp(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)

    # perform
    pydirectinput.keyDown(ControlKey.LEFT.value)


def walk_right():
    logger.info(f"Performing walk_right")

    pydirectinput.keyUp(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)

    pydirectinput.keyDown(ControlKey.RIGHT.value)


def run_left():
    logger.info(f"Performing run_left")

    pydirectinput.keyUp(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)

    pydirectinput.keyDown(ControlKey.LEFT.value)
    pydirectinput.keyDown(ControlKey.SHIFT.value)


def run_right():
    logger.info(f"Performing run_right")

    pydirectinput.keyUp(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)

    pydirectinput.keyDown(ControlKey.RIGHT.value)
    pydirectinput.keyDown(ControlKey.SHIFT.value)


def drop_coin():
    # drop can be done while walking/running

    logger.info(f"Performing drop_coin")
    pydirectinput.keyDown(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)


# in real game you could pay as you walk (not run), but for making it easier for model to pay
# the payment will be done without movement
# Also when running, paying works as dropping coins
def pay():
    logger.info(f"Performing pay")

    pydirectinput.keyUp(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.SHIFT.value)

    pydirectinput.keyDown(ControlKey.DOWN.value)


def do_nothing():
    logger.info("Performing do_nothing")
    unhold_all()
    pass


def do_random_action():
    actions_list = [
        walk_left,
        walk_right,
        run_left,
        run_right,
        pay,
        drop_coin,
        unhold_all,
        do_nothing,
    ]  # Action functions

    action = random.choice(actions_list)
    action()


def un_pause():
    pydirectinput.keyDown(ControlKey.ESCAPE.value)
    pydirectinput.keyUp(ControlKey.ESCAPE.value)
