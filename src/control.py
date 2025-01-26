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

def unhold_all():
    pydirectinput.keyUp(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)

def walk_left():
    logger.debug(f"Performing walk_left")

    unhold_all()

    # perform
    pydirectinput.keyDown(ControlKey.LEFT.value)

def walk_right():
    logger.debug(f"Performing walk_right")

    unhold_all()

    pydirectinput.keyDown(ControlKey.RIGHT.value)

def run_left():
    logger.debug(f"Performing run_left")

    unhold_all()

    pydirectinput.keyDown(ControlKey.LEFT.value)
    pydirectinput.keyUp(ControlKey.LEFT.value)
    sleep(SIMPLE_CLICK_DURATION)

    pydirectinput.keyDown(ControlKey.LEFT.value)

def run_right():
    logger.debug(f"Performing run_right")

    unhold_all()

    pydirectinput.keyDown(ControlKey.RIGHT.value)
    pydirectinput.keyUp(ControlKey.RIGHT.value)
    sleep(SIMPLE_CLICK_DURATION)

    pydirectinput.keyDown(ControlKey.RIGHT.value)


def drop_coin():
    # drop can be done while walking/running

    logger.debug(f"Performing drop_coin")
    pydirectinput.keyDown(ControlKey.DOWN.value)
    pydirectinput.keyUp(ControlKey.DOWN.value)


def pay():
    logger.debug(f"Performing pay")

    unhold_all()

    pydirectinput.keyDown(ControlKey.DOWN.value)


def do_nothing():
    logger.debug("Performing do_nothing")
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
        do_nothing
    ]  # Action functions

    action = random.choice(actions_list)
    action()

def un_pause():
    pydirectinput.keyDown(ControlKey.ESCAPE.value)
    pydirectinput.keyUp(ControlKey.ESCAPE.value)
