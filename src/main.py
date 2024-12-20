import time

import cv2
import numpy as np
import win32con
import win32gui
import win32ui

# from src.logging_config import logger
# from src.window_resize import (bring_window_to_foreground,
#                                get_client_window_coords,
#                                set_client_window_size)
#
#
# def save_screenshot(output_dir: str, frame):
#     logger.debug("Saving screenshot")
#     timestamp = time.strftime("%H%M%S")
#     file_path = os.path.join(output_dir, f"{timestamp}.png")
#     logger.debug(file_path)
#     cv2.imwrite(file_path, frame)
#
#
# def control_and_capture(window_name):
#     """
#     Control the character and capture screenshots simultaneously.
#     """
#     x, y, width, height = get_client_window_coords(window_name)
#
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     screen_output_dir = f"../logs/screenshots/{timestamp}"
#     os.makedirs(screen_output_dir)
#
#     # Set up frame capture and action loop
#     with mss.mss() as sct:
#         monitor = {
#             "top": 175,
#             "left": 160,
#             "width": width,
#             "height": height,
#         }
#         logger.debug(monitor)
#
#         logger.info("Start capture")
#         while True:
#             # Capture the game screen
#             screenshot = sct.grab(monitor)
#             frame = np.array(screenshot)
#             save_screenshot(screen_output_dir, frame)
#
#             # Perform an action (example: move right)
#             # do_random_action()
#             # Wait for a short duration to simulate frame rate
#             time.sleep(2)
#
#             # Break the loop if 'q' is pressed (for testing/debugging)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 logger.info("Stop capture")
#                 break


# # Example usage
#
# game_window_title = "Kingdom"
# bring_window_to_foreground(game_window_title)
# set_client_window_size(game_window_title, x=100, y=100, width=640, height=310)
#
# control_and_capture(game_window_title)


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name=None):
        # find the handle for the window we want to capture.
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # set the offset so we can translate screenshot images into actual screen positions
        self.offset_x = window_rect[0]
        self.offset_y = window_rect[1]

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)

        # Capture the entire window area (no cropping)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (0, 0), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')  # Fix: np.fromstring() is deprecated
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, as it's often not necessary for most image processing tasks
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)


# WindowCapture.list_window_names()
wincap = WindowCapture('Kingdom')  # Kingdom
loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    cv2.imshow('Computer Vision', screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time.time() - loop_time)))
    loop_time = time.time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break