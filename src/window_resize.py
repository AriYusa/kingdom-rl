import win32api
import win32con
import win32gui

from src.logging_config import logger


def set_client_window_size(window_title, x, y, width, height):
    # Get the current window rectangle (left, top, right, bottom)
    hwnd = win32gui.FindWindow(None, window_title)
    window_rect = win32gui.GetWindowRect(hwnd)
    # Get the current client area (left, top, right, bottom)
    client_rect = win32gui.GetClientRect(hwnd)

    # Calculate the window's borders (titlebar and window borders)
    border_width = (window_rect[2] - window_rect[0]) - (client_rect[2] - client_rect[0])
    border_height = (window_rect[3] - window_rect[1]) - (
        client_rect[3] - client_rect[1]
    )

    # Calculate the full window size including borders/titlebar
    new_width = width + border_width
    new_height = height + border_height

    # Set the window position and size (x, y, width, height)
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOP, x, y, new_width, new_height, win32con.SWP_NOZORDER
    )

    logger.debug(
        f"Window size set with window_coords - {x, y, new_width, new_height}"
        f" and client_window_coords - {get_client_window_coords(window_title)}"
    )


def bring_window_to_foreground(window_title):
    # Find the window handle (HWND) by its title
    hwnd = win32gui.FindWindow(None, window_title)

    if hwnd:
        # Check if the window is minimized
        is_minimized = win32gui.IsIconic(hwnd)

        # Restore the window if it is minimized
        if is_minimized:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            logger.debug(f"Window '{window_title}' was minimized and is now restored.")

        # Bring the window to the foreground
        win32gui.SetForegroundWindow(hwnd)

        logger.debug(f"Window '{window_title}' is now in the foreground.")
    else:
        logger.error(f"Window with title '{window_title}' not found.")


def get_client_window_coords(window_title):
    """

    :param window_title:
    :return:
    """
    hwnd = win32gui.FindWindow(None, window_title)

    # Get client rectangle (left, top, right, bottom)
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    wight = right - left
    height = bottom - top

    # Convert client top-left corner to screen coordinates
    absolute_y, cli_absolute_x = win32gui.ClientToScreen(hwnd, (left, top))

    return cli_absolute_x, absolute_y, wight, height


def get_actual_client_rect(window_title):
    """

    :param window_title:
    :return:
    """
    hwnd = win32gui.FindWindow(None, window_title)

    # Get client rectangle in device-independent units
    client_rect = win32gui.GetClientRect(hwnd)

    # Get the device context for the window
    hdc = win32gui.GetDC(hwnd)

    # Get the DPI (dots per inch) for the device context
    dpi = win32api.GetDeviceCaps(hdc, win32con.LOGPIXELSX)

    # Release the device context
    win32gui.ReleaseDC(hwnd, hdc)

    # Calculate scaling factor based on DPI (default is 96 DPI)
    scale_factor = dpi / 96.0

    # Convert client rect dimensions to actual pixels by scaling
    actual_width = int(client_rect[2] * scale_factor)
    actual_height = int(client_rect[3] * scale_factor)

    return actual_width, actual_height


# Example usage
hwnd = win32gui.FindWindow(None, "Window Title")
actual_width, actual_height = get_actual_client_rect(hwnd)
print(f"Actual client area size in pixels: {actual_width}x{actual_height}")
