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
        print(
            f"Set window '{window_title}' to position ({x}, {y})"
            f" with size ({width}x{height})."
        )
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
