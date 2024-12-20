# Example usage
from src.control import test_movements
from src.window_resize import bring_window_to_foreground, set_window_size

game_window_title = "Kingdom"  # Replace with your game window title
set_window_size(game_window_title, x=100, y=100, width=640, height=360)
bring_window_to_foreground(game_window_title)
test_movements()
