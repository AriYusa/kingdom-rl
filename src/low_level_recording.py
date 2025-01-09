from pynput import keyboard
from datetime import datetime
import csv

# List to store keyboard events
keyboard_events = []


def on_press(key):
    try:
        # Get current timestamp
        timestamp = datetime.now()
        # Record the key press event
        event = {
            'timestamp': timestamp,
            'event_type': 'press',
            'key': str(key)
        }
        keyboard_events.append(event)
        print(f"Key pressed: {key} at {timestamp}")
    except AttributeError:
        print(f"Special key pressed: {key}")


def on_release(key):
    try:
        # Get current timestamp
        timestamp = datetime.now()
        # Record the key release event
        event = {
            'timestamp': timestamp,
            'event_type': 'release',
            'key': str(key)
        }
        keyboard_events.append(event)
        print(f"Key released: {key} at {timestamp}")

        # Stop listener if 'esc' key is pressed
        if key == keyboard.Key.esc:
            # Save events to CSV file
            save_events_to_csv()
            return False
    except AttributeError:
        print(f"Special key released: {key}")


def save_events_to_csv():
    """Save recorded events to a CSV file"""
    filename = f"keyboard_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'event_type', 'key']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for event in keyboard_events:
            writer.writerow(event)

    print(f"Events saved to {filename}")


def main():
    print("Recording keyboard events... Press 'esc' to stop recording.")

    # Create and start keyboard listener
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()