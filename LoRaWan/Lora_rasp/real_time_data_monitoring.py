"""
Uses watchdog to track csv file evolution and send data as it is being written.
"""

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from i2c_master import I2CMaster
from smbus import SMBus

class CSVHandler(FileSystemEventHandler):
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback
        self.last_position = 0
        self._initialize_position()


    def _initialize_position(self):
        """Initialize the last read position to the end of the file."""
        if os.path.exists(self.file_path):
            self.last_position = os.path.getsize(self.file_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.file_path:
            print(f"Change detected in {self.file_path}")
            self._read_new_lines()

    def _read_new_lines(self):
        """Read only the new lines appended to the file."""
        with open(self.file_path, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            if new_lines:
                self.callback(new_lines)
            self.last_position = f.tell()

def process_new_lines(lines):
    """Process new lines (e.g., print or parse)."""
    i2c_master.send_lines(lines)

if __name__ == "__main__":

    addr = 0x8 # bus address
    bus = SMBus(1) # indicates /dev/ic2-1
    max_pos_value = 100 * 100  # cm (precision)
    max_cost_value = 1
    frequency = 5  # Hz
    i2c_master = I2CMaster(addr, bus, max_pos_value, max_cost_value, frequency)

    file_path = "/home/rivatonz/i2c_internest/lora_data.csv"  # Replace with your CSV file path
    event_handler = CSVHandler(file_path, process_new_lines)
    observer = Observer()
    print(os.path.dirname(file_path))
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    observer.start()
    print(f"Monitoring {file_path} for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
