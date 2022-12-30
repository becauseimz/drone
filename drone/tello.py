import socket
import threading
import time
from stats import Stats

class Tello:
    def __init__(self):
        pass

    def send_command(self, command):
        if "?" in command:
            return "800mm\r\n"
        print(f"command: {command}")

    def _receive_thread(self):
        pass

    def on_close(self):
        pass

    def get_log(self):
        pass
