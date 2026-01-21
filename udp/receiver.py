import socket
import time
from pathlib import Path
import numpy as np
from io import BytesIO

CHANNELS = 1
RATE = 44100
CHUNK = 4096
FILE_DURATION = 1  # seconds

# Network parameters
localPort = 8888


def listening():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", localPort))
    file_number = 0
    while True:
        file_path = Path("bin_f") / ("array" + str(file_number) + ".bin")
        receiving_one_array(sock, file_path, time.perf_counter())
        file_number += 1


def receiving_one_array(sock, file_path, time_init):
    # Buffer to store audio chunks
    audio_buffer = bytearray()
    while time.perf_counter() - time_init < FILE_DURATION:
        data, addr = sock.recvfrom(CHUNK)
        audio_buffer.extend(data)

    with open(file_path, "wb") as f:
        # print(audio_buffer)
        audio_buffer = BytesIO(audio_buffer)
        print(type(audio_buffer))
        f.write(audio_buffer.getbuffer())


if __name__ == "__main__":
    listening()
