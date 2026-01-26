import socket
import math
import time
from CONFIG import *


def sin_sig(start_date):
    freq = 440
    buf = b""
    number_of_points = (CHUNK_SIZE - BYTES_FOR_DATE * 8) // BITS_PER_POINT
    for i in range(number_of_points):
        t = i / FREQ_E
        real_value = math.sin(2 * math.pi * freq * (t + start_date))
        encoded_value = round(
            real_value * 2 ** (BITS_PER_POINT - 1) / SIGNAL_MAX_VALUE
        )
        if encoded_value == 2 ** (BITS_PER_POINT - 1):
            encoded_value -= 1
        if encoded_value == - 2 ** (BITS_PER_POINT - 1):
            encoded_value += 1   
        encoded_value = encoded_value.to_bytes(BITS_PER_POINT // 8, "little", signed=True)
        buf += encoded_value
        time.sleep(1 / FREQ_E)
    return buf


def encoded_byte_date():
    date = time.time()
    # on choisit une précision à l'heure.
    max_value = 3600
    max_int = 2 ** (8 * BYTES_FOR_DATE)
    hour = date % max_value  # seconds of the current hour
    return round(hour / max_value * max_int).to_bytes(8, "little"), hour


def send_signal(server_ip, server_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while True:
            date, start_date = encoded_byte_date()
            data = sin_sig(start_date)
            signal = date + data
            sock.sendto(signal, (server_ip, server_port))
            print(f"Chunk sent of size {len(signal)}o at date = {start_date}")
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        sock.close()


if __name__ == "__main__":
    send_signal(server_ip, localPort)
