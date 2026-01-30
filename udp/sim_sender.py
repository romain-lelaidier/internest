import socket
import math
import time
from CONFIG import *


def sin_sig(start_date, sig_type="sin"):
    freq1 = 440
    freq2 = 330
    buf = b""
    number_of_points = (CHUNK_SIZE - BYTES_FOR_DATE * 8) // BITS_PER_POINT
    for i in range(number_of_points):
        t = i / FREQ_E
        if sig_type == "sin":
            real_value = math.sin(2 * math.pi * freq1 * (t + start_date))
        if sig_type == "sum_sin":
            real_value = (math.sin(2 * math.pi * freq1 * (t + start_date)) + math.sin(2 * math.pi * freq2 * (t + start_date))) / 2
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
    micros = round(time.time() * 1e6)
    return micros.to_bytes(8, "little"), micros // 1e6


def send_signal(server_ip, server_port, sig_type="sin"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        i = 0
        while True:
            date, start_date = encoded_byte_date()
            data = sin_sig(start_date, sig_type)
            signal = date + data
            sock.sendto(signal, (server_ip, server_port))
            if i % 25 == 0:
                print(f"Chunk sent of size {len(signal)}o at date = {start_date}")
            i += 1
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        sock.close()


if __name__ == "__main__":
    send_signal(server_ip, localPort)
