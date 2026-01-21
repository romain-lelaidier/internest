import socket
import math
import time
import struct


def sin_sig(n=1024, verbose=True):
    freq_e = 1000
    freq = 440
    L = []
    for i in range(n):
        t = i / freq_e
        L.append(math.sin(2 * math.pi * freq * t))
        time.sleep(1 / freq_e)
    buf = struct.pack("%sf" % len(L), *L)
    if verbose:
        print("L : ", L)
        print("buf : ", buf)
    return buf


def send_signal(server_ip, server_port):
    # Initialize UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            data = sin_sig()
            sock.sendto(data, (server_ip, server_port))
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        sock.close()


if __name__ == "__main__":
    # Network parameters
    server_ip = "172.16.16.240"  # Replace with the server's IP
    server_port = 8888

    send_signal(server_ip, server_port)
