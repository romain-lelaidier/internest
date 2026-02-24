import socket
import time
import os

PORT = 8002
PACKET_LEN = 1024*8*3

dir = './packets'
os.makedirs(dir, exist_ok=True)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('', PORT))

def add_zeros_before(c, n):
    # adding zeros before c until length is n
    while len(c) < n:
        c = '0' + c
    return c

while True:
    message, address = server_socket.recvfrom(6 + 8 + PACKET_LEN)

    if len(message) > 9:
        mac = ':'.join(list(map(lambda c: add_zeros_before(hex(c)[2:], 2), message[0:6])))
        # esp_id = int(message[0])
        esp_time = int.from_bytes(message[6:6+8], 'little')
        micros = round(time.time() * 1e6)
        open(f"{dir}/{mac}_{esp_time}.bin", "wb").write(message[6+8:])

        print(f"packet of length {len(message)-14} from {mac} with ts {esp_time} (delta = {micros - esp_time})")