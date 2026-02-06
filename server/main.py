import socket
import asyncio
import numpy as np

from esp import ESP
from utils import micros, add_padding_zeros

PORT = 8002                 # port d'écoute UDP
ESP_ID_LENGTH = 6           # nombre d'octets pour identifier un ESP
ESP_TIME_LENGTH = 8         # nombre d'octets pour le timestamp d'un paquet
PACKET_LENGTH = 1024*8*3    # nombre d'octets de data d'un paquet (sans les metadata)

esps = {}

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('', PORT))

async def use_packet(mac, esp_time, samples, rpi_time):
    if mac not in esps:
        esps[mac] = ESP(mac)
        print(f"new ESP: {mac}")
    print(f" ⸱ received a packet of length {len(samples)} from {mac} with ts {esp_time} (delta = {rpi_time - esp_time})")
    esps[mac].receive_packet(esp_time, samples)
    t1, t2, s = esps[mac].read_window(10*1e6, 15*1e6)
    print(t1, t2, len(s))

while True:
    message, address = server_socket.recvfrom(ESP_ID_LENGTH + ESP_TIME_LENGTH + PACKET_LENGTH)

    if len(message) == ESP_ID_LENGTH + ESP_TIME_LENGTH + PACKET_LENGTH:
        # décodage du paquet
        rpi_time = micros()
        mac = ':'.join(list(map(lambda c: add_padding_zeros(hex(c)[2:], 2), message[0 : ESP_ID_LENGTH])))
        esp_time = int.from_bytes(message[ESP_ID_LENGTH : ESP_ID_LENGTH + ESP_TIME_LENGTH], 'little')
        samples = np.frombuffer(message[ESP_ID_LENGTH + ESP_TIME_LENGTH : ], dtype='<i2')

        # analyse
        asyncio.run(use_packet(mac, esp_time, samples, rpi_time))
