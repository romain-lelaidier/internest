import numpy as np
import socket
import asyncio
from multiprocessing import Process

from esp import ESP
from utils import micros, add_padding_zeros
from birdnet_loop import start_birdnet
from localisation import routine_localiser

PORT_AUDIO = 8002           # port d'écoute UDP pour les paquets audio
PORT_SYNC = 8001            # port d'écoute UDP pour les paquets audio
ESP_ID_LENGTH = 6           # nombre d'octets pour identifier un ESP
ESP_TIME_LENGTH = 8         # nombre d'octets pour le timestamp d'un paquet
PACKET_LENGTH = 1024*8*3    # nombre d'octets de data d'un paquet (sans les metadata)

esps = {}

async def use_packet(mac, esp_time, samples, rpi_time):
    if mac not in esps:
        esps[mac] = ESP(mac, len(esps))
        print(f"new ESP: {mac}")
        start_birdnet(mac, esps[mac])
    print(f" ⸱ received a packet of length {len(samples)} from {mac} with ts {esp_time} (delta = {rpi_time - esp_time})")
    esps[mac].receive_packet(esp_time, samples)

def routine_audio_server(_):
    global esps

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT_AUDIO))

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

def routine_sync_server(_):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT_SYNC))

    while True:
        message, address = server_socket.recvfrom(64)
        printer = f"msg {message} from {address}"

        t = micros()
        rmsg = message + t.to_bytes(8, 'little')
        printer += f" {t} len={len(rmsg)}"

        # print(printer)
        server_socket.sendto(rmsg + rmsg + b'\0', address)

def routine_wrapper(func):
    while True:
        try:
            func()
        except Exception as err:
            print(err)

if __name__ == "__main__":
    routines = [
        routine_audio_server,
        routine_sync_server,
        routine_localiser
    ]

    for routine in routines:
        Process(target=routine_wrapper, args=(routine,)).start()