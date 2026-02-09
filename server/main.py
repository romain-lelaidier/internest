import numpy as np
import scipy.fft
import scipy.signal
import socket
import threading
import time

# Pré-initialiser scipy FFT sur le main thread pour éviter
# "can't register atexit after shutdown" dans les threads
scipy.fft.fft(np.zeros(256))

from esp import ESP
from utils import micros, add_padding_zeros
from localisation import routine_localiser
from config import CONFIG
from ihm_birdnet import start_ihm
from ihm_localisation import start_ihm_localisation

esps = {}

def use_packet(mac, esp_time, samples, rpi_time):
    global esps
    if mac not in esps:
        esps[mac] = ESP(mac, len(esps))
        print(f"new ESP: {mac}")
    print(f" ⸱ received a packet of length {len(samples)} from {mac} with ts {esp_time} (delta = {rpi_time - esp_time})")
    esps[mac].receive_packet(esp_time, samples)

def routine_audio_server(_):
    global esps

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', CONFIG.PORT_AUDIO))

    while True:
        message, address = server_socket.recvfrom(CONFIG.ESP_ID_LENGTH + CONFIG.ESP_TIME_LENGTH + CONFIG.PACKET_LENGTH)

        if len(message) == CONFIG.ESP_ID_LENGTH + CONFIG.ESP_TIME_LENGTH + CONFIG.PACKET_LENGTH:
            # décodage du paquet
            rpi_time = micros()
            mac = ':'.join(list(map(lambda c: add_padding_zeros(hex(c)[2:], 2), message[0 : CONFIG.ESP_ID_LENGTH])))
            esp_time = int.from_bytes(message[CONFIG.ESP_ID_LENGTH : CONFIG.ESP_ID_LENGTH + CONFIG.ESP_TIME_LENGTH], 'little')
            samples = np.frombuffer(message[CONFIG.ESP_ID_LENGTH + CONFIG.ESP_TIME_LENGTH : ], dtype='<i2')

            # analyse
            use_packet(mac, esp_time, samples, rpi_time)

def routine_sync_server(_):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', CONFIG.PORT_SYNC))

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
            func(esps)
        except Exception as err:
            print(err)

if __name__ == "__main__":

    # On lance les IHM qui lancent leurs propres threads daemon.
    start_ihm()
    start_ihm_localisation()

    routines = [
        routine_audio_server,
        routine_sync_server,
        routine_localiser
    ]

    for routine in routines:
        threading.Thread(target=routine_wrapper, args=(routine,), daemon=True).start()

    while True:
        time.sleep(100)