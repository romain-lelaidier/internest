import numpy as np
import scipy.fft
import socket
import threading
import time

scipy.fft.fft(np.zeros(256))

from esp import ESP
from utils import micros, add_padding_zeros
from localisation2 import routine_localiser
from config import CONFIG
from ihm_birdnet import start_ihm
from ihm_localisation import start_ihm_localisation

# Imports pour la calibration (ceux qu'on a créés précédemment)
# from calibration import run_step_by_step_calibration, POSITIONS_FILE
# On importe le contrôleur pour pouvoir envoyer les commandes de buzz pendant la calib
# from esp_controller import ESPController

esps = {}
esps_positions = {
    "1c:db:d4:34:79:6c": [ 0, 0, 0],        # vert
    "1c:db:d4:38:43:00": [ 0, 5, 0 ],       # rouge
    "1c:db:d4:36:6f:5c": [ 7, 0, 0 ],       # jaune avec buzzer
    "1c:db:d4:34:5c:04": [ 5, 6, 2.3 ],     # bleu
    "1c:db:d4:33:6a:78": [ ],               # jaune sans buzzer
}

def handle_request(message):
    mac = ':'.join(list(map(lambda c: add_padding_zeros(hex(c)[2:], 2), message[0 : CONFIG.ESP_ID_LENGTH])))
    code = message[CONFIG.ESP_ID_LENGTH]
    payload = message[CONFIG.ESP_ID_LENGTH + 1 :]

    if mac not in esps:
        esps[mac] = ESP(mac, len(esps), esps_positions[mac])
        print(f"new ESP: {mac} {esps[mac].position}")

    if code == 0:
        esps[mac].init_window()
        print(f"esp {mac} initalized")
        return esps[mac].frequency.to_bytes(8, 'little')

    esp_time = int.from_bytes(payload[0 : CONFIG.ESP_TIME_LENGTH], 'little')
    print(code, mac, esp_time, len(payload))

    if code == 1:
        f = int.from_bytes(payload[CONFIG.ESP_TIME_LENGTH : CONFIG.ESP_TIME_LENGTH + 8], 'little')
        esps[mac].register_buzztime(esp_time, f)

    if code == 2:
        samples = np.frombuffer(payload[CONFIG.ESP_TIME_LENGTH : ], dtype='<i2')
        esps[mac].receive_packet(esp_time, samples)

def handle_client(client_socket):
    global esps
    # printing what the client sends
    while True:
        request = client_socket.recv(128)
        rtmsg = handle_request(request)
        if rtmsg:
            client_socket.send(rtmsg)

def routine_buzz_server(_):
    global esps
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', CONFIG.PORT_BUZZER))
    server_socket.listen(10)
    while True: 
        client, addr = server_socket.accept()
        print(f"[TCP] Accepted connection from: {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=handle_client, args=(client,))
        client_handler.start()

def routine_audio_server(_):
    global esps
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', CONFIG.PORT_AUDIO))
    while True:
        message, address = server_socket.recvfrom(CONFIG.ESP_ID_LENGTH + 1 + CONFIG.ESP_TIME_LENGTH + CONFIG.PACKET_LENGTH)
        if len(message) == CONFIG.ESP_ID_LENGTH + 1 + CONFIG.ESP_TIME_LENGTH + CONFIG.PACKET_LENGTH:
            # décodage du paquet
            handle_request(message)

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
        server_socket.sendto(rmsg + rmsg + b'\0', address)

def routine_wrapper(func):
    while True:
        try:
            func(esps)
        except Exception as err:
            print(err)

if __name__ == "__main__":

    # On lance les IHM qui lancent leurs propres threads daemon.
    # start_ihm()
    # start_ihm_localisation()

    routines = [
        routine_audio_server,
        routine_sync_server,
        routine_localiser,
        # routine_buzz_server,
        # run_step_by_step_calibration
    ]

    for routine in routines:
        threading.Thread(target=routine_wrapper, args=(routine,)).start()

    while True:
        time.sleep(100)