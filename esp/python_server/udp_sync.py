import socket
import time

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('', 8001))

while True:
    message, address = server_socket.recvfrom(1024)
    message = message.upper()

    if len(message) == 8 and message[0:6] == b'/TIME/':
        micros = round(time.time() * 1e6)
        print(message, address, micros)
        server_socket.sendto(message + micros.to_bytes(8, 'big'), address)
    
    else:
        print(message)