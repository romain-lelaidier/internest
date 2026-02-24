import socket
import time
import random

# config
PORT = 8001
packet_loss_prob = 0

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('', PORT))
print(f"UDP sync server listening on port {PORT}")

def micros():
	return round(time.time() * 1e6)

while True:
	message, address = server_socket.recvfrom(64)
	printer = f"msg {message} from {address}"

	t = micros()
	pin_written_at = t
	rmsg = message + t.to_bytes(8, 'little')
	printer += f" {t}"

	if random.random() < packet_loss_prob:
		print("dropping socket")
	else:
		print(printer)
		server_socket.sendto(rmsg + rmsg + '\0', address)
