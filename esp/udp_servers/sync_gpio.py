import socket
import time
import random
import RPi.GPIO as gpio

# config
PORT = 8001
packet_loss_prob = 0
rpin = 20
wpin = 21

# code
pin_written_at = 0
pin_received_at = 0
should_print = False

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('', PORT))
print(f"UDP sync server listening on port {PORT}")

def micros():
	return round(time.time() * 1e6)

def on_high_edge(arg):
	global in_request, pin_received_at, should_print
	pin_received_at = micros()
	gpio.output(wpin, 0)
	should_print = pin_received_at - pin_written_at
	# print(arg, pin_received_at - pin_written_at)

gpio.setmode(gpio.BCM)
gpio.setup(wpin, gpio.OUT)
gpio.setup(rpin, gpio.IN)
gpio.output(wpin, gpio.LOW)
gpio.add_event_detect(rpin, gpio.RISING, on_high_edge)

while True:
	if should_print:
		print(should_print)
		should_print = False
	
	message, address = server_socket.recvfrom(64)
	printer = f"msg {message} from {address}"

	t = micros()
	pin_written_at = t
	rmsg = message + t.to_bytes(8, 'little') + b'\0'
	printer += f" {t}"

	if random.random() < packet_loss_prob:
		print("dropping socket")
	else:
		gpio.output(wpin, 1)
		print(printer)
		server_socket.sendto(rmsg, address)
