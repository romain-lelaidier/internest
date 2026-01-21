import socket
import pyaudio

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Network parameters
server_ip = "172.16.16.240"  # Replace with the server's IP
server_port = 8888

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    while True:
        data = stream.read(CHUNK)
        sock.sendto(data, (server_ip, server_port))
except KeyboardInterrupt:
    print("Stopping audio stream...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sock.close()
