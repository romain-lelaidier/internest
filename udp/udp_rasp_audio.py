import socket
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import io

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FILE_DURATION = 10  # seconds

# Network parameters
localPort = 8888

def compute_audio_length(buffer, sample_width=2):
    total_bytes = len(buffer)
    length_seconds = total_bytes / (RATE * CHANNELS * sample_width)
    return length_seconds


# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', localPort))

# Buffer to store audio chunks
audio_buffer = bytearray()

while True:
    while compute_audio_length(audio_buffer) < 10:
        data, addr = sock.recvfrom(CHUNK)
        audio_buffer.extend(data)

    with open("temp_audio.wav", "wb") as f:
        f.write(bytes(audio_buffer))

        # Convert WAV to MP3 using pydub
        audio = AudioSegment.from_wav("temp_audio.wav")
        audio.export("received_audio.mp3", format="mp3")

        print("Audio saved as 'received_audio.mp3'")
        sock.close()


