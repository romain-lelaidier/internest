FREQ_E = 48000
SIGNAL_MAX_VALUE = 1

CHUNK_SIZE = 2048 * 3 + 64  # bits
CHUNK_PER_FILE = 1

# Network parameters
localPort = 8888
localPort2 = 5000
server_ip = "172.16.16.240"
PATH_TO_FOLDER = "bin_f"

ports = [localPort, localPort2]
esp_ids = ["ESP32_1", "ESP32_2"]

BITS_PER_POINT = 24
BYTES_FOR_DATE = 8

assert (CHUNK_SIZE - BYTES_FOR_DATE * 8) // BITS_PER_POINT == (
    CHUNK_SIZE - BYTES_FOR_DATE * 8
) / BITS_PER_POINT, "The available space for data in the chunk is not a multiple of the size of one point."
