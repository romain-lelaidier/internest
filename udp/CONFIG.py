FREQ_E = 48000
SIGNAL_MAX_VALUE = 1

CHUNK_SIZE = 2048 + 8  # bits
CHUNK_PER_FILE = 100

# Network parameters
localPort = 8888
server_ip = "172.16.16.240"

BITS_PER_POINT = 24
BYTES_FOR_DATE = 8

assert (CHUNK_SIZE - BYTES_FOR_DATE * 8) // BITS_PER_POINT == (
    CHUNK_SIZE - BYTES_FOR_DATE * 8
) / BITS_PER_POINT, "The available space for data in the chunk is not a multiple of the size of one point."
