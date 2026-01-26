import socket
from pathlib import Path
import datetime
import os
from CONFIG import *


def listening():
    os.makedirs("bin_f", exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", localPort))
    while True:
        start_time = str(datetime.datetime.now())
        start_time = start_time.replace(" ", "_").replace(":", "-")
        file_path = Path("bin_f") / (start_time + ".bin")
        print(file_path)
        print(os.path.isdir("bin_f"))
        receiving_one_array(sock, file_path)


def receiving_one_array(sock, file_path):
    buf = b''
    chunk_num = 0
    print("start_rec")
    while chunk_num < CHUNK_PER_FILE:
        data, addr = sock.recvfrom(CHUNK_SIZE)
        buf += data
        chunk_num += 1
    print("done")
    with open(file_path, "wb") as f:
        f.write(buf)


if __name__ == "__main__":
    listening()
