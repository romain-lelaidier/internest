import asyncio
import socket
from CONFIG import *
import time
from pathlib import Path
import os


async def udp_receiver(port, esp_id):
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    # print(f"Listening for {device_name} on port {port}...")

    while True:
        message, addr = await loop.sock_recvfrom(sock, CHUNK_SIZE)
        esp_time = int.from_bytes(message[:8], "little")
        micros = round(time.time() * 1e6)
        open(Path(PATH_TO_FOLDER) / esp_id / f"{micros}.bin", "wb").write(message[8:])
        print(
            f"packet of length {len(message)-8} from {esp_id} with ts {esp_time} (delta = {micros - esp_time})"
        )


async def main():

    for esp_id in esp_ids:
        os.makedirs(Path(PATH_TO_FOLDER) / esp_id, exist_ok=True)

    tasks = [
        asyncio.create_task(udp_receiver(port, device))
        for port, device in zip(ports, esp_ids)
    ]
    await asyncio.gather(*tasks)


asyncio.run(main())
