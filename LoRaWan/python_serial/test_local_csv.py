"""
Simulates real time data writing into a csv file.
It's a local simulation, not to be integrated in the real channel.
"""

import time
from math import *
import csv
import os
import random as rd


def create_ex_csv(csv_path, lines_per_seconds=2):
    columns = [
        "time",
        "x",
        "y",
        "z",
        "cost",
        "id",
        "confidence",
        "start_time",
        "end_time",
        "birdnet_id",
    ]
    while True:
        current_time = time.time()
        x = 5 * cos(1000 * current_time)
        y = 5 * sin(1000 * current_time) ** 2
        z = 10 * cos(1000 * current_time + 1)
        cost = cos(2 * 1000 * current_time) ** 2
        ID = rd.randint(0,5)
        confidence = rd.randint(0,100) / 100
        start_time = time.time() - 10
        end_time = time.time() - rd.randint(0,10)
        birdnet_id = str(ID) + "_birdnet"
        sim_data = [current_time, x, y, z, cost, ID, confidence, start_time, end_time, birdnet_id]
        mode = "a" if os.path.isfile(csv_path) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(columns)
            writer.writerow(sim_data)

        time.sleep(1 / lines_per_seconds)


if __name__ == "__main__":
    create_ex_csv("lora_data.csv")
