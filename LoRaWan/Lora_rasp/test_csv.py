"""
Writes in real time the csv file using simulated data.
"""


import time
from math import *
import csv
import os
import random as rd

starting_time = time.time()


def get_pos(ID):
    print(starting_time)
    pos_time = time.time()
    print(pos_time)
    if ID == 0:
        x = 5000 * cos(0.5 * pos_time)
        y = 3000 * sin(0.5 * pos_time)
        z = 4000
    if ID == 1:
        x = (pos_time - starting_time) * 400
        y = 4000
        z = 1000
        print(x, y, z)
    if ID == 2:
        x = 5000
        y = 3000 * cos(0.2 * pos_time)
        z = (pos_time - starting_time) * 400
    return x, y, z


def create_ex_csv(csv_path, lines_per_seconds=5):
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
        cost = cos(2 * 100 * current_time) ** 2
        ID = rd.randint(0, 2)
        x, y, z = get_pos(ID)
        confidence = rd.randint(0, 100) / 100
        start_time = time.time() - 10
        end_time = time.time() - rd.randint(0, 10)
        birdnet_id = ["rufwar1", "blfwar1", "yebwar1"][ID]
        sim_data = [
            current_time,
            x,
            y,
            z,
            cost,
            ID,
            confidence,
            start_time,
            end_time,
            birdnet_id,
        ]
        mode = "a" if os.path.isfile(csv_path) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(columns)
            writer.writerow(sim_data)

        time.sleep(1 / lines_per_seconds)


if __name__ == "__main__":
    create_ex_csv("lora_data.csv")
