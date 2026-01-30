import matplotlib.pyplot as plt
import numpy as np
from CONFIG import *
import os
from pathlib import Path
import time
import sounddevice as sd


class ReadBin:
    def __init__(self):
        self.current_time = time.time()

    def date_data_tab(self, file_path):
        with open(file_path, "rb") as fb:
            file = fb.read()
            data = [
                file[BITS_PER_POINT // 8 * i : BITS_PER_POINT // 8 * (i + 1)]
                for i in range(len(file) // (BITS_PER_POINT // 8))
            ]

        return data

    def point_decryption(self, encrypted_point):
        int_point = int.from_bytes(encrypted_point, "little", signed=True)
        max_int = 2**BITS_PER_POINT
        return int_point / max_int * SIGNAL_MAX_VALUE

    def to_human_readable(self, data):
        data = [self.point_decryption(data[i]) for i in range(len(data))]
        return data

    def plot_series(self, series, time_series=None):
        if time_series is None:
            time_series = np.linspace(0, len(series) / FREQ_E * 1000, len(series))
        plt.scatter(time_series, series)
        plt.xlabel("time (ms)")
        plt.ylabel("Signal")
        plt.title(f"decrypted signal. Start time : {time_series[0]}")
        plt.show()

    def hour_time(self, epoch_time):
        """Returns time in seconds since beginning of hour (for epoch_time)"""
        return epoch_time - (self.current_time - self.current_time % 3600)

    def concat_data(self, file_path_list):
        times = []
        conc_data = []
        for path in file_path_list:
            bin_data = self.date_data_tab(path)
            data = self.to_human_readable(bin_data)
            conc_data += data
            start_time = float(path.name[:-4]) / 1e6  # convert in seconds
            start_time = round(self.hour_time(start_time) * 1e3)  # converts in ms
            time_series = [start_time + 1000 * i / FREQ_E for i in range(len(data))]
            times += time_series
        return times, conc_data

    def file_path_list(self, esp_id, limit=3):
        esp_path = Path(PATH_TO_FOLDER) / esp_id
        file_paths = [Path(esp_path / file) for file in os.listdir(esp_path)]
        return file_paths[:limit]
    
    def play_array(self, arr):
        sd.play(arr, samplerate=FREQ_E)
        sd.wait()


if __name__ == "__main__":
    Reader = ReadBin()
    file_paths = Reader.file_path_list("ESP32_1", limit=200)
    print(file_paths)
    times, conc_data = Reader.concat_data(file_paths)
    Reader.play_array(np.array(conc_data))
    Reader.plot_series(conc_data, time_series=times)
