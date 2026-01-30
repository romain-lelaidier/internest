import matplotlib.pyplot as plt
import numpy as np
from CONFIG import *


def verbose_data_bin_tab(file, dates, data):
    print("len(binary_file) = ", len(file)), print()
    print("Information on dates : ")
    print("dates[0] = ", dates[0])
    print("len(dates) = ", len(dates))
    print("len(dates[0]) = ", len(dates[0])), print()
    print("Information on data : ")
    print("data[0][:8] = ", data[0][:8])
    print("len(data) = ", len(data))
    print("len(data[0]) = ", len(data[0]))
    print([len(data[i]) for i in range(len(data))])


def date_data_tab(file_path, verbose=False):
    with open(file_path, "rb") as fb:
        file = fb.read()
        dates = [
            file[CHUNK_SIZE // 8 * i : CHUNK_SIZE // 8 * i + BYTES_FOR_DATE]
            for i in range(CHUNK_PER_FILE)
        ]
        data = [
            file[CHUNK_SIZE // 8 * i + BYTES_FOR_DATE : CHUNK_SIZE // 8 * (i + 1)]
            for i in range(CHUNK_PER_FILE)
        ]
    data = [
        [
            data[i][j * BITS_PER_POINT // 8 : (j+1) * BITS_PER_POINT // 8] for j in range(len(data[i]) // (BITS_PER_POINT // 8))
        ]  # data[i][j] converts the signed binary to int as if it were unsigned
        for i in range(len(data))
    ]
    if verbose:
        verbose_data_bin_tab(file, dates, data)

    return dates, np.array(data)


def time_decryption(encrypted_time):
    time_int = int.from_bytes(encrypted_time, "little")
    max_value = 3600
    max_int = 2 ** (8 * BYTES_FOR_DATE)
    hour = time_int * max_value / max_int  # seconds of the current hour
    return hour


def point_decryption(encrypted_point):
    int_point = int.from_bytes(encrypted_point, "little", signed=True)
    max_int = 2**BITS_PER_POINT
    return int_point / max_int * SIGNAL_MAX_VALUE


def to_human_readable(dates, data, verbose=False):
    dates = [time_decryption(date) for date in dates]
    data = [
        [point_decryption(data[i, j]) for j in range(len(data[i]))]
        for i in range(len(data))
    ]
    data = np.array(data)

    if verbose:
        print("len(dates) : ", len(dates), "; dates[:5] : ", dates[:5])
        print("len(data) : ", len(data), "; data[0, :5] : ", data[0, :5])

    return dates, data


def plot_series(date, series):
    time_series = np.linspace(0, len(series) / FREQ_E * 1000, len(series))
    plt.scatter(time_series, series)
    plt.xlabel("time (ms)")
    plt.ylabel("Signal")
    plt.title(f"decrypted signal. Start time : {date}")
    plt.show()


if __name__ == "__main__":
    FILE_PATH = "C:/Users/colin/COLIN/Mines/TI_IDS/Internest/internest/udp/bin_f/ESP32_1/1769769551281198.bin"
    bin_dates, bin_data = date_data_tab(FILE_PATH, verbose=True)
    dates, data = to_human_readable(bin_dates, bin_data, verbose=True)

    series_number = 0
    first_series = data[series_number, :]
    first_date = dates[series_number]

    plot_series(first_date, first_series)
