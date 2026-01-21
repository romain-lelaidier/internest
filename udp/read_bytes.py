import struct
import matplotlib.pyplot as plt
import numpy as np
import sys

def time_array(dt, n):
    return np.linspace(0, n * dt, n)

def float_from_bin(file_path, verbose=True):
    with open(file_path, "rb") as binary_file:
        buf = binary_file.read()
        series = struct.unpack("%sf" % (len(buf) // 4), buf)  # Unpack all floats
        if verbose:
            print(series[:5])
            print("number of bytes in read file : ", sys.getsizeof(buf))
    return series

def plot_series(time_, series, limit = 100):
    plt.plot(time_[:limit], series[:limit])


if __name__ == "__main__":
    FILE_PATH = (
        "C:/Users/colin/COLIN/Mines/TI_IDS/Internest/internest/udp/bin_f/array0.bin"
    )
    dt = 10
    series = float_from_bin(FILE_PATH)
    n = len(series)
    time_ = time_array(dt, n)
    plot_series(time_, series)

    plt.show()
