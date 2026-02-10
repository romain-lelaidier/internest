import numpy as np
from scipy.io.wavfile import write

from config import CONFIG

n_samples = 0

class Sample:
    def __init__(self, origin, s):
        global n_samples
        self.origin = origin
        self.s = s
        self.n = n_samples
        n_samples += 1

    def save(self):
        write(f"./out/{self.n}.wav", CONFIG.SAMPLE_RATE, np.int16(self.s / np.max(np.abs(self.s)) * 32767))
