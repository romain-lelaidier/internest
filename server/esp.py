import numpy as np
from scipy import signal

THEORETICAL_SAMPLE_RATE = 48000     # sample rate théorique des ESP
MAX_WINDOW_S = 20                    # durée maximale d'enregistrement (au-delà, on oublie les paquets)

WINDOW_BUFFER_SIZE = MAX_WINDOW_S * THEORETICAL_SAMPLE_RATE

class ESP:
    def __init__(self, mac, id):
        self.mac = mac
        self.id = id
        self.buffer = np.zeros(WINDOW_BUFFER_SIZE, dtype=np.int16)
        self.silenced = np.zeros(WINDOW_BUFFER_SIZE, dtype=bool)
        self.t0 = None
        self.buffer_end_t = None
        self.last_written_index = 0

    def time_to_index(self, t):
        return round((t - self.t0) * THEORETICAL_SAMPLE_RATE / 1e6)
    
    def index_to_time(self, i):
        return self.t0 + i * 1e6 / THEORETICAL_SAMPLE_RATE

    def receive_packet(self, t2, samples):
        # un paquet vient d'être reçu par le serveur UDP.
        # cette fonction l'interpole pour prendre en compte la fréquence d'échantillonnage effective.

        # calcul de la durée effective du paquet
        default_duration = len(samples) * 1e6 / THEORETICAL_SAMPLE_RATE

        if self.t0 == None:
            # premier échantillon reçu
            t1 = t2 - default_duration
            interpolated_samples = samples
            self.t0 = t1

        elif self.buffer_end_t < t2 - default_duration * 1.1:
            # on fait l'hypothèse que la fréquence d'échantillonnage effective est égale à la théorique
            t1 = t2 - default_duration
            interpolated_samples = samples
        
        else:
            # la fréquence d'échantillonnage effective est calculée en fonction de l'écart entre les deux paquets
            # les samples sont rééchantillonés à la fréquence d'échantillonnage théorique
            t1 = self.buffer_end_t
            effective_sample_rate = len(samples) * 1e6 / (t2 - t1)
            n_interpolated_samples = int(len(samples) * THEORETICAL_SAMPLE_RATE / effective_sample_rate)
            interpolated_samples = signal.resample(samples, n_interpolated_samples)

        write_at = self.time_to_index(t1)

        for i in range(write_at - self.last_written_index):
            self.buffer[(self.last_written_index + i) % WINDOW_BUFFER_SIZE] = 0
            self.silenced[(self.last_written_index + i) % WINDOW_BUFFER_SIZE] = True
        for i in range(len(interpolated_samples)):
            self.buffer[(write_at + i) % WINDOW_BUFFER_SIZE] = interpolated_samples[i]
            self.silenced[(write_at + i) % WINDOW_BUFFER_SIZE] = False

        self.last_written_index = write_at + len(interpolated_samples)
        self.buffer_end_t = t2

    def read_window(self, t1, t2):
        # retourne une fenêtre audio donnant l'intersection maximale entre t1 et t2
        # renvoie t1_reel, t2_reel, samples

        if self.t0 == None or self.buffer_end_t == None:
            return 0, 0, np.array([], np.int16)
        
        t_min_av = self.index_to_time(max(0, self.last_written_index - WINDOW_BUFFER_SIZE))

        t_r1 = max(t1, t_min_av)
        t_r2 = min(t2, self.buffer_end_t)

        if t_r1 >= t_r2:
            return 0, 0, np.array([], np.int16)

        i_r1 = self.time_to_index(t_r1)
        i_r2 = self.time_to_index(t_r2)

        if i_r1 // WINDOW_BUFFER_SIZE == i_r2 // WINDOW_BUFFER_SIZE:
            return int(t_r1), int(t_r2), self.buffer[i_r1 % WINDOW_BUFFER_SIZE : i_r2 % WINDOW_BUFFER_SIZE]
        
        return int(t_r1), int(t_r2), np.concatenate((self.buffer[i_r1 % WINDOW_BUFFER_SIZE : WINDOW_BUFFER_SIZE], self.buffer[0 : i_r2 % WINDOW_BUFFER_SIZE]))