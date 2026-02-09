import numpy as np
from scipy import signal
import threading

from config import CONFIG
from birdnet_loop import _esp_loop
from utils import micros
from scipy.io.wavfile import write

WINDOW_BUFFER_SIZE = CONFIG.MAX_WINDOW_S * CONFIG.SAMPLE_RATE

class ESP:
    def __init__(self, mac, id):
        self.mac = mac
        self.id = id
        self.buffer = np.zeros(WINDOW_BUFFER_SIZE, dtype=np.int16)
        self.silenced = np.zeros(WINDOW_BUFFER_SIZE, dtype=bool)
        self.birdnet_thread = None
        self.frequency = 1000 + id * 100
        self.init_window()

    def init_window(self):
        self.t0 = None
        self.buffer_end_t = None
        self.last_written_index = 0
        self.buzzobjs = []
        self.synced = False
        self.coordinates = None

    def time_to_index(self, t):
        return round((t - self.t0) * CONFIG.SAMPLE_RATE / 1e6)
    
    def index_to_time(self, i):
        return self.t0 + i * 1e6 / CONFIG.SAMPLE_RATE
    
    def register_buzztime(self, te, f):
        self.buzzobjs.append({
            'te': te,
            'f': f
        })
        print(self.buzzobjs)
    
    def start_birdnet(self):
        if self.birdnet_thread != None and self.birdnet_thread.is_alive():
            return
        self.birdnet_thread = threading.Thread(target=_esp_loop, args=(self.mac, self), daemon=True)
        self.birdnet_thread.start()
        print(f"Analyse BirdNET demarree pour {self.mac}")

    def try_assign_buzz(self, samples, t1, t2):
        return
        f, i = look_for_buzzes(samples, self.frequency)
        if i > 0:
            for buzzobj in self.buzzobjs:
                if np.abs(buzzobj['f'] - f) < 5:
                    buzzobj['t'] = t1 + (t2 - t1) * i / len(samples)

    def receive_packet(self, t2, samples):
        # un paquet vient d'être reçu par le serveur UDP.
        # cette fonction l'interpole pour prendre en compte la fréquence d'échantillonnage effective.

        # calcul de la durée effective du paquet
        default_duration = len(samples) * 1e6 / CONFIG.SAMPLE_RATE

        if self.t0 == None:
            # premier échantillon reçu
            t1 = t2 - default_duration
            interpolated_samples = samples
            self.t0 = t1
            self.start_birdnet()

        elif self.buffer_end_t < t2 - default_duration * 0:
            # on fait l'hypothèse que la fréquence d'échantillonnage effective est égale à la théorique
            t1 = t2 - default_duration
            interpolated_samples = samples
        
        else:
            # la fréquence d'échantillonnage effective est calculée en fonction de l'écart entre les deux paquets
            # les samples sont rééchantillonés à la fréquence d'échantillonnage théorique
            t1 = self.buffer_end_t
            effective_sample_rate = len(samples) * 1e6 / (t2 - t1)
            n_interpolated_samples = int(len(samples) * CONFIG.SAMPLE_RATE / effective_sample_rate)
            interpolated_samples = signal.resample(samples, n_interpolated_samples)

        write_at = self.time_to_index(t1)

        for i in range(write_at - self.last_written_index):
            self.buffer[(self.last_written_index + i) % WINDOW_BUFFER_SIZE] = 0
            self.silenced[(self.last_written_index + i) % WINDOW_BUFFER_SIZE] = True
        for i in range(len(interpolated_samples)):
            self.buffer[(write_at + i) % WINDOW_BUFFER_SIZE] = interpolated_samples[i]
            self.silenced[(write_at + i) % WINDOW_BUFFER_SIZE] = False

        self.try_assign_buzz(samples, t1, t2)

        self.last_written_index = write_at + len(interpolated_samples)
        self.buffer_end_t = t2

        # write('test.wav', CONFIG.SAMPLE_RATE, np.int16(self.buffer / np.max(np.abs(self.buffer)) * 32767))

    def read_window_esp(self, t1, t2):
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

    def t_to_te(self, t):
        if self.synced:
            return None
        return t

    def read_window(self, t1, t2):
        if self.synced:
            return self.read_window_esp(self.t_to_te(t1), self.t_to_te(t2))
        return -1, -1, np.array([], dtype=np.int16)