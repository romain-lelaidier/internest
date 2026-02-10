import numpy as np
from scipy.io.wavfile import write
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer

from config import CONFIG

n_samples = 0

print("Chargement du modele BirdNET...")
analyzer = Analyzer()
print("Modele BirdNET chargé.")

class Sample:
    def __init__(self, origin, s):
        global n_samples
        self.origin = origin
        self.s = s
        self.n = n_samples
        n_samples += 1

    def save(self):
        write(f"./out/{self.n}.wav", CONFIG.SAMPLE_RATE, np.int16(self.s / np.max(np.abs(self.s)) * 32767))

    def analyze(self):
        recording = RecordingBuffer(
            analyzer, self.s, CONFIG.SAMPLE_RATE,
            min_conf=CONFIG.BIRDNET_MIN_CONFIDENCE_2
        )
        # print(f"[L2] Analyse BirdNET pour {mac} ({len(samples)} samples, {CONFIG.BIRDNET_WINDOW_2_S}s)")
        recording.analyze()

        printer = f" ! DETECTION ! {self.n} ; origin = {self.origin})"

        for det in recording.detections:
            species = det['common_name']
            printer += f" ({species})"

        print(printer)
        
        # is_new = species not in active_species
        # if is_new:
        #     print(f"[L2] >>> {species} sur {mac} (conf {det['confidence']:.2f})")
        #     if CONFIG.AFFICHAGE_IHM:
        #         notify_arrival(mac, species, det['confidence'])
        # else:
        #     print(f"[L2] === {species} toujours sur {mac} (conf {det['confidence']:.2f})")
        # active_species[species] = now
