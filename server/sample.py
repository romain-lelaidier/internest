import numpy as np
from scipy.io.wavfile import write
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer
import sys
from io import StringIO

from config import CONFIG

n_samples = 0

class NullIO(StringIO):
    def write(self, txt):
       pass

print("Chargement du modele BirdNET...")
analyzer = Analyzer()
print("Modele BirdNET chargé.")

class Sample:
    def __init__(self, origin, s):
        global n_samples
        self.origin = origin
        self.s = s
        self.n = n_samples
        self.species = []
        n_samples += 1

    def save(self):
        # Pour sauvegarder le sample pour debug
        write(f"./out/{self.n}.wav", CONFIG.SAMPLE_RATE, np.int16(self.s / np.max(np.abs(self.s)) * 32767))

    def analyze(self):
        recording = RecordingBuffer(
            analyzer, self.s, CONFIG.SAMPLE_RATE,
            min_conf=CONFIG.BIRDNET_MIN_CONFIDENCE_2
        )

        sys.stdout = NullIO()
        recording.analyze()
        sys.stdout = sys.__stdout__

        for det in recording.detections:
            species = det['common_name']
            confidence = det['confidence']
            self.species.append((species, confidence))

        self.log()

    def log(self):
        printer = f" ! DETECTION ! {self.n} ; origin = {self.origin})"
        for species, confidence in self.species:
            printer += f" ({species} {round(confidence*100)}%)"
        print(printer)

        # is_new = species not in active_species
        # if is_new:
        #     print(f"[L2] >>> {species} sur {mac} (conf {det['confidence']:.2f})")
        #     if CONFIG.AFFICHAGE_IHM:
        #         notify_arrival(mac, species, det['confidence'])
        # else:
        #     print(f"[L2] === {species} toujours sur {mac} (conf {det['confidence']:.2f})")
        # active_species[species] = now
