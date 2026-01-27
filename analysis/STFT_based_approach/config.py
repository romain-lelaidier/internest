"""
Configuration globale pour la reconstruction de trajectoires.
"""

import numpy as np


class Config:
    # Audio
    FS = 44100
    C = 343.0  # Vitesse du son (m/s)

    # STFT
    N_FFT = 2048
    HOP_LENGTH = 512
    DB_THRESHOLD = 30

    # Détection YOLO
    MODEL_PATH = "./best.pt"
    CONF_THRESHOLD = 0.25

    # Clustering
    EPS_FREQ = 500  # Hz - epsilon pour DBSCAN sur fréquences
    MIN_SAMPLES = 2

    # TDOA
    PADDING = 0.08  # Marge temporelle (s) autour des détections
    TIME_TOL = 0.25  # Tolérance pour regrouper les événements (s)
    MIN_MICS = 4    # Minimum de micros pour triangulation
    COST_THRESHOLD = 50.0  # Seuil d'erreur géométrique (plus permissif)
    MIN_TDOA_SCORE = 0.02  # Score minimum pour accepter un TDOA

    # Tracking
    ALPHA = 0.4  # Facteur de lissage EMA (plus de poids à l'historique)

    # Filtrage trajectoires (pour validation par sous-ensembles)
    V_MAX = 25.0  # Vitesse max d'un oiseau (m/s)
    CONFIDENCE_MIN = 0.5  # Confiance minimum pour accepter un point
    STD_MAX = 10.0  # Écart-type max entre sous-ensembles (m)

    # Géométrie - Cube 100m x 100m x 40m
    MICROS = np.array([
        [0.0, 0.0, 0.0],       # Mic 0
        [100.0, 0.0, 0.0],     # Mic 1
        [0.0, 100.0, 0.0],     # Mic 2
        [100.0, 100.0, 0.0],   # Mic 3
        [0.0, 0.0, 40.0],      # Mic 4
        [100.0, 0.0, 40.0],    # Mic 5
        [0.0, 100.0, 40.0],    # Mic 6
        [100.0, 100.0, 40.0]   # Mic 7
    ])

    # Couleurs pour visualisation
    COLORS = {
        0: (0, 255, 0),    # Vert
        1: (255, 0, 0),    # Bleu
        2: (0, 0, 255),    # Rouge
        3: (255, 255, 0),  # Cyan
    }
