"""
Gestion des trajectoires avec lissage.
"""

import numpy as np
from scipy.signal import savgol_filter

from config import Config


class TrackManager:
    """Gestionnaire de trajectoires avec lissage exponentiel."""

    def __init__(self):
        self.tracks = {}  # {bird_id: [[x, y, z, t], ...]}
        self.smoothed_last_pos = {}  # {bird_id: [x, y, z]}

    def add_point(self, pos: np.ndarray, t: float, bird_id: int):
        """
        Ajoute un point à la trajectoire d'un oiseau avec lissage EMA.

        Args:
            pos: Position (x, y, z)
            t: Temps
            bird_id: Identifiant de l'oiseau
        """
        if bird_id not in self.tracks:
            self.tracks[bird_id] = []
            self.smoothed_last_pos[bird_id] = pos

        # Lissage exponentiel
        prev = self.smoothed_last_pos[bird_id]
        smooth_pos = Config.ALPHA * pos + (1 - Config.ALPHA) * prev

        self.tracks[bird_id].append(np.append(smooth_pos, t))
        self.smoothed_last_pos[bird_id] = smooth_pos

    def get_smoothed_trajectories(self) -> dict:
        """
        Applique le lissage Savitzky-Golay sur les trajectoires.

        Returns:
            dict: {bird_id: array (N x 4) avec [x, y, z, t]}
        """
        smoothed = {}

        for bird_id, points in self.tracks.items():
            if len(points) < 3:
                smoothed[bird_id] = np.array(points)
                continue

            data = np.array(points)
            data = data[data[:, 3].argsort()]  # Tri temporel

            x, y, z = data[:, 0], data[:, 1], data[:, 2]

            if len(x) > 5:
                win = min(7, len(x))
                if win % 2 == 0:
                    win -= 1
                x = savgol_filter(x, win, 2)
                y = savgol_filter(y, win, 2)
                z = savgol_filter(z, win, 2)

            smoothed[bird_id] = np.column_stack([x, y, z, data[:, 3]])

        return smoothed
