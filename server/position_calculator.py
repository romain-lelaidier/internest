"""
Calculateur de positions 3D par triangulation TDOA.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List

from config import SPEED_OF_SOUND


class PositionCalculator:
    def __init__(self, macs: List[str], speed_of_sound: float = SPEED_OF_SOUND):
        self.macs = macs
        self.speed_of_sound = speed_of_sound
        self.positions: Dict[str, np.ndarray] = {}

    def compute_distance_matrix(
        self,
        all_tdoa: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """
        Construit la matrice des distances.
        all_tdoa[buzzer_mac][receiver_mac] = temps de détection
        """
        n = len(self.macs)
        mac_to_idx = {mac: i for i, mac in enumerate(self.macs)}

        time_matrix = np.zeros((n, n))

        for buzzer_mac, detections in all_tdoa.items():
            buzzer_idx = mac_to_idx.get(buzzer_mac)
            if buzzer_idx is None:
                continue

            ref_time = detections.get(buzzer_mac, 0)

            for receiver_mac, detect_time in detections.items():
                receiver_idx = mac_to_idx.get(receiver_mac)
                if receiver_idx is None:
                    continue

                propagation_time = detect_time - ref_time
                time_matrix[buzzer_idx, receiver_idx] = propagation_time

        return time_matrix * self.speed_of_sound

    def mds_positioning(self, distance_matrix: np.ndarray) -> np.ndarray:
        """MDS classique pour positions initiales."""
        n = distance_matrix.shape[0]

        D = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(D, 0)

        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J

        eigenvalues, eigenvectors = np.linalg.eigh(B)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues_3d = np.maximum(eigenvalues[:3], 0)
        eigenvectors_3d = eigenvectors[:, :3]

        return eigenvectors_3d * np.sqrt(eigenvalues_3d)

    def refine_positions(
        self,
        initial: np.ndarray,
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Affine par least-squares."""
        n = initial.shape[0]

        def residuals(flat):
            pos = flat.reshape(n, 3)
            errors = []
            for i in range(n):
                for j in range(n):
                    if i != j and distance_matrix[i, j] > 0:
                        computed = np.linalg.norm(pos[i] - pos[j])
                        errors.append(computed - distance_matrix[i, j])
            return errors

        result = least_squares(residuals, initial.flatten(), method='lm')
        return result.x.reshape(n, 3)

    def calculate_positions(
        self,
        all_tdoa: Dict[str, Dict[str, float]]
    ) -> Dict[str, np.ndarray]:
        """Calcule les positions 3D."""
        distance_matrix = self.compute_distance_matrix(all_tdoa)

        print("Matrice des distances (m):")
        print(distance_matrix.round(3))
        print()

        positions_mds = self.mds_positioning(distance_matrix)
        positions_refined = self.refine_positions(positions_mds, distance_matrix)

        centroid = positions_refined.mean(axis=0)
        positions_refined -= centroid

        self.positions = {mac: positions_refined[i] for i, mac in enumerate(self.macs)}
        return self.positions

    def print_positions(self):
        """Affiche les positions."""
        print("\n=== Positions (mètres) ===\n")

        for i, mac in enumerate(self.macs):
            if mac in self.positions:
                pos = self.positions[mac]
                print(f"ESP {i+1} ({mac}):")
                print(f"  X={pos[0]:+.3f}  Y={pos[1]:+.3f}  Z={pos[2]:+.3f}")

        print("\n=== Distances ===\n")
        for i, mac1 in enumerate(self.macs):
            for mac2 in self.macs[i+1:]:
                dist = np.linalg.norm(self.positions[mac1] - self.positions[mac2])
                j = self.macs.index(mac2)
                print(f"ESP {i+1} <-> ESP {j+1}: {dist:.3f} m")


if __name__ == "__main__":
    print("=== Test ===\n")

    test_macs = ["MAC1", "MAC2", "MAC3", "MAC4", "MAC5"]

    true_pos = {
        "MAC1": np.array([0.0, 0.0, 0.0]),
        "MAC2": np.array([1.0, 0.0, 0.0]),
        "MAC3": np.array([0.5, 0.866, 0.0]),
        "MAC4": np.array([0.5, 0.289, 0.816]),
        "MAC5": np.array([0.5, 0.433, 0.3]),
    }

    all_tdoa = {}
    for buzzer, bpos in true_pos.items():
        all_tdoa[buzzer] = {}
        for receiver, rpos in true_pos.items():
            dist = np.linalg.norm(bpos - rpos)
            t = dist / SPEED_OF_SOUND + np.random.randn() * 0.0001
            all_tdoa[buzzer][receiver] = t

    calc = PositionCalculator(test_macs)
    calc.calculate_positions(all_tdoa)
    calc.print_positions()
