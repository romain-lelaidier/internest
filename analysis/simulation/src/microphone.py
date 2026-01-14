"""
Microphone class and optimal placement strategies for 3D localization.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .config import SAMPLE_RATE, SPEED_OF_SOUND


@dataclass
class Microphone:
    """Represents a microphone sensor in 3D space."""
    id: int
    position: np.ndarray  # (x, y, z) in meters

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.buffer: Optional[np.ndarray] = None
        self.timestamps: List[float] = []

    def distance_to(self, point: np.ndarray) -> float:
        """Calculate Euclidean distance to a point."""
        return np.linalg.norm(self.position - point)

    def time_of_arrival(self, source_position: np.ndarray, emission_time: float) -> float:
        """Calculate when sound from source reaches this microphone."""
        distance = self.distance_to(source_position)
        travel_time = distance / SPEED_OF_SOUND
        return emission_time + travel_time

    def clear_buffer(self):
        """Clear the audio buffer."""
        self.buffer = None
        self.timestamps = []


class MicrophoneArray:
    """
    Array of microphones for 3D sound source localization.

    For TDOA-based localization, microphones should be:
    - Not coplanar (for 3D localization)
    - Not collinear (for 2D localization)
    - Spread out to maximize coverage
    """

    def __init__(self, microphones: List[Microphone] = None):
        self.microphones = microphones or []

    def add_microphone(self, mic: Microphone):
        """Add a microphone to the array."""
        self.microphones.append(mic)

    @property
    def num_microphones(self) -> int:
        return len(self.microphones)

    @property
    def positions(self) -> np.ndarray:
        """Return all microphone positions as Nx3 array."""
        return np.array([m.position for m in self.microphones])

    def get_microphone(self, mic_id: int) -> Optional[Microphone]:
        """Get microphone by ID."""
        for mic in self.microphones:
            if mic.id == mic_id:
                return mic
        return None

    def centroid(self) -> np.ndarray:
        """Return the centroid of all microphone positions."""
        return np.mean(self.positions, axis=0)

    def is_coplanar(self, tolerance: float = 1e-6) -> bool:
        """
        Check if all microphones are coplanar.
        For 3D localization, they should NOT be coplanar.
        """
        if len(self.microphones) < 4:
            return True  # 3 points are always coplanar

        positions = self.positions
        # Use SVD to check dimensionality
        centered = positions - positions.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)

        # If smallest singular value is ~0, points are coplanar
        return s[-1] < tolerance

    @classmethod
    def create_optimal_array(cls,
                             num_mics: int = 6,
                             radius: float = 50.0,
                             height_variation: float = 15.0,
                             center: np.ndarray = None) -> 'MicrophoneArray':
        """
        Create an optimal microphone array for 3D localization.

        Strategy: Two staggered triangles at different heights,
        providing excellent 3D coverage and non-coplanarity.

        Args:
            num_mics: Number of microphones (default 6)
            radius: Horizontal spread radius in meters
            height_variation: Vertical spread in meters
            center: Center point of the array

        Returns:
            MicrophoneArray with optimal placement
        """
        if center is None:
            center = np.array([0.0, 0.0, 0.0])

        array = cls()

        if num_mics == 6:
            # Optimal 6-mic configuration:
            # - 3 mics in lower triangle (z = 2m, ground-mounted on poles)
            # - 3 mics in upper triangle, rotated 60° (z = 12-17m, tree-mounted)

            # Lower triangle
            for i in range(3):
                angle = 2 * np.pi * i / 3
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 2.0 + np.random.uniform(0, 1)  # Ground level with small variation
                array.add_microphone(Microphone(id=i, position=np.array([x, y, z])))

            # Upper triangle (rotated 60° = π/3)
            for i in range(3):
                angle = 2 * np.pi * i / 3 + np.pi / 3  # 60° offset
                x = center[0] + radius * 0.7 * np.cos(angle)  # Slightly smaller radius
                y = center[1] + radius * 0.7 * np.sin(angle)
                z = 10.0 + height_variation * (0.5 + 0.5 * np.sin(angle))  # 10-17m
                array.add_microphone(Microphone(id=i + 3, position=np.array([x, y, z])))
        else:
            # Generic placement: distribute on a sphere-like pattern
            for i in range(num_mics):
                # Golden angle distribution
                golden_angle = np.pi * (3 - np.sqrt(5))
                theta = golden_angle * i
                phi = np.arccos(1 - 2 * (i + 0.5) / num_mics)

                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + height_variation * np.cos(phi)

                # Ensure z is positive (above ground)
                z = max(1.0, z + height_variation / 2)

                array.add_microphone(Microphone(id=i, position=np.array([x, y, z])))

        return array

    def get_pair_distances(self) -> dict:
        """Get distances between all microphone pairs."""
        distances = {}
        n = len(self.microphones)
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.microphones[i].distance_to(self.microphones[j].position)
                distances[(i, j)] = dist
        return distances

    def coverage_volume(self) -> dict:
        """Estimate the coverage volume of the array."""
        positions = self.positions
        return {
            'x_range': (positions[:, 0].min(), positions[:, 0].max()),
            'y_range': (positions[:, 1].min(), positions[:, 1].max()),
            'z_range': (positions[:, 2].min(), positions[:, 2].max()),
            'centroid': self.centroid()
        }

    def __repr__(self):
        return f"MicrophoneArray(n={self.num_microphones}, coplanar={self.is_coplanar()})"
