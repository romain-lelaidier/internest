"""
Bird class for simulating bird vocalizations.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
from .config import SAMPLE_RATE, BIRD_FREQ_MIN, BIRD_FREQ_MAX


class CallType(Enum):
    """Types of bird vocalizations."""
    CHIRP = "chirp"           # Short, single frequency burst
    SONG = "song"             # Complex, multi-frequency pattern
    ALARM = "alarm"           # Sharp, repeated calls
    TRILL = "trill"           # Rapid frequency modulation


@dataclass
class BirdCall:
    """Represents a single bird vocalization event."""
    timestamp: float          # When the call starts (seconds)
    duration: float           # Duration in seconds
    call_type: CallType
    frequency_range: Tuple[float, float]  # (min_freq, max_freq) in Hz
    amplitude: float = 1.0    # Relative amplitude


@dataclass
class Bird:
    """
    Simulates a bird that can move and emit vocalizations.
    """
    id: int
    position: np.ndarray                    # Current (x, y, z) position in meters
    call_type: CallType = CallType.CHIRP
    frequency_range: Tuple[float, float] = (2000, 4000)  # Hz
    amplitude: float = 1.0
    is_moving: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.call_history: List[BirdCall] = []

    def move(self, dt: float):
        """Update position based on velocity and time step."""
        if self.is_moving:
            self.position = self.position + self.velocity * dt

    def set_position(self, position: np.ndarray):
        """Set bird position directly."""
        self.position = np.array(position, dtype=np.float64)

    def set_velocity(self, velocity: np.ndarray):
        """Set bird velocity for movement."""
        self.velocity = np.array(velocity, dtype=np.float64)
        self.is_moving = np.linalg.norm(velocity) > 0

    def emit_call(self, timestamp: float, duration: float = 0.1) -> BirdCall:
        """
        Emit a vocalization at the current timestamp.

        Args:
            timestamp: Time when the call starts
            duration: Duration of the call in seconds

        Returns:
            BirdCall object describing the vocalization
        """
        call = BirdCall(
            timestamp=timestamp,
            duration=duration,
            call_type=self.call_type,
            frequency_range=self.frequency_range,
            amplitude=self.amplitude
        )
        self.call_history.append(call)
        return call

    def generate_call_signal(self,
                            duration: float = 0.1,
                            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Generate the actual audio waveform for a bird call.

        Args:
            duration: Duration in seconds
            sample_rate: Samples per second

        Returns:
            Audio signal as numpy array
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)

        if self.call_type == CallType.CHIRP:
            signal = self._generate_chirp(t)
        elif self.call_type == CallType.SONG:
            signal = self._generate_song(t)
        elif self.call_type == CallType.ALARM:
            signal = self._generate_alarm(t)
        elif self.call_type == CallType.TRILL:
            signal = self._generate_trill(t)
        else:
            signal = self._generate_chirp(t)

        # Apply amplitude envelope (smooth onset/offset)
        envelope = self._create_envelope(n_samples)
        return self.amplitude * signal * envelope

    def _generate_chirp(self, t: np.ndarray) -> np.ndarray:
        """Generate a frequency-sweeping chirp."""
        f0, f1 = self.frequency_range
        # Linear chirp from f0 to f1
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1]))
        return np.sin(phase)

    def _generate_song(self, t: np.ndarray) -> np.ndarray:
        """Generate a complex multi-harmonic song."""
        f0 = self.frequency_range[0]
        f1 = self.frequency_range[1]
        fc = (f0 + f1) / 2

        # Multiple harmonics with frequency modulation
        fm = 10  # Modulation frequency
        mod_depth = (f1 - f0) / 4

        signal = np.zeros_like(t)
        for harmonic in [1, 2, 3]:
            freq = fc * harmonic + mod_depth * np.sin(2 * np.pi * fm * t)
            signal += (1 / harmonic) * np.sin(2 * np.pi * freq * t)

        return signal / np.max(np.abs(signal))

    def _generate_alarm(self, t: np.ndarray) -> np.ndarray:
        """Generate sharp, repeated alarm calls."""
        f0, f1 = self.frequency_range
        fc = (f0 + f1) / 2

        # Pulsed signal
        pulse_rate = 15  # pulses per second
        pulse_envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * pulse_rate * t)))

        signal = np.sin(2 * np.pi * fc * t) * pulse_envelope
        return signal

    def _generate_trill(self, t: np.ndarray) -> np.ndarray:
        """Generate rapid frequency modulation trill."""
        f0, f1 = self.frequency_range

        # Fast oscillation between frequencies
        trill_rate = 30  # Hz
        freq = f0 + (f1 - f0) * 0.5 * (1 + np.sin(2 * np.pi * trill_rate * t))

        # Integrate frequency to get phase
        dt = t[1] - t[0] if len(t) > 1 else 1 / SAMPLE_RATE
        phase = 2 * np.pi * np.cumsum(freq) * dt

        return np.sin(phase)

    def _create_envelope(self, n_samples: int, attack: float = 0.1, release: float = 0.1) -> np.ndarray:
        """Create smooth amplitude envelope to avoid clicks."""
        envelope = np.ones(n_samples)

        attack_samples = int(n_samples * attack)
        release_samples = int(n_samples * release)

        # Attack (fade in)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Release (fade out)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)

        return envelope

    def __repr__(self):
        return f"Bird(id={self.id}, pos={self.position}, type={self.call_type.value})"


class BirdFlock:
    """Collection of birds for multi-source scenarios."""

    def __init__(self):
        self.birds: List[Bird] = []

    def add_bird(self, bird: Bird):
        """Add a bird to the flock."""
        self.birds.append(bird)

    def update(self, dt: float):
        """Update all bird positions."""
        for bird in self.birds:
            bird.move(dt)

    def get_bird(self, bird_id: int) -> Optional[Bird]:
        """Get bird by ID."""
        for bird in self.birds:
            if bird.id == bird_id:
                return bird
        return None

    @classmethod
    def create_random_flock(cls,
                            num_birds: int,
                            area_bounds: Tuple[Tuple[float, float], ...],
                            seed: int = None) -> 'BirdFlock':
        """
        Create a flock with random positions.

        Args:
            num_birds: Number of birds
            area_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        flock = cls()
        call_types = list(CallType)

        for i in range(num_birds):
            position = np.array([
                np.random.uniform(*area_bounds[0]),
                np.random.uniform(*area_bounds[1]),
                np.random.uniform(*area_bounds[2])
            ])

            call_type = np.random.choice(call_types)
            freq_center = np.random.uniform(BIRD_FREQ_MIN + 500, BIRD_FREQ_MAX - 500)
            freq_range = (freq_center - 500, freq_center + 500)

            bird = Bird(
                id=i,
                position=position,
                call_type=call_type,
                frequency_range=freq_range,
                amplitude=np.random.uniform(0.7, 1.0)
            )
            flock.add_bird(bird)

        return flock
