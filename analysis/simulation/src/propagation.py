"""
Sound propagation simulator.
Simulates how bird calls travel through space to reach microphones.
"""
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .config import (
    SAMPLE_RATE, SPEED_OF_SOUND, SYNC_PRECISION_S,
    AMBIENT_NOISE_LEVEL
)
from .microphone import MicrophoneArray, Microphone
from .bird import Bird, BirdCall


@dataclass
class ReceivedSignal:
    """Signal received at a microphone."""
    mic_id: int
    signal: np.ndarray
    arrival_time: float      # Time when signal first arrives
    sample_offset: int       # Sample index where signal starts
    snr: float              # Signal-to-noise ratio


class SoundPropagator:
    """
    Simulates sound propagation from sources to microphones.

    Includes:
    - Distance-based attenuation (inverse square law)
    - Time delay based on distance
    - Ambient noise addition
    - Synchronization jitter
    """

    def __init__(self,
                 mic_array: MicrophoneArray,
                 sample_rate: int = SAMPLE_RATE,
                 noise_level: float = AMBIENT_NOISE_LEVEL,
                 sync_jitter: float = SYNC_PRECISION_S):
        """
        Initialize the sound propagator.

        Args:
            mic_array: Array of microphones
            sample_rate: Audio sample rate
            noise_level: Relative amplitude of ambient noise
            sync_jitter: Synchronization uncertainty in seconds
        """
        self.mic_array = mic_array
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.sync_jitter = sync_jitter

    def calculate_attenuation(self, distance: float, reference_distance: float = 1.0) -> float:
        """
        Calculate amplitude attenuation using inverse square law.

        Args:
            distance: Distance from source in meters
            reference_distance: Reference distance for unit amplitude

        Returns:
            Attenuation factor (0 to 1)
        """
        if distance < reference_distance:
            return 1.0
        return reference_distance / distance

    def calculate_delay_samples(self, distance: float) -> int:
        """
        Calculate delay in samples for a given distance.

        Args:
            distance: Distance in meters

        Returns:
            Number of samples delay
        """
        time_delay = distance / SPEED_OF_SOUND
        return int(time_delay * self.sample_rate)

    def generate_ambient_noise(self, n_samples: int) -> np.ndarray:
        """
        Generate ambient forest/environment noise.

        Uses filtered white noise to simulate natural background sounds.
        """
        # Pink noise approximation (1/f spectrum, more natural)
        white = np.random.randn(n_samples)

        # Simple low-pass filter for more natural sound
        # Using cumulative sum and differentiation for pink-ish noise
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.std(pink) + 1e-10)

        return self.noise_level * pink

    def propagate_call(self,
                       bird: Bird,
                       call_signal: np.ndarray,
                       emission_time: float,
                       total_duration: float) -> Dict[int, ReceivedSignal]:
        """
        Propagate a bird call to all microphones.

        Args:
            bird: Bird that emitted the call
            call_signal: The audio waveform of the call
            emission_time: Time when the bird emitted the call
            total_duration: Total simulation duration in seconds

        Returns:
            Dictionary mapping mic_id to ReceivedSignal
        """
        total_samples = int(total_duration * self.sample_rate)
        received_signals = {}

        for mic in self.mic_array.microphones:
            # Calculate propagation parameters
            distance = mic.distance_to(bird.position)
            attenuation = self.calculate_attenuation(distance)
            delay_samples = self.calculate_delay_samples(distance)

            # Add synchronization jitter
            jitter_samples = int(np.random.uniform(-1, 1) * self.sync_jitter * self.sample_rate)
            total_delay = delay_samples + jitter_samples

            # Calculate arrival time
            arrival_time = emission_time + distance / SPEED_OF_SOUND

            # Create output buffer with noise
            output = self.generate_ambient_noise(total_samples)

            # Calculate where to insert the signal
            emission_sample = int(emission_time * self.sample_rate)
            insert_start = emission_sample + total_delay

            # Insert attenuated signal if it fits in the buffer
            if 0 <= insert_start < total_samples:
                insert_end = min(insert_start + len(call_signal), total_samples)
                signal_end = insert_end - insert_start
                output[insert_start:insert_end] += attenuation * call_signal[:signal_end]

            # Calculate SNR
            signal_power = np.var(attenuation * call_signal)
            noise_power = np.var(output) - signal_power
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

            received_signals[mic.id] = ReceivedSignal(
                mic_id=mic.id,
                signal=output,
                arrival_time=arrival_time,
                sample_offset=insert_start,
                snr=snr
            )

        return received_signals


class SimulationEnvironment:
    """
    Complete simulation environment for bird localization testing.
    """

    def __init__(self,
                 mic_array: MicrophoneArray,
                 sample_rate: int = SAMPLE_RATE,
                 noise_level: float = AMBIENT_NOISE_LEVEL):
        """
        Initialize simulation environment.

        Args:
            mic_array: Microphone array for recording
            sample_rate: Audio sample rate
            noise_level: Background noise level
        """
        self.mic_array = mic_array
        self.sample_rate = sample_rate
        self.propagator = SoundPropagator(mic_array, sample_rate, noise_level)
        self.birds: List[Bird] = []
        self.time = 0.0

    def add_bird(self, bird: Bird):
        """Add a bird to the environment."""
        self.birds.append(bird)

    def simulate_call(self,
                      bird: Bird,
                      call_duration: float = 0.1,
                      total_duration: float = 1.0) -> Dict[int, ReceivedSignal]:
        """
        Simulate a single bird call and its reception at all microphones.

        Args:
            bird: Bird emitting the call
            call_duration: Duration of the call
            total_duration: Total recording duration

        Returns:
            Received signals at each microphone
        """
        # Generate the call signal
        call_signal = bird.generate_call_signal(call_duration, self.sample_rate)

        # Emit the call (record in bird's history)
        emission_time = total_duration * 0.2  # Start call at 20% of total duration
        bird.emit_call(emission_time, call_duration)

        # Propagate to all microphones
        return self.propagator.propagate_call(
            bird, call_signal, emission_time, total_duration
        )

    def simulate_multiple_calls(self,
                                bird: Bird,
                                call_times: List[float],
                                call_duration: float = 0.1,
                                total_duration: float = 5.0) -> Dict[int, np.ndarray]:
        """
        Simulate multiple calls from a bird at specified times.

        Args:
            bird: Bird emitting calls
            call_times: List of emission times
            call_duration: Duration of each call
            total_duration: Total simulation duration

        Returns:
            Combined signals at each microphone
        """
        total_samples = int(total_duration * self.sample_rate)

        # Initialize with noise
        combined_signals = {
            mic.id: self.propagator.generate_ambient_noise(total_samples)
            for mic in self.mic_array.microphones
        }

        for emission_time in call_times:
            call_signal = bird.generate_call_signal(call_duration, self.sample_rate)
            bird.emit_call(emission_time, call_duration)

            for mic in self.mic_array.microphones:
                distance = mic.distance_to(bird.position)
                attenuation = self.propagator.calculate_attenuation(distance)
                delay_samples = self.propagator.calculate_delay_samples(distance)

                emission_sample = int(emission_time * self.sample_rate)
                insert_start = emission_sample + delay_samples

                if 0 <= insert_start < total_samples:
                    insert_end = min(insert_start + len(call_signal), total_samples)
                    signal_end = insert_end - insert_start
                    combined_signals[mic.id][insert_start:insert_end] += (
                        attenuation * call_signal[:signal_end]
                    )

        return combined_signals

    def get_theoretical_tdoa(self, bird: Bird) -> Dict[Tuple[int, int], float]:
        """
        Calculate theoretical TDOA between all microphone pairs.

        Args:
            bird: Bird position to calculate TDOA for

        Returns:
            Dictionary mapping (mic_i, mic_j) to TDOA in seconds
        """
        tdoa = {}
        mics = self.mic_array.microphones

        for i, mic_i in enumerate(mics):
            for j, mic_j in enumerate(mics):
                if i < j:
                    dist_i = mic_i.distance_to(bird.position)
                    dist_j = mic_j.distance_to(bird.position)
                    time_diff = (dist_i - dist_j) / SPEED_OF_SOUND
                    tdoa[(mic_i.id, mic_j.id)] = time_diff

        return tdoa
