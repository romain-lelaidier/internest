"""
Bird call detection using time-frequency analysis.

This module uses spectrogram-based detection which is well-suited for bird
vocalizations because:
1. Time-frequency localization: Can detect when AND at what frequency events occur
2. Multi-scale analysis: Bird calls have varying durations and frequencies
3. Robust to noise: Better at detecting transients in noisy environments
4. No assumption of periodicity: Works for irregular/aperiodic calls

Note: We use STFT-based spectrogram instead of CWT for better compatibility
with modern scipy versions, while achieving similar detection performance.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import signal as sig
from scipy.ndimage import maximum_filter

from .config import (
    SAMPLE_RATE, BIRD_FREQ_MIN, BIRD_FREQ_MAX,
    CWT_WAVELET, DETECTION_THRESHOLD, MIN_EVENT_DURATION_MS
)


@dataclass
class DetectedEvent:
    """Represents a detected bird call event."""
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    peak_frequency: float
    confidence: float
    energy: float


class CWTDetector:
    """
    Time-frequency based bird call detector.

    Uses spectrogram analysis which is well-suited for audio analysis
    due to its good time-frequency resolution.
    """

    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 freq_min: float = BIRD_FREQ_MIN,
                 freq_max: float = BIRD_FREQ_MAX,
                 threshold: float = DETECTION_THRESHOLD):
        """
        Initialize the detector.

        Args:
            sample_rate: Audio sample rate
            freq_min: Minimum frequency of interest (Hz)
            freq_max: Maximum frequency of interest (Hz)
            threshold: Detection threshold (0-1)
        """
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.threshold = threshold

        # STFT parameters optimized for bird call detection
        self.nperseg = 1024  # Window size
        self.noverlap = 768  # 75% overlap for smooth time resolution
        self.nfft = 2048     # FFT size for frequency resolution

    def compute_spectrogram(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram (time-frequency representation) using STFT.

        Args:
            signal_data: Input audio signal

        Returns:
            Tuple of (spectrogram, frequencies, times)
        """
        frequencies, times, Sxx = sig.spectrogram(
            signal_data,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            window='hann',
            scaling='spectrum'
        )

        # Filter to frequency range of interest
        freq_mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]

        return Sxx, frequencies, times

    def compute_scalogram(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute scalogram (time-frequency representation).
        This is an alias for compute_spectrogram for API compatibility.

        Args:
            signal_data: Input audio signal

        Returns:
            Tuple of (scalogram, times, frequencies)
        """
        Sxx, frequencies, times = self.compute_spectrogram(signal_data)
        return Sxx, times, frequencies

    def detect_events(self, signal_data: np.ndarray) -> List[DetectedEvent]:
        """
        Detect bird call events in the signal using spectrogram analysis.

        Algorithm:
        1. Compute spectrogram
        2. Sum energy across frequency bands of interest
        3. Apply threshold to find high-energy regions
        4. Group consecutive time frames into events
        5. Filter by minimum duration

        Args:
            signal_data: Input audio signal

        Returns:
            List of detected events
        """
        # Compute spectrogram
        scalogram, frequencies, times = self.compute_spectrogram(signal_data)

        # Sum energy across all frequency bands
        energy_envelope = np.sum(scalogram, axis=0)

        # Normalize energy
        energy_envelope = energy_envelope / (np.max(energy_envelope) + 1e-10)

        # Find regions above threshold
        above_threshold = energy_envelope > self.threshold

        # Find contiguous regions
        events = []
        in_event = False
        event_start_idx = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                # Start of event
                event_start_idx = i
                in_event = True
            elif not is_above and in_event:
                # End of event
                event_end_idx = i
                in_event = False

                # Convert spectrogram frame indices to time/samples
                start_time = times[event_start_idx]
                end_time = times[event_end_idx - 1] if event_end_idx > 0 else times[0]
                duration_ms = (end_time - start_time) * 1000

                if duration_ms >= MIN_EVENT_DURATION_MS:
                    # Find peak frequency in this region
                    event_scalogram = scalogram[:, event_start_idx:event_end_idx]
                    freq_energies = np.sum(event_scalogram, axis=1)
                    peak_freq_idx = np.argmax(freq_energies)
                    peak_freq = frequencies[peak_freq_idx]

                    # Calculate confidence and energy
                    event_energy = np.sum(energy_envelope[event_start_idx:event_end_idx])
                    confidence = np.mean(energy_envelope[event_start_idx:event_end_idx])

                    # Convert time to samples
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)

                    events.append(DetectedEvent(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        start_time=start_time,
                        end_time=end_time,
                        peak_frequency=peak_freq,
                        confidence=confidence,
                        energy=event_energy
                    ))

        # Handle event that extends to end of signal
        if in_event:
            event_end_idx = len(times)
            start_time = times[event_start_idx]
            end_time = times[-1]
            duration_ms = (end_time - start_time) * 1000

            if duration_ms >= MIN_EVENT_DURATION_MS:
                event_scalogram = scalogram[:, event_start_idx:event_end_idx]
                freq_energies = np.sum(event_scalogram, axis=1)
                peak_freq_idx = np.argmax(freq_energies)
                peak_freq = frequencies[peak_freq_idx]

                event_energy = np.sum(energy_envelope[event_start_idx:event_end_idx])
                confidence = np.mean(energy_envelope[event_start_idx:event_end_idx])

                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)

                events.append(DetectedEvent(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time=start_time,
                    end_time=end_time,
                    peak_frequency=peak_freq,
                    confidence=confidence,
                    energy=event_energy
                ))

        return events

    def get_event_signal(self,
                        signal_data: np.ndarray,
                        event: DetectedEvent,
                        padding_ms: float = 10) -> np.ndarray:
        """
        Extract signal segment for a detected event with padding.

        Args:
            signal_data: Full signal
            event: Detected event
            padding_ms: Padding before/after event in milliseconds

        Returns:
            Signal segment containing the event
        """
        padding_samples = int(padding_ms * self.sample_rate / 1000)

        start = max(0, event.start_sample - padding_samples)
        end = min(len(signal_data), event.end_sample + padding_samples)

        return signal_data[start:end]


class MultiChannelDetector:
    """
    Detector that processes multiple microphone channels simultaneously.
    """

    def __init__(self,
                 num_channels: int,
                 sample_rate: int = SAMPLE_RATE,
                 **kwargs):
        """
        Initialize multi-channel detector.

        Args:
            num_channels: Number of microphone channels
            sample_rate: Audio sample rate
            **kwargs: Additional arguments for CWTDetector
        """
        self.num_channels = num_channels
        self.detector = CWTDetector(sample_rate=sample_rate, **kwargs)

    def detect_synchronized_events(self,
                                   signals: Dict[int, np.ndarray],
                                   min_channels: int = 3) -> List[Dict]:
        """
        Detect events that appear in multiple channels (likely real bird calls).

        Args:
            signals: Dictionary mapping channel ID to signal
            min_channels: Minimum channels that must detect event

        Returns:
            List of multi-channel event detections
        """
        # Detect events in each channel
        channel_events = {}
        for ch_id, signal_data in signals.items():
            channel_events[ch_id] = self.detector.detect_events(signal_data)

        # Find overlapping events across channels
        # Use a simple time-window approach
        all_events = []
        time_tolerance = 0.5  # seconds - max expected TDOA

        # Collect all event start times
        event_times = []
        for ch_id, events in channel_events.items():
            for event in events:
                event_times.append((event.start_time, ch_id, event))

        event_times.sort(key=lambda x: x[0])

        # Group events by time proximity
        if not event_times:
            return []

        groups = []
        current_group = [event_times[0]]

        for i in range(1, len(event_times)):
            time_diff = event_times[i][0] - current_group[-1][0]

            if time_diff <= time_tolerance:
                current_group.append(event_times[i])
            else:
                if len(set(e[1] for e in current_group)) >= min_channels:
                    groups.append(current_group)
                current_group = [event_times[i]]

        # Don't forget last group
        if len(set(e[1] for e in current_group)) >= min_channels:
            groups.append(current_group)

        # Convert groups to structured output
        for group in groups:
            channels_detected = {}
            for time, ch_id, event in group:
                channels_detected[ch_id] = event

            all_events.append({
                'channels': channels_detected,
                'num_channels': len(channels_detected),
                'mean_start_time': np.mean([e.start_time for _, _, e in group]),
                'mean_confidence': np.mean([e.confidence for _, _, e in group])
            })

        return all_events
