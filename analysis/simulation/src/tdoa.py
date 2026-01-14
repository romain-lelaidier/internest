"""
Time Difference of Arrival (TDOA) estimation using cross-correlation.

TDOA is the core technique for sound source localization.
By measuring the time difference between signal arrivals at different
microphones, we can triangulate the source position.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import scipy.signal as signal
from scipy.fft import fft, ifft

from .config import SAMPLE_RATE, SPEED_OF_SOUND
from .microphone import MicrophoneArray


@dataclass
class TDOAEstimate:
    """TDOA estimate between a pair of microphones."""
    mic_i: int
    mic_j: int
    tdoa_seconds: float      # Time difference in seconds (positive = i before j)
    tdoa_samples: int        # Time difference in samples
    confidence: float        # Correlation peak value (0-1)
    distance_diff: float     # Equivalent distance difference in meters


class TDOAEstimator:
    """
    Estimates Time Difference of Arrival between microphone pairs.

    Uses Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    which is robust to noise and reverberation.
    """

    def __init__(self,
                 mic_array: MicrophoneArray,
                 sample_rate: int = SAMPLE_RATE,
                 max_tdoa: float = None):
        """
        Initialize TDOA estimator.

        Args:
            mic_array: Microphone array configuration
            sample_rate: Audio sample rate
            max_tdoa: Maximum expected TDOA in seconds (auto-calculated if None)
        """
        self.mic_array = mic_array
        self.sample_rate = sample_rate

        # Calculate max TDOA based on array geometry
        if max_tdoa is None:
            max_distance = self._calculate_max_mic_distance()
            self.max_tdoa = max_distance / SPEED_OF_SOUND
        else:
            self.max_tdoa = max_tdoa

        self.max_lag_samples = int(self.max_tdoa * sample_rate)

    def _calculate_max_mic_distance(self) -> float:
        """Calculate maximum distance between any two microphones."""
        max_dist = 0
        positions = self.mic_array.positions
        n = len(positions)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                max_dist = max(max_dist, dist)

        return max_dist

    def cross_correlation(self,
                         signal_i: np.ndarray,
                         signal_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute standard cross-correlation between two signals.

        Args:
            signal_i: Signal from microphone i
            signal_j: Signal from microphone j

        Returns:
            Tuple of (correlation, lags in samples)
        """
        correlation = signal.correlate(signal_i, signal_j, mode='full')
        lags = signal.correlation_lags(len(signal_i), len(signal_j), mode='full')

        return correlation, lags

    def gcc_phat(self,
                signal_i: np.ndarray,
                signal_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

        GCC-PHAT whitens the spectrum, making it more robust to:
        - Reverberations
        - Background noise
        - Signal coloration

        Args:
            signal_i: Signal from microphone i
            signal_j: Signal from microphone j

        Returns:
            Tuple of (gcc_phat, lags in samples)
        """
        n = len(signal_i) + len(signal_j) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))  # Next power of 2

        # FFT of both signals
        X_i = fft(signal_i, n_fft)
        X_j = fft(signal_j, n_fft)

        # Cross-power spectrum
        cross_spectrum = X_i * np.conj(X_j)

        # Phase transform (whitening)
        magnitude = np.abs(cross_spectrum)
        magnitude[magnitude < 1e-10] = 1e-10  # Avoid division by zero
        phat = cross_spectrum / magnitude

        # Inverse FFT to get GCC-PHAT
        gcc = np.real(ifft(phat))

        # Shift to center zero-lag
        gcc = np.fft.fftshift(gcc)
        lags = np.arange(-n_fft // 2, n_fft // 2)

        return gcc, lags

    def estimate_tdoa_pair(self,
                          signal_i: np.ndarray,
                          signal_j: np.ndarray,
                          mic_i: int,
                          mic_j: int,
                          method: str = 'gcc_phat') -> TDOAEstimate:
        """
        Estimate TDOA between a pair of microphones.

        Args:
            signal_i: Signal from microphone i
            signal_j: Signal from microphone j
            mic_i: ID of microphone i
            mic_j: ID of microphone j
            method: 'gcc_phat' or 'cross_correlation'

        Returns:
            TDOAEstimate object
        """
        if method == 'gcc_phat':
            corr, lags = self.gcc_phat(signal_i, signal_j)
        else:
            corr, lags = self.cross_correlation(signal_i, signal_j)

        # Limit search to valid TDOA range
        valid_mask = np.abs(lags) <= self.max_lag_samples
        corr_valid = corr[valid_mask]
        lags_valid = lags[valid_mask]

        # Find peak
        peak_idx = np.argmax(np.abs(corr_valid))
        peak_lag = lags_valid[peak_idx]
        peak_value = np.abs(corr_valid[peak_idx])

        # Calculate confidence based on peak prominence
        # A good correlation has a sharp, distinct peak
        corr_abs = np.abs(corr_valid)
        peak_height = corr_abs[peak_idx]

        # Compute mean and std excluding the peak region
        peak_region = 50  # samples around peak to exclude
        mask = np.ones(len(corr_abs), dtype=bool)
        start_exclude = max(0, peak_idx - peak_region)
        end_exclude = min(len(corr_abs), peak_idx + peak_region)
        mask[start_exclude:end_exclude] = False

        if np.sum(mask) > 0:
            background_mean = np.mean(corr_abs[mask])
            background_std = np.std(corr_abs[mask])
            # Confidence based on how many std deviations the peak is above background
            if background_std > 0:
                prominence = (peak_height - background_mean) / background_std
                confidence = min(1.0, prominence / 10.0)  # Normalize to 0-1
            else:
                confidence = 1.0 if peak_height > background_mean else 0.0
        else:
            confidence = 0.5

        # Subsample refinement using parabolic interpolation
        if 0 < peak_idx < len(corr_valid) - 1:
            alpha = corr_valid[peak_idx - 1]
            beta = corr_valid[peak_idx]
            gamma = corr_valid[peak_idx + 1]

            # Parabolic interpolation
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-10:
                delta = 0.5 * (alpha - gamma) / denom
                refined_lag = peak_lag + delta
            else:
                refined_lag = float(peak_lag)
        else:
            refined_lag = float(peak_lag)

        tdoa_seconds = refined_lag / self.sample_rate
        distance_diff = tdoa_seconds * SPEED_OF_SOUND

        return TDOAEstimate(
            mic_i=mic_i,
            mic_j=mic_j,
            tdoa_seconds=tdoa_seconds,
            tdoa_samples=int(refined_lag),
            confidence=max(0.0, min(1.0, confidence)),
            distance_diff=distance_diff
        )

    def estimate_all_pairs(self,
                          signals: Dict[int, np.ndarray],
                          method: str = 'gcc_phat') -> List[TDOAEstimate]:
        """
        Estimate TDOA for all microphone pairs.

        Args:
            signals: Dictionary mapping mic_id to signal
            method: Correlation method

        Returns:
            List of TDOA estimates for all pairs
        """
        estimates = []
        mic_ids = sorted(signals.keys())

        for i, mic_i in enumerate(mic_ids):
            for mic_j in mic_ids[i + 1:]:
                estimate = self.estimate_tdoa_pair(
                    signals[mic_i],
                    signals[mic_j],
                    mic_i,
                    mic_j,
                    method
                )
                estimates.append(estimate)

        return estimates

    def filter_by_confidence(self,
                            estimates: List[TDOAEstimate],
                            min_confidence: float = 0.5) -> List[TDOAEstimate]:
        """Filter TDOA estimates by confidence threshold."""
        return [e for e in estimates if e.confidence >= min_confidence]

    def filter_physically_valid(self,
                               estimates: List[TDOAEstimate],
                               margin: float = 0.1,
                               min_tdoa_ratio: float = 0.05) -> List[TDOAEstimate]:
        """
        Filter TDOA estimates that are physically impossible or suspicious.

        A TDOA between two microphones cannot exceed the distance between them
        divided by the speed of sound. Also rejects near-zero TDOAs for distant mics.

        Args:
            estimates: List of TDOA estimates
            margin: Margin of error (0.1 = 10% tolerance)
            min_tdoa_ratio: Minimum TDOA as fraction of max possible (reject if smaller)

        Returns:
            List of physically valid estimates
        """
        valid = []
        for est in estimates:
            mic_i = self.mic_array.get_microphone(est.mic_i)
            mic_j = self.mic_array.get_microphone(est.mic_j)

            if mic_i is None or mic_j is None:
                continue

            # Maximum possible TDOA based on microphone distance
            mic_distance = np.linalg.norm(mic_i.position - mic_j.position)
            max_possible_tdoa = mic_distance / SPEED_OF_SOUND

            # Check if measured TDOA is physically possible
            if abs(est.tdoa_seconds) > max_possible_tdoa * (1 + margin):
                continue

            # Suspicious: TDOA near zero for distant microphones
            # This often indicates noise correlation rather than signal
            min_expected_tdoa = max_possible_tdoa * min_tdoa_ratio
            if abs(est.tdoa_seconds) < min_expected_tdoa and mic_distance > 20:
                # Near-zero TDOA for distant mics is suspicious - reduce confidence
                est_copy = TDOAEstimate(
                    mic_i=est.mic_i,
                    mic_j=est.mic_j,
                    tdoa_seconds=est.tdoa_seconds,
                    tdoa_samples=est.tdoa_samples,
                    confidence=est.confidence * 0.3,  # Reduce confidence significantly
                    distance_diff=est.distance_diff
                )
                valid.append(est_copy)
            else:
                valid.append(est)

        return valid

    def get_validated_estimates(self,
                               signals: Dict[int, np.ndarray],
                               method: str = 'gcc_phat') -> List[TDOAEstimate]:
        """
        Get TDOA estimates with physical validation.

        Args:
            signals: Dictionary mapping mic_id to signal
            method: Correlation method

        Returns:
            List of validated TDOA estimates
        """
        estimates = self.estimate_all_pairs(signals, method)
        return self.filter_physically_valid(estimates)


class WindowedTDOAEstimator(TDOAEstimator):
    """
    TDOA estimator that uses detected events for windowed estimation.

    More accurate because it focuses on signal segments containing bird calls.
    """

    def __init__(self,
                 mic_array: MicrophoneArray,
                 sample_rate: int = SAMPLE_RATE,
                 window_padding_ms: float = 50):
        """
        Initialize windowed TDOA estimator.

        Args:
            mic_array: Microphone array
            sample_rate: Audio sample rate
            window_padding_ms: Padding around detected events
        """
        super().__init__(mic_array, sample_rate)
        self.window_padding_samples = int(window_padding_ms * sample_rate / 1000)

    def estimate_from_event(self,
                           signals: Dict[int, np.ndarray],
                           event_start_sample: int,
                           event_end_sample: int) -> List[TDOAEstimate]:
        """
        Estimate TDOA using only the signal window containing the event.

        The window is expanded to account for maximum possible TDOA between microphones,
        ensuring the bird call signal is captured on all microphones.

        Args:
            signals: Full signals from all microphones
            event_start_sample: Start of detected event
            event_end_sample: End of detected event

        Returns:
            TDOA estimates for all pairs
        """
        # The event was detected on one microphone.
        # The same sound could arrive up to max_tdoa later on other microphones.
        # We need to include signal from:
        #   earliest_arrival = event_start - max_lag (in case event mic was furthest)
        #   latest_arrival = event_end + max_lag (in case event mic was closest)

        window_start = max(0, event_start_sample - self.max_lag_samples - self.window_padding_samples)
        window_end = event_end_sample + self.max_lag_samples + self.window_padding_samples

        # Extract windowed signals
        windowed_signals = {}
        for mic_id, full_signal in signals.items():
            end_idx = min(window_end, len(full_signal))
            windowed_signals[mic_id] = full_signal[window_start:end_idx]

        return self.estimate_all_pairs(windowed_signals)
