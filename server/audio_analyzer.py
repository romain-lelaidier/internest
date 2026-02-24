"""
Analyseur audio pour détecter les instants de buzz dans les fichiers .bin.

Les ESPs envoient en continu des petits fichiers .bin.
On cherche dans quel fichier se trouve le buzz, puis à quel sample.
Instant exact = timestamp_fichier + (sample_position / sample_rate)
"""

import os
import glob
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import signal

from config import (
    AUDIO_PACKETS_DIR,
    SAMPLE_RATE,
    BUZZ_FREQUENCY_HZ
)


def load_bin_audio(filepath: str) -> np.ndarray:
    """Charge un fichier .bin audio (16-bit PCM, mono, 48kHz)."""
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    audio = np.frombuffer(raw_data, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def detect_buzz_in_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    buzz_freq: int = BUZZ_FREQUENCY_HZ,
    threshold: float = 0.3
) -> Optional[int]:
    """
    Détecte le buzz par corrélation.
    Retourne l'index du sample où le buzz commence, ou None.
    """
    if len(audio) < sample_rate * 0.05:  # Fichier trop court
        return None

    # Signal de référence (buzz de 50ms)
    duration_s = 0.05
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    reference = np.sin(2 * np.pi * buzz_freq * t)
    reference *= np.hanning(len(reference))

    # Corrélation
    correlation = signal.correlate(audio, reference, mode='valid')
    correlation = np.abs(correlation)

    if len(correlation) == 0:
        return None

    max_corr = np.max(correlation)
    if max_corr < 1e-6:
        return None

    correlation /= max_corr

    # Pic de corrélation
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]

    if peak_value >= threshold:
        return peak_idx

    return None


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse {mac}_{timestamp}.bin
    Retourne (mac, timestamp)
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]

    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Format invalide: {filename}")

    return parts[0].upper(), float(parts[1])


def get_files_after_timestamp(
    mac: str,
    after_timestamp: float,
    max_files: int = 50
) -> List[Tuple[str, float]]:
    """
    Récupère les fichiers .bin d'un ESP après un certain timestamp.
    Retourne [(filepath, file_timestamp), ...] triés par timestamp.
    """
    pattern = os.path.join(AUDIO_PACKETS_DIR, "*.bin")
    files = []

    for filepath in glob.glob(pattern):
        try:
            file_mac, file_ts = parse_filename(filepath)
            if file_mac == mac.upper() and file_ts >= after_timestamp:
                files.append((filepath, file_ts))
        except ValueError:
            continue

    # Trier par timestamp et limiter
    files.sort(key=lambda x: x[1])
    return files[:max_files]


def find_buzz_instant(
    mac: str,
    buzz_sent_timestamp: float,
    sample_rate: int = SAMPLE_RATE
) -> Optional[float]:
    """
    Cherche l'instant exact où le buzz a été entendu par un ESP.

    1. Récupère les fichiers .bin de cet ESP après l'envoi du buzz
    2. Scanne chaque fichier pour trouver le buzz
    3. Calcule : timestamp_fichier + (sample_position / sample_rate)

    Retourne l'instant absolu (timestamp Unix), ou None si non trouvé.
    """
    files = get_files_after_timestamp(mac, buzz_sent_timestamp)

    for filepath, file_timestamp in files:
        try:
            audio = load_bin_audio(filepath)
            sample_idx = detect_buzz_in_audio(audio)

            if sample_idx is not None:
                # Instant exact = timestamp du fichier + offset en secondes
                exact_time = file_timestamp + (sample_idx / sample_rate)
                return exact_time

        except Exception:
            continue

    return None


def analyze_buzz_sequence(
    buzzer_mac: str,
    buzz_timestamp: float,
    known_macs: List[str],
    sample_rate: int = SAMPLE_RATE
) -> Dict[str, float]:
    """
    Pour un buzz donné, trouve l'instant de détection sur chaque ESP.

    Retourne {mac: instant_absolu} (timestamps Unix)
    """
    detection_times = {}

    print(f"\nAnalyse du buzz de {buzzer_mac}:")

    for mac in known_macs:
        instant = find_buzz_instant(mac, buzz_timestamp, sample_rate)

        if instant is not None:
            # On stocke le temps relatif par rapport à l'envoi du buzz
            relative_time = instant - buzz_timestamp
            detection_times[mac] = relative_time
            print(f"  {mac}: +{relative_time*1000:.2f} ms")
        else:
            print(f"  {mac}: non détecté")

    return detection_times


# === Pour les tests sans fichiers réels ===

def detect_buzz_onset(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    buzz_freq: int = BUZZ_FREQUENCY_HZ,
    threshold_db: float = -20.0
) -> Optional[float]:
    """Détection par seuil d'énergie (pour tests)."""
    bandwidth = 500
    low_freq = max(buzz_freq - bandwidth, 100)
    high_freq = min(buzz_freq + bandwidth, sample_rate // 2 - 100)

    nyquist = sample_rate / 2
    b, a = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
    filtered = signal.filtfilt(b, a, audio)

    window_samples = int(10 * sample_rate / 1000)
    energy = np.convolve(filtered ** 2, np.ones(window_samples), mode='same')
    energy_db = 10 * np.log10(energy + 1e-10)

    onset_indices = np.where(energy_db > threshold_db)[0]

    if len(onset_indices) == 0:
        return None

    return onset_indices[0] / sample_rate


def detect_buzz_correlation(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    buzz_freq: int = BUZZ_FREQUENCY_HZ
) -> Optional[float]:
    """Détection par corrélation (pour tests)."""
    sample_idx = detect_buzz_in_audio(audio, sample_rate, buzz_freq)
    if sample_idx is not None:
        return sample_idx / sample_rate
    return None


if __name__ == "__main__":
    print("Test de détection audio...")

    # Signal test avec buzz à 300ms
    duration = 1.0
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.zeros_like(t)

    buzz_start = int(0.3 * SAMPLE_RATE)
    buzz_end = int(0.4 * SAMPLE_RATE)
    audio[buzz_start:buzz_end] = 0.5 * np.sin(
        2 * np.pi * BUZZ_FREQUENCY_HZ * t[buzz_start:buzz_end]
    )
    audio += 0.01 * np.random.randn(len(audio))

    sample_idx = detect_buzz_in_audio(audio)
    if sample_idx:
        print(f"Buzz réel:   300.00 ms")
        print(f"Détecté:     {sample_idx / SAMPLE_RATE * 1000:.2f} ms")
    else:
        print("Non détecté")
