import numpy as np
import librosa
from scipy.signal import find_peaks

def estimate_num_birds(y_seg: np.ndarray, sr: int, f_low=2000, f_high=8000):
    # Mel-spectrogramme léger
    S = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_fft=2048, hop_length=256, n_mels=128, fmin=0, fmax=sr/2)
    freqs_mel = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)

    band = (freqs_mel >= f_low) & (freqs_mel <= f_high)
    band_energy_t = S[band, :].sum(axis=0)

    # Normaliser et lisser
    x = band_energy_t.astype(np.float32)
    if x.size < 5:
        return 1, {"peaks": [], "band_energy": x}

    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = np.convolve(x, np.ones(5)/5, mode="same")

    # pics = événements “chant” => proxy nb individus (min)
    peaks, props = find_peaks(x, height=0.35, distance=3)
    n = int(max(1, min(6, len(peaks))))  # borne prudente
    return n, {"peaks": peaks.tolist(), "band_energy": x}
