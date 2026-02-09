"""
Détection précise de l'onset d'une fréquence dans une gamme donnée
(max 100 Hz de large) dans un buffer audio int16.

Approche en 2 passes :
  1. Filtre passe-bande (élargi) → enveloppe → détection de l'onset exact
  2. Courte FFT à l'onset pour identifier la fréquence et valider qu'elle
     est bien dans [f_min, f_max]
"""

import numpy as np
import scipy.signal
from config import CONFIG

MAX_RANGE_HZ = 100          # largeur max de la gamme acceptée
FILTER_ORDER = 4            # ordre du filtre Butterworth
FILTER_MARGIN_HZ = 50       # marge ajoutée au filtre de chaque côté
ENVELOPE_WINDOW = 32        # samples pour lisser l'enveloppe
NOISE_FACTOR = 5.0          # seuil = NOISE_FACTOR × bruit médian
FFT_WINDOW_S = 0.02         # fenêtre FFT courte (20ms) pour identifier la freq
FFT_DOMINANCE = 0.3         # le pic dans la bande doit valoir ≥ 30% du pic global


def detect_frequency(buffer_int16, f_min, f_max, sample_rate=CONFIG.SAMPLE_RATE):
    """
    Détecte l'onset précis d'une fréquence dans la gamme [f_min, f_max].

    Args:
        buffer_int16: numpy array int16 du signal audio
        f_min: borne basse de la gamme (Hz)
        f_max: borne haute de la gamme (Hz), f_max - f_min <= 100 Hz
        sample_rate: fréquence d'échantillonnage

    Returns:
        (freq, index) : fréquence détectée (Hz) et index du premier sample,
        ou None si aucune fréquence trouvée dans la gamme.
    """
    if f_max - f_min > MAX_RANGE_HZ:
        raise ValueError(f"Gamme trop large: {f_max - f_min:.0f} Hz (max {MAX_RANGE_HZ} Hz)")

    if len(buffer_int16) == 0:
        return None

    audio = buffer_int16.astype(np.float32) / 32768.0

    # --- Passe 1 : filtre passe-bande élargi + onset ---
    nyquist = sample_rate / 2.0
    low = max((f_min - FILTER_MARGIN_HZ) / nyquist, 1e-5)
    high = min((f_max + FILTER_MARGIN_HZ) / nyquist, 0.9999)
    sos = scipy.signal.butter(FILTER_ORDER, [low, high], btype='band', output='sos')
    filtered = scipy.signal.sosfilt(sos, audio)

    # enveloppe lissée
    envelope = np.abs(filtered)
    kernel = np.ones(ENVELOPE_WINDOW) / ENVELOPE_WINDOW
    envelope = np.convolve(envelope, kernel, mode='same')

    # seuil adaptatif (percentile bas pour ne pas être pollué si le tone domine)
    noise_level = np.percentile(envelope, 10)
    threshold = max(noise_level * NOISE_FACTOR, 1e-6)

    above = np.where(envelope > threshold)[0]
    if len(above) == 0:
        return None

    coarse_idx = int(above[0])

    # raffinement : chercher en avant sur le signal ORIGINAL (pas de group delay)
    # le premier sample non-nul = vrai onset
    noise_rms = np.std(audio[:max(1, coarse_idx - ENVELOPE_WINDOW * 2)])
    audio_threshold = max(noise_rms * 3, 1e-6)
    search_start = max(0, coarse_idx - ENVELOPE_WINDOW * 3)
    search_end = min(len(audio), coarse_idx + ENVELOPE_WINDOW)
    onset_idx = coarse_idx
    for i in range(search_start, search_end):
        if abs(audio[i]) >= audio_threshold:
            onset_idx = i
            break

    # --- Passe 2 : FFT courte pour identifier et valider la fréquence ---
    fft_samples = int(FFT_WINDOW_S * sample_rate)
    fft_start = onset_idx
    fft_end = min(fft_start + fft_samples, len(audio))

    if fft_end - fft_start < fft_samples // 2:
        return None

    chunk = audio[fft_start:fft_end]
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

    # vérifier que le pic dans [f_min, f_max] est significatif
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return None

    peak_in_band = np.max(spectrum[mask])
    peak_total = np.max(spectrum[1:])  # ignorer DC

    if peak_total == 0 or peak_in_band / peak_total < FFT_DOMINANCE:
        return None

    band_idx = np.argmax(spectrum[mask])
    detected_freq = freqs[mask][band_idx]

    return (detected_freq, onset_idx)
