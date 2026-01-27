"""
Génération de spectrogrammes STFT.
"""

import numpy as np
import scipy.signal as signal
import cv2

from config import Config


def generate_spectrogram(audio: np.ndarray, fs: int, output_path: str) -> tuple:
    """
    Génère et sauvegarde un spectrogramme STFT en PNG.

    Args:
        audio: Signal audio (1D array)
        fs: Fréquence d'échantillonnage
        output_path: Chemin de sortie pour l'image PNG

    Returns:
        (height, width) de l'image générée
    """
    f, t, Zxx = signal.stft(
        audio, fs,
        nperseg=Config.N_FFT,
        noverlap=Config.N_FFT - Config.HOP_LENGTH
    )
    S_db = 20 * np.log10(np.abs(Zxx) + 1e-9)

    max_db = np.max(S_db)
    min_val = max_db - Config.DB_THRESHOLD
    S_db[S_db < min_val] = min_val

    img = ((S_db - min_val) / (max_db - min_val) * 255).astype(np.uint8)
    img = np.flipud(img)

    cv2.imwrite(output_path, img)
    return img.shape
