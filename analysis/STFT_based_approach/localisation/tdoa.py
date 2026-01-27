"""
Calcul des délais TDOA (Time Difference of Arrival).
"""

import numpy as np
import scipy.signal as signal

from config import Config


def get_envelope_band(sig: np.ndarray, fs: int, f_min: float, f_max: float) -> np.ndarray:
    """
    Calcule l'enveloppe du signal filtré dans une bande de fréquence.

    Args:
        sig: Signal audio
        fs: Fréquence d'échantillonnage
        f_min: Fréquence minimale du filtre (Hz)
        f_max: Fréquence maximale du filtre (Hz)

    Returns:
        Enveloppe du signal filtré
    """
    # Filtre passe-bande zero-phase
    sos = signal.butter(6, [f_min, f_max], btype='band', fs=fs, output='sos')
    sig_filt = signal.sosfiltfilt(sos, sig)

    # Enveloppe via transformée de Hilbert
    analytic = signal.hilbert(sig_filt)
    env = np.abs(analytic)

    # Lissage basse fréquence
    sos_low = signal.butter(4, 50, 'low', fs=fs, output='sos')
    return signal.sosfiltfilt(sos_low, env)


def compute_delay_envelope(sig_tar: np.ndarray, sig_ref: np.ndarray,
                           fs: int, f_min: float, f_max: float) -> tuple:
    """
    Calcule le délai par corrélation des enveloppes (méthode robuste).

    Args:
        sig_tar: Signal cible
        sig_ref: Signal de référence
        fs: Fréquence d'échantillonnage
        f_min: Fréquence min de la bande
        f_max: Fréquence max de la bande

    Returns:
        (delay en secondes, score de corrélation)
    """
    env_tar = get_envelope_band(sig_tar, fs, f_min, f_max)
    env_ref = get_envelope_band(sig_ref, fs, f_min, f_max)

    # Normalisation
    std_tar = np.std(env_tar)
    if std_tar < 1e-6:
        return 0, 0

    env_tar = (env_tar - np.mean(env_tar)) / std_tar
    env_ref = (env_ref - np.mean(env_ref)) / (np.std(env_ref) + 1e-9)

    # Corrélation croisée
    cc = signal.correlate(env_tar, env_ref, mode='full')
    lags = signal.correlation_lags(len(env_tar), len(env_ref), mode='full') / fs

    # Masque pour délais physiquement possibles
    mask = (np.abs(lags) > 0.001) & (np.abs(lags) < 0.5)
    cc_masked = cc.copy()
    cc_masked[~mask] = 0

    if np.max(np.abs(cc_masked)) == 0:
        return 0, 0

    best_idx = np.argmax(np.abs(cc_masked))
    return lags[best_idx], np.max(np.abs(cc_masked))


def gcc_phat_band(sig: np.ndarray, refsig: np.ndarray,
                  fs: int, f_min: float, f_max: float,
                  interp: int = 16) -> tuple:
    """
    GCC-PHAT avec filtrage sur la bande de fréquence de la détection.
    Utilise l'interpolation pour une meilleure précision.

    Args:
        sig: Signal cible
        refsig: Signal de référence
        fs: Fréquence d'échantillonnage
        f_min: Fréquence min de la bande
        f_max: Fréquence max de la bande
        interp: Facteur d'interpolation

    Returns:
        (delay en secondes, score)
    """
    # Fenêtrage pour réduire les artefacts
    win_sig = sig * np.hanning(len(sig))
    win_ref = refsig * np.hanning(len(refsig))

    n = len(sig) + len(refsig)

    SIG = np.fft.rfft(win_sig, n=n)
    REFSIG = np.fft.rfft(win_ref, n=n)

    # Masque fréquentiel
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mask = (freqs >= f_min) & (freqs <= f_max)

    R = SIG * np.conj(REFSIG)
    norm_factor = np.abs(R) + 1e-9
    R_phat = (R / norm_factor) * mask

    # Interpolation pour meilleure précision
    cc = np.fft.irfft(R_phat, n=(interp * n))

    max_shift = int(interp * n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # Limiter la recherche aux délais physiquement possibles
    # Distance max ~150m -> délai max ~0.44s
    max_delay_samples = int(0.5 * fs * interp)
    center = max_shift
    search_start = max(0, center - max_delay_samples)
    search_end = min(len(cc), center + max_delay_samples)

    cc_search = cc[search_start:search_end]
    local_idx = np.argmax(np.abs(cc_search))
    shift = (search_start + local_idx) - max_shift

    tau = shift / float(interp * fs)
    score = np.max(np.abs(cc_search))

    return tau, score
