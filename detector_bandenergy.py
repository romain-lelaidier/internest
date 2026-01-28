import numpy as np
import librosa

def band_energy_series(y: np.ndarray, sr: int, win_s: float, hop_s: float, f_low: float, f_high: float):
    n_fft = 2048
    hop = max(1, int(round(hop_s * sr)))
    win = max(1, int(round(win_s * sr)))

    # sécurité: librosa exige win_length <= n_fft
    win = min(win, n_fft)
    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= f_low) & (freqs <= f_high)
    band_power = S[band_mask, :].sum(axis=0)

    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return times, band_power

def detect_events_from_energy(times, energy, threshold_quantile=0.92, min_event_s=0.6, merge_gap_s=0.4):
    if len(energy) == 0:
        return []

    thr = np.quantile(energy, threshold_quantile)
    active = energy >= thr

    events = []
    start = None
    for t, a in zip(times, active):
        if a and start is None:
            start = t
        if (not a) and start is not None:
            end = t
            events.append((start, end))
            start = None
    if start is not None:
        events.append((start, times[-1]))

    # Filtrer durée min
    events = [(s, e) for (s, e) in events if (e - s) >= min_event_s]

    # Fusionner si gap petit
    merged = []
    for s, e in events:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s - pe <= merge_gap_s:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])

    return [(s, e) for s, e in merged]
