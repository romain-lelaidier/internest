import os
import soundfile as sf
import numpy as np
import librosa

def read_audio(path: str, target_sr: int = 48000, mono: bool = True):
    y, sr = sf.read(path, always_2d=True)
    # y shape: (n_samples, n_channels)
    y = y.astype(np.float32)

    if mono:
        y_mono = np.mean(y, axis=1)
        if sr != target_sr:
            y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y_mono, sr
    else:
        # multicanal : on resample chaque canal
        chans = []
        for c in range(y.shape[1]):
            ch = y[:, c]
            if sr != target_sr:
                ch = librosa.resample(ch, orig_sr=sr, target_sr=target_sr)
            chans.append(ch.astype(np.float32))
        sr = target_sr if sr != target_sr else sr
        # (n_channels, n_samples)
        return np.stack(chans, axis=0), sr

def write_wav(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)

def clamp_interval(start_s: float, end_s: float, total_s: float):
    s = max(0.0, start_s)
    e = min(total_s, end_s)
    if e < s:
        e = s
    return s, e

def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float):
    i0 = int(round(start_s * sr))
    i1 = int(round(end_s * sr))
    i0 = max(0, min(len(y), i0))
    i1 = max(0, min(len(y), i1))
    return y[i0:i1]
