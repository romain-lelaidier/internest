"""
triangulation_mvt_stream_v3.py

Pipeline v2 (VAD + cross-correlation + multilateration) identique,
avec en plus l'analyse BirdNET sur les segments détectés par la VAD
(logique récupérée de postproc_2.py).

Sorties :
    - live_positions.csv   (positions 3D, comme v2)
    - live_species.json    (espèces détectées, pour l'IHM v2)
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import itertools
import os
import time
import csv
import json
import shutil
import threading
import tempfile

# ============================================================================
# 1. CONFIGURATION (identique v2)
# ============================================================================

WATCH_DIR = "./output_wavs"
OUTPUT_CSV = "live_positions.csv"
SPECIES_FILE = "live_species.json"
EXTENSION = ".wav"
VITESSE_SON = 343.0

TIMEOUT_BATCH = 0.5

MICROS = np.array([
    [0, 0, 0],
    [10.0, 0, 0],
    [0, 10.0, 0],
    [0, 0, 10.0],
    [10.0, 10.0, 10.0]
])
N_MICROS = len(MICROS)

STEP_SIZE_SEC = 1.0
FREQ_MIN_OISEAU = 800
SEUIL_DETECT_SIGMA = 0.1
MIN_DURATION = 0.01
L_WINDOW_TDOA = 0.3
NB_PICS = 1
MAX_COST_ACCEPTABLE = 40.0

USE_PARABOLIC_INTERP = True
USE_MULTISTART = False
USE_PHYSICAL_FILTER = True

# --- BirdNET ---
BIRDNET_WINDOW_SEC = 10.0      # fenêtre d'analyse BirdNET (s)
BIRDNET_COOLDOWN_SEC = 5.0     # cooldown entre deux analyses
BIRDNET_MIN_CONFIDENCE = 0.25  # seuil de confiance minimal
SPECIES_TIMEOUT = 30.0         # secondes avant qu'une espèce soit considérée partie
BIRDNET_SAMPLE_RATE = 48000    # BirdNET attend 48 kHz


# ============================================================================
# 2. ÉTAT PARTAGÉ POUR L'IHM (écrit dans live_species.json)
# ============================================================================

_species_lock = threading.Lock()
_active_species = {}   # { espece: { confidence, last_seen } }
_events = []           # [{ time, type, species }]
_birdnet_status = {
    'status': 'idle',           # 'idle' | 'cooldown' | 'analyzing'
    'cooldown_end': 0.0,
    'cooldown_total': BIRDNET_COOLDOWN_SEC,
}


def _flush_species():
    """Écrit l'état espèces dans live_species.json."""
    with _species_lock:
        now = time.time()
        active = []
        for sp, info in _active_species.items():
            active.append({
                'species': sp,
                'confidence': info['confidence'],
                'ago': round(now - info['last_seen'], 1),
            })
        payload = {
            'active': active,
            'events': _events[-200:],
            'birdnet': {
                'status': _birdnet_status['status'],
                'cooldown_end': _birdnet_status['cooldown_end'],
                'cooldown_total': _birdnet_status['cooldown_total'],
            },
        }
    try:
        with open(SPECIES_FILE, 'w') as f:
            json.dump(payload, f)
    except Exception:
        pass


def _update_species(detections):
    """Met à jour les espèces actives à partir des détections BirdNET."""
    now = time.time()
    with _species_lock:
        for det in detections:
            species = det['common_name']
            confidence = det['confidence']
            if confidence < BIRDNET_MIN_CONFIDENCE:
                continue
            is_new = species not in _active_species
            _active_species[species] = {'confidence': confidence, 'last_seen': now}
            if is_new:
                _events.append({'time': now, 'type': 'arrivee', 'species': species})

        # Expirations
        expired = [sp for sp, info in _active_species.items()
                   if (now - info['last_seen']) > SPECIES_TIMEOUT]
        for sp in expired:
            _events.append({'time': now, 'type': 'depart', 'species': sp})
            del _active_species[sp]

        if len(_events) > 500:
            _events[:] = _events[-500:]
    _flush_species()


# ============================================================================
# 3. BirdNET ANALYZER (lazy init)
# ============================================================================

_analyzer = None
_birdnet_available = None


def _get_analyzer():
    global _analyzer, _birdnet_available
    if _birdnet_available is False:
        return None
    if _analyzer is not None:
        return _analyzer
    try:
        from birdnetlib.analyzer import Analyzer
        _analyzer = Analyzer()
        _birdnet_available = True
        print("[v3] BirdNET chargé avec succès")
        return _analyzer
    except ImportError:
        print("[v3] ATTENTION : birdnetlib non installé — pip install birdnetlib")
        print("[v3] La localisation fonctionnera, mais pas l'identification d'espèces.")
        _birdnet_available = False
        return None


_birdnet_busy = threading.Lock()


def _run_birdnet_analysis(audio_data, sr):
    """Lance BirdNET dans un thread séparé."""
    try:
        _birdnet_status['status'] = 'analyzing'
        _flush_species()

        analyzer = _get_analyzer()
        if analyzer is None:
            return

        # Resample à 48 kHz si nécessaire
        if sr != BIRDNET_SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=BIRDNET_SAMPLE_RATE)

        # BirdNET a besoin d'un fichier — on passe par un fichier temporaire
        from birdnetlib import Recording
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            import soundfile as sf
            sf.write(tmp.name, audio_data, BIRDNET_SAMPLE_RATE)
            recording = Recording(analyzer, tmp.name)
            recording.analyze()
            detections = recording.detections
        finally:
            os.unlink(tmp.name)

        if detections:
            for det in detections:
                print(f"[v3] BirdNET : {det['common_name']} ({det['confidence']:.0%})")
            _update_species(detections)
        else:
            print("[v3] BirdNET : aucune espèce détectée")
    except Exception as e:
        print(f"[v3] Erreur BirdNET : {e}")
    finally:
        _birdnet_busy.release()
        _birdnet_status['status'] = 'idle'
        _flush_species()


# ============================================================================
# 4. CORRÉLATION (identique v2)
# ============================================================================

def robust_cross_correlation(sig_target, sig_ref, fs):
    cc = signal.correlate(sig_target, sig_ref, mode='full')
    lags = signal.correlation_lags(len(sig_target), len(sig_ref), mode='full') / fs
    return cc, lags


def refine_peak_parabolic(cc, peak_idx, lags):
    if not USE_PARABOLIC_INTERP:
        return lags[peak_idx]
    if peak_idx <= 0 or peak_idx >= len(cc) - 1:
        return lags[peak_idx]
    y0 = float(cc[peak_idx - 1])
    y1 = float(cc[peak_idx])
    y2 = float(cc[peak_idx + 1])
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-10:
        return lags[peak_idx]
    delta = 0.5 * (y0 - y2) / denom
    delta = np.clip(delta, -0.5, 0.5)
    if peak_idx + 1 < len(lags):
        dt = lags[peak_idx + 1] - lags[peak_idx]
    else:
        dt = lags[1] - lags[0]
    return lags[peak_idx] + delta * dt


# ============================================================================
# 5. ÉQUATIONS TDOA (identique v2)
# ============================================================================

def equations_tdoa(pos_source, mics_subset, delays_measured, c):
    residuals = []
    k = 0
    N = len(mics_subset)
    for i in range(N - 1):
        d_ref = np.sqrt(np.sum((mics_subset[i] - pos_source) ** 2))
        for j in range(i + 1, N):
            d_i = np.sqrt(np.sum((mics_subset[j] - pos_source) ** 2))
            residuals.append((d_i - d_ref) - (delays_measured[k] * c))
            k += 1
    return residuals


def is_delay_physically_valid(delay, mic_i, mic_j, c=VITESSE_SON, margin=1.2):
    if not USE_PHYSICAL_FILTER:
        return True
    max_delay = np.linalg.norm(MICROS[mic_i] - MICROS[mic_j]) / c
    return abs(delay) <= max_delay * margin


# ============================================================================
# 6. LOCALISATION (identique v2)
# ============================================================================

def process_localization(t_start, signals, fs):
    idx_start = int(t_start * fs)
    idx_end = idx_start + int(L_WINDOW_TDOA * fs)
    if idx_end >= len(signals[0]):
        return None

    chunks = [s[idx_start:idx_end] for s in signals]
    candidats_delais = []

    for i in range(N_MICROS - 1):
        for j in range(i + 1, N_MICROS):
            cc, lags = robust_cross_correlation(chunks[j], chunks[i], fs)

            center_idx = len(cc) // 2
            cc_masked = cc.copy()
            cc_masked[center_idx - 5:center_idx + 5] = 0

            peaks, _ = signal.find_peaks(cc_masked, distance=int(fs * 0.0005))

            if len(peaks) > 0:
                sorted_idx = np.argsort(cc_masked[peaks])[::-1]
                top_peaks = peaks[sorted_idx][:NB_PICS]

                delays_for_pair = []
                for pidx in top_peaks:
                    delay = refine_peak_parabolic(cc, pidx, lags)
                    if is_delay_physically_valid(delay, i, j):
                        delays_for_pair.append(delay)

                if len(delays_for_pair) == 0:
                    delays_for_pair = [lags[np.argmax(cc_masked)]]

                candidats_delais.append(delays_for_pair)
            else:
                candidats_delais.append([lags[np.argmax(cc_masked)]])

    combos = list(itertools.product(*candidats_delais))
    if len(combos) > 500:
        combos = combos[:500]

    best_res = None
    min_cost = 99999.0

    if USE_MULTISTART:
        init_points = [
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
            [8.0, 8.0, 8.0],
            [2.0, 8.0, 5.0],
            [8.0, 2.0, 5.0],
        ]
    else:
        init_points = [[5.0, 5.0, 5.0]]

    for combo in combos:
        for init_pos in init_points:
            try:
                res = least_squares(
                    equations_tdoa,
                    init_pos,
                    args=(MICROS, combo, VITESSE_SON),
                    bounds=([0, 0, 0], [15, 15, 15]),
                    xtol=1e-3,
                    ftol=1e-3,
                    max_nfev=100
                )
                if res.cost < min_cost:
                    min_cost = res.cost
                    best_res = res.x
            except Exception:
                pass

    if min_cost < MAX_COST_ACCEPTABLE and best_res is not None:
        return best_res, min_cost
    return None


# ============================================================================
# 7. VAD & TRACKING (identique v2)
# ============================================================================

class BirdTracker:
    def __init__(self, max_v=15.0, time_threshold=5.0):
        self.max_v = max_v
        self.time_threshold = time_threshold
        self.tracks = {}
        self.next_id = 1
        self.alpha = 0.6

    def assign_id(self, pos_measured, t):
        best_id = None
        min_error = float('inf')

        to_delete = [tid for tid, data in self.tracks.items()
                     if (t - data["time"]) > self.time_threshold]
        for tid in to_delete:
            del self.tracks[tid]

        for tid, data in self.tracks.items():
            dt = t - data["time"]
            pos_pred = data["pos"] + data["vel"] * dt
            dist_pred = np.linalg.norm(pos_measured - pos_pred)
            maneuver_radius = 2.0 + (self.max_v * dt * 0.5)

            if dist_pred < maneuver_radius and dist_pred < min_error:
                min_error = dist_pred
                best_id = tid

        if best_id is not None:
            data = self.tracks[best_id]
            dt = t - data["time"]
            if dt > 0:
                new_vel = (pos_measured - data["pos"]) / dt
                data["vel"] = data["vel"] * 0.4 + new_vel * 0.6
            pos_pred = data["pos"] + data["vel"] * dt
            data["pos"] = pos_pred * (1 - self.alpha) + pos_measured * self.alpha
            data["time"] = t
            return best_id
        else:
            new_id = self.next_id
            self.tracks[new_id] = {"pos": pos_measured.copy(), "vel": np.zeros(3), "time": t}
            self.next_id += 1
            return new_id


def detect_bird_segments(y, sr, f_min=1000, n_sigma=3.0):
    sos = signal.butter(4, f_min, 'hp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    hop_length = 512
    rms = librosa.feature.rms(y=y_filtered, frame_length=1024, hop_length=hop_length)[0]
    threshold = np.median(rms) + (n_sigma * np.std(rms))
    is_active = rms > threshold
    times = librosa.frames_to_time(np.arange(len(is_active)), sr=sr, hop_length=hop_length)

    segments = []
    in_segment = False
    start_t = 0
    for i, active in enumerate(is_active):
        if active and not in_segment:
            in_segment = True
            start_t = times[i]
        elif not active and in_segment:
            in_segment = False
            if (times[i] - start_t) > MIN_DURATION:
                segments.append(start_t)
    return segments


# ============================================================================
# 8. BOUCLE PRINCIPALE (v2 + BirdNET)
# ============================================================================

def main_loop():
    print(f"[v3] Bird Localizer + BirdNET")
    print(f"     Parabolic={USE_PARABOLIC_INTERP}, MultiStart={USE_MULTISTART}, PhysFilter={USE_PHYSICAL_FILTER}")
    print(f"     BirdNET window={BIRDNET_WINDOW_SEC}s, cooldown={BIRDNET_COOLDOWN_SEC}s")
    print("-" * 60)

    if os.path.exists(WATCH_DIR):
        shutil.rmtree(WATCH_DIR)
    os.makedirs(WATCH_DIR)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Batch_ID", "Global_Time_s", "X", "Y", "Z", "Cost", "Bird_ID"])

    # Init species file
    _flush_species()

    tracker = BirdTracker(max_v=15.0, time_threshold=5.0)
    current_batch_id = 0

    # --- Buffer audio pour BirdNET ---
    birdnet_buffer = np.array([], dtype=np.float64)
    birdnet_sr = None
    birdnet_cooldown_until = 0  # timestamp (wall clock) jusqu'auquel on attend

    while True:
        batch_folder = os.path.join(WATCH_DIR, f"batch_{current_batch_id}")
        if not os.path.exists(batch_folder):
            time.sleep(0.1)
            continue

        start_wait = time.time()
        files_found = []
        while (time.time() - start_wait) < TIMEOUT_BATCH:
            files_found = [i for i in range(N_MICROS)
                           if os.path.exists(os.path.join(batch_folder, f"mic_{i}{EXTENSION}"))]
            if len(files_found) == N_MICROS:
                break
            time.sleep(0.05)

        if len(files_found) < 4:
            current_batch_id += 1
            continue

        # --- Charger les WAVs ---
        signals = []
        first_valid_mic = files_found[0]
        y_ref, fs = librosa.load(
            os.path.join(batch_folder, f"mic_{first_valid_mic}{EXTENSION}"), sr=None)
        ref_len = len(y_ref)

        for i in range(N_MICROS):
            filename = os.path.join(batch_folder, f"mic_{i}{EXTENSION}")
            if i in files_found:
                y, _ = librosa.load(filename, sr=None)
                if len(y) != ref_len:
                    y = np.resize(y, ref_len)
                signals.append(y)
            else:
                signals.append(np.zeros(ref_len, dtype=np.float32))

        # --- VAD ---
        sum_signal = np.sum(np.abs(np.array(signals)), axis=0)
        t_detections = detect_bird_segments(
            sum_signal, fs, f_min=FREQ_MIN_OISEAU, n_sigma=SEUIL_DETECT_SIGMA)

        # --- Localisation TDOA (identique v2) ---
        valid_pts = 0
        if len(t_detections) > 0:
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                for t_local in t_detections:
                    res = process_localization(t_local, signals, fs)
                    if res:
                        pos, cost = res
                        t_global = (current_batch_id * STEP_SIZE_SEC) + t_local
                        bird_id = tracker.assign_id(pos, t_global)
                        writer.writerow([
                            current_batch_id, f"{t_global:.3f}",
                            f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                            f"{cost:.2f}", bird_id
                        ])
                        valid_pts += 1

        print(f"[v3] Batch {current_batch_id} : {valid_pts} pts "
              f"(IDs actifs: {list(tracker.tracks.keys())})")

        # --- BirdNET : accumuler audio & analyser si prêt ---
        if t_detections:
            # On prend le signal du micro avec le plus d'énergie
            energies = [np.sum(s.astype(np.float64) ** 2) for s in signals]
            best_mic = int(np.argmax(energies))
            best_signal = signals[best_mic].astype(np.float64)

            birdnet_buffer = np.concatenate([birdnet_buffer, best_signal])
            birdnet_sr = fs

            # Garder seulement les dernières BIRDNET_WINDOW_SEC
            max_samples = int(BIRDNET_WINDOW_SEC * fs)
            if len(birdnet_buffer) > max_samples:
                birdnet_buffer = birdnet_buffer[-max_samples:]

            now = time.time()
            buffer_duration = len(birdnet_buffer) / fs
            in_cooldown = now < birdnet_cooldown_until

            if in_cooldown:
                remaining = birdnet_cooldown_until - now
                _birdnet_status['status'] = 'cooldown'
                print(f"[v3] BirdNET cooldown ({remaining:.1f}s)")
            elif buffer_duration >= 3.0:
                # On a assez d'audio pour lancer BirdNET (min 3s)
                audio_for_birdnet = birdnet_buffer.copy()

                if _birdnet_busy.acquire(blocking=False):
                    threading.Thread(
                        target=_run_birdnet_analysis,
                        args=(audio_for_birdnet, birdnet_sr),
                        daemon=True
                    ).start()
                    print(f"[v3] BirdNET lancé ({buffer_duration:.1f}s d'audio)")
                else:
                    print("[v3] BirdNET occupé, skip")

                # Cooldown
                birdnet_cooldown_until = now + BIRDNET_COOLDOWN_SEC
                _birdnet_status['cooldown_end'] = birdnet_cooldown_until
                _birdnet_status['cooldown_total'] = BIRDNET_COOLDOWN_SEC
                birdnet_buffer = np.array([], dtype=np.float64)

        _flush_species()
        current_batch_id += 1


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[v3] Stop.")
