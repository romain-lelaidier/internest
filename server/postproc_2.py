"""
Post-traitement audio v2 : pipeline adaptee de triangulation_mvt_stream_v2.py.

Differences avec postproc.py :
  - VAD par seuil RMS (au lieu de spectrogramme + bounding boxes)
  - Cross-correlation classique + interpolation parabolique (au lieu de GCC-PHAT)
  - Masquage du centre de la CC pour forcer les delais non-nuls
  - Filtrage physique des delais impossibles
  - Multi-start optionnel pour l'optimisation
  - Meme interface : localiser(esps, t1, t2) → liste de Sample
"""

import time
import threading
import itertools
import numpy as np
from scipy import signal
from scipy.optimize import least_squares

from utils import micros
from config import CONFIG
from sample import Sample


# ---------------------------------------------------------------------------
#  Etat partage pour l'IHM (ihm_postproc2.py)
# ---------------------------------------------------------------------------

ihm_positions = []    # [{ x, y, z, time }]
ihm_species   = {}    # { espece: { confidence, last_seen } }
ihm_events    = []    # [{ time, type: "arrivee"|"depart", species }]
ihm_esps      = {}    # reference vers le dict esps de main.py

SPECIES_TIMEOUT = 30.0  # secondes avant qu'une espece soit consideree partie

# Etat BirdNET pour l'IHM
ihm_birdnet = {
    'status': 'idle',        # 'idle' | 'cooldown' | 'analyzing'
    'cooldown_end': 0.0,     # timestamp (s) fin de cooldown
    'cooldown_total': 5.0,   # duree totale du cooldown (s)
}


def set_esps(esps):
    """Enregistre la reference vers le dict esps pour l'IHM."""
    global ihm_esps
    ihm_esps = esps


def _update_species(sample):
    """Met a jour ihm_species et ihm_events apres une analyse BirdNET."""
    now = time.time()

    # Ajouter / mettre a jour les especes detectees
    for species, confidence in sample.species:
        is_new = species not in ihm_species
        ihm_species[species] = {'confidence': confidence, 'last_seen': now}
        if is_new:
            ihm_events.append({
                'time': now, 'type': 'arrivee', 'species': species
            })

    # Verifier les expirations
    expired = [sp for sp, info in ihm_species.items()
               if (now - info['last_seen']) > SPECIES_TIMEOUT]
    for sp in expired:
        ihm_events.append({'time': now, 'type': 'depart', 'species': sp})
        del ihm_species[sp]

    # Limiter la taille du journal
    if len(ihm_events) > 500:
        ihm_events[:] = ihm_events[-500:]


# ---------------------------------------------------------------------------
#  Constantes
# ---------------------------------------------------------------------------

VITESSE_SON = 343.0
FREQ_MIN = 800           # Hz, passe-haut
SEUIL_DETECT_SIGMA = 0.1 # n_sigma pour la VAD
MIN_DURATION = 0.01       # duree minimale d'un segment (s)
L_WINDOW_TDOA = 0.3       # fenetre TDOA (s)
NB_PICS = 1               # nombre de pics de CC a considerer par paire
MAX_COST_ACCEPTABLE = 40.0

USE_PARABOLIC_INTERP = True
USE_MULTISTART = True
USE_PHYSICAL_FILTER = True


# ---------------------------------------------------------------------------
#  VAD : detection de segments actifs par RMS (remplace librosa)
# ---------------------------------------------------------------------------

def compute_rms(y, frame_length=1024, hop_length=512):
    """RMS par fenetres glissantes (equivalent librosa.feature.rms)."""
    n_frames = 1 + (len(y) - frame_length) // hop_length
    rms = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def detect_segments(y, sr, f_min=FREQ_MIN, n_sigma=SEUIL_DETECT_SIGMA,
                    min_duration=MIN_DURATION):
    """
    Detecte les segments temporels actifs (oiseaux) dans un signal.
    Retourne une liste de timestamps de debut de segment.
    """
    # Passe-haut
    sos = signal.butter(4, f_min, 'hp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)

    hop_length = 512
    frame_length = 1024
    rms = compute_rms(y_filtered, frame_length=frame_length, hop_length=hop_length)
    threshold = np.median(rms) + n_sigma * np.std(rms)
    is_active = rms > threshold
    times = np.arange(len(is_active)) * hop_length / sr

    segments = []
    in_segment = False
    start_t = 0.0
    for i, active in enumerate(is_active):
        if active and not in_segment:
            in_segment = True
            start_t = times[i]
        elif not active and in_segment:
            in_segment = False
            if (times[i] - start_t) > min_duration:
                segments.append(start_t)
    # Segment qui court jusqu'a la fin
    if in_segment and (times[-1] - start_t) > min_duration:
        segments.append(start_t)
    return segments


# ---------------------------------------------------------------------------
#  Cross-correlation + interpolation parabolique
# ---------------------------------------------------------------------------

def robust_cross_correlation(sig_target, sig_ref, fs):
    """Cross-correlation classique (mode 'full')."""
    cc = signal.correlate(sig_target, sig_ref, mode='full')
    lags = signal.correlation_lags(len(sig_target), len(sig_ref), mode='full') / fs
    return cc, lags


def refine_peak_parabolic(cc, peak_idx, lags):
    """Interpolation parabolique sub-echantillon."""
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
    delta = np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)
    dt = lags[min(peak_idx + 1, len(lags) - 1)] - lags[peak_idx]
    return lags[peak_idx] + delta * dt


# ---------------------------------------------------------------------------
#  TDOA : equations + filtrage physique
# ---------------------------------------------------------------------------

def equations_tdoa(pos_source, mics_positions, delays_measured, c):
    """
    Residus de multilateration.
    mics_positions : array (N, 3)
    delays_measured : tuple de N*(N-1)/2 delais (ordre : paires (i,j), i<j)
    """
    residuals = []
    k = 0
    N = len(mics_positions)
    for i in range(N - 1):
        d_ref = np.linalg.norm(pos_source - mics_positions[i])
        for j in range(i + 1, N):
            d_j = np.linalg.norm(pos_source - mics_positions[j])
            residuals.append((d_j - d_ref) - (delays_measured[k] * c))
            k += 1
    return residuals


def is_delay_valid(delay, pos_i, pos_j, c=VITESSE_SON, margin=1.2):
    """Verifie si un delai est physiquement possible."""
    if not USE_PHYSICAL_FILTER:
        return True
    max_delay = np.linalg.norm(pos_i - pos_j) / c
    return abs(delay) <= max_delay * margin


# ---------------------------------------------------------------------------
#  Localisation d'un segment
# ---------------------------------------------------------------------------

def process_localization(t_start, signals, macs, positions, fs):
    """
    Pour un instant t_start, extrait les chunks TDOA et triangule.
    Retourne (position, cost) ou None.
    """
    idx_start = int(t_start * fs)
    idx_end = idx_start + int(L_WINDOW_TDOA * fs)

    n_mics = len(macs)
    mic_positions = np.array([positions[m] for m in macs])

    # Extraction des chunks
    chunks = []
    for m in macs:
        s = signals[m]
        if idx_end > len(s):
            chunk = s[idx_start:]
            chunk = np.pad(chunk, (0, idx_end - idx_start - len(chunk)))
        else:
            chunk = s[idx_start:idx_end]
        chunks.append(chunk.astype(np.float64))

    # TDOA par paires
    candidats_delais = []
    for i in range(n_mics - 1):
        for j in range(i + 1, n_mics):
            cc, lags = robust_cross_correlation(chunks[j], chunks[i], fs)

            # Masquer le centre (lag ≈ 0) pour forcer un vrai delai
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
                    if is_delay_valid(delay, mic_positions[i], mic_positions[j]):
                        delays_for_pair.append(delay)

                if not delays_for_pair:
                    delays_for_pair = [lags[np.argmax(cc_masked)]]
                candidats_delais.append(delays_for_pair)
            else:
                candidats_delais.append([lags[np.argmax(cc_masked)]])

    # Multilateration : essayer toutes les combinaisons de pics
    combos = list(itertools.product(*candidats_delais))
    if len(combos) > 500:
        combos = combos[:500]

    # Points d'initialisation
    centroid = np.mean(mic_positions, axis=0)
    if USE_MULTISTART:
        init_points = [
            centroid,
            centroid + [2, 0, 0],
            centroid - [2, 0, 0],
            centroid + [0, 2, 0],
            centroid - [0, 2, 0],
        ]
    else:
        init_points = [centroid]

    # Bornes raisonnables autour de la zone des micros
    lb = mic_positions.min(axis=0) - 5
    ub = mic_positions.max(axis=0) + 5

    best_res = None
    min_cost = 99999.0

    for combo in combos:
        for init_pos in init_points:
            try:
                res = least_squares(
                    equations_tdoa, init_pos,
                    args=(mic_positions, combo, VITESSE_SON),
                    bounds=(lb, ub),
                    xtol=1e-3, ftol=1e-3, max_nfev=100,
                )
                if res.cost < min_cost:
                    min_cost = res.cost
                    best_res = res.x
            except Exception:
                pass

    if min_cost < MAX_COST_ACCEPTABLE and best_res is not None:
        return best_res, min_cost
    return None


# ---------------------------------------------------------------------------
#  Pipeline principal : meme signature que postproc.localiser()
# ---------------------------------------------------------------------------

def localiser(esps, t1, t2):
    """
    Pipeline de detection + localisation sonore (v2).

    Parametres :
        esps : dict { mac: ESP } avec les buffers audio remplis
        t1   : timestamp debut de la fenetre (microsecondes)
        t2   : timestamp fin de la fenetre (microsecondes)
    Retourne :
        (has_activity, positions_3d)
        has_activity  : bool, True si la VAD a detecte de l'activite
        positions_3d  : liste de np.array([x, y, z]) pour chaque segment localise
                        (vide si < 2 ESPs ou si localisation echouee)
    """
    signals = {}   # { mac: signal float64 }
    positions = {} # { mac: np.array([x, y, z]) }

    # --- Etape 1 : lecture audio ---
    for mac, esp in esps.items():
        try:
            _, _, s = esp.read_window(t1, t2)
            if len(s) > 100:
                signals[mac] = s.astype(np.float64)
                positions[mac] = esp.position
        except Exception:
            continue

    if len(signals) == 0:
        return False, []

    macs = list(signals.keys())

    # --- Etape 2 : VAD sur la somme des signaux ---
    ref_len = min(len(signals[m]) for m in macs)
    sum_signal = np.sum([np.abs(signals[m][:ref_len]) for m in macs], axis=0)
    t_detections = detect_segments(sum_signal, CONFIG.SAMPLE_RATE)

    if len(t_detections) == 0:
        return False, []

    # --- Etape 3 : localisation TDOA (necessite >= 2 ESPs) ---
    estimated_positions = []
    if len(signals) >= 2:
        for t_det in t_detections:
            result = process_localization(t_det, signals, macs, positions,
                                          CONFIG.SAMPLE_RATE)
            if result is not None:
                pos, cost = result
                estimated_positions.append(pos)

    return True, estimated_positions


# ---------------------------------------------------------------------------
#  Routine principale (tourne en thread) — meme interface que postproc
# ---------------------------------------------------------------------------

BIRDNET_WINDOW_US = 10.0 * 1e6   # 10s pour BirdNET
BIRDNET_OVERLAP_US = 5.0 * 1e6   # overlap 5s → cooldown = 10 - 5 = 5s

# Verrou pour eviter de lancer plusieurs analyses BirdNET en parallele
_birdnet_busy = threading.Lock()


def _run_birdnet(sample):
    """Analyse BirdNET dans un thread separe (signal 20s, pas de position)."""
    try:
        ihm_birdnet['status'] = 'analyzing'
        sample.analyze()
        _update_species(sample)
    finally:
        _birdnet_busy.release()
        ihm_birdnet['status'] = 'idle'


def routine_postproc(esps):
    print("[postproc2] routine lancee")
    # Cooldown : timestamp (us) jusqu'auquel on ignore les nouvelles detections
    cooldown_until = 0

    while True:
        t = micros()
        target_t2 = t - CONFIG.BUFFER_DELAY_US
        target_t1 = target_t2 - CONFIG.WINDOW_SIZE_VAD_US
        try:
            n_esps = len(esps)

            # --- 1) Localisation (VAD + TDOA) sur fenetre courte (2s) ---
            has_activity, positions_3d = localiser(esps, target_t1, target_t2)

            if not has_activity:
                print(f"[postproc2] pas d'activite ({n_esps} ESP(s))")
            elif positions_3d:
                now = time.time()
                for pos in positions_3d:
                    print(f"[postproc2] POSITION : [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    ihm_positions.append({
                        'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                        'time': now
                    })
                if len(ihm_positions) > 2000:
                    ihm_positions[:] = ihm_positions[-2000:]
            else:
                print(f"[postproc2] ACTIVITE detectee ({n_esps} ESP(s), pas assez pour localiser)")

            # --- 2) BirdNET (decorrele de la localisation) ---
            in_cooldown = target_t2 <= cooldown_until
            if in_cooldown:
                cd_restant = (cooldown_until - target_t2) / 1e6
                ihm_birdnet['status'] = 'cooldown'
                print(f"[postproc2] cooldown BirdNET ({cd_restant:.1f}s restantes)")
            else:
                # Lire 10s depuis le meilleur ESP pour BirdNET
                birdnet_t2 = target_t2
                birdnet_t1 = birdnet_t2 - BIRDNET_WINDOW_US

                best_signal = None
                best_energy = 0
                for _, esp in esps.items():
                    try:
                        _, _, s = esp.read_window(birdnet_t1, birdnet_t2)
                        if len(s) > 0:
                            energy = np.sum(s.astype(np.float64) ** 2)
                            if energy > best_energy:
                                best_energy = energy
                                best_signal = s
                    except Exception:
                        continue

                if best_signal is not None:
                    dur_s = len(best_signal) / CONFIG.SAMPLE_RATE
                    print(f"[postproc2] -> BirdNET : duree signal = {dur_s:.1f}s")
                    sample = Sample(np.zeros(3), best_signal.astype(np.float64))

                    if _birdnet_busy.acquire(blocking=False):
                        threading.Thread(
                            target=_run_birdnet, args=(sample,), daemon=True
                        ).start()
                    else:
                        print("[postproc2] -> BirdNET occupe, skip")

                # Cooldown = fenetre - overlap = 5s
                cooldown_total_s = (BIRDNET_WINDOW_US - BIRDNET_OVERLAP_US) / 1e6
                cooldown_until = target_t2 + (BIRDNET_WINDOW_US - BIRDNET_OVERLAP_US)
                ihm_birdnet['cooldown_end'] = time.time() + cooldown_total_s
                ihm_birdnet['cooldown_total'] = cooldown_total_s

        finally:
            time.sleep(CONFIG.COMPUTE_INTERVAL_US / 1e6)
