"""
Reconstruction de trajectoires d'oiseaux par détection YOLO + TDOA

Pipeline:
1. Chargement des WAVs générés (8 micros)
2. Génération des spectrogrammes STFT
3. Détection YOLO (best.pt) sur chaque spectrogramme
4. Clustering DBSCAN pour identifier les oiseaux
5. Correspondance globale des clusters entre micros
6. Calcul TDOA (GCC-PHAT) pour chaque événement
7. Triangulation 3D par moindres carrés
8. Lissage des trajectoires (EMA + Savitzky-Golay)
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
import cv2
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from ultralytics import YOLO
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Audio
    FS = 44100
    C = 343.0  # Vitesse du son (m/s)

    # STFT
    N_FFT = 2048
    HOP_LENGTH = 512
    DB_THRESHOLD = 30

    # Détection YOLO
    MODEL_PATH = "./best.pt"
    CONF_THRESHOLD = 0.25

    # Clustering
    EPS_FREQ = 500  # Hz - epsilon pour DBSCAN sur fréquences
    MIN_SAMPLES = 2

    # TDOA
    PADDING = 0.08  # Marge temporelle (s) autour des détections
    TIME_TOL = 0.25  # Tolérance pour regrouper les événements (s)
    MIN_MICS = 4    # Minimum de micros pour triangulation
    COST_THRESHOLD = 50.0  # Seuil d'erreur géométrique (plus permissif)
    MIN_TDOA_SCORE = 0.02  # Score minimum pour accepter un TDOA

    # Tracking
    ALPHA = 0.4  # Facteur de lissage EMA (plus de poids à l'historique)

    # Géométrie - Cube 100m x 100m x 40m
    MICROS = np.array([
        [0.0, 0.0, 0.0],       # Mic 0
        [100.0, 0.0, 0.0],     # Mic 1
        [0.0, 100.0, 0.0],     # Mic 2
        [100.0, 100.0, 0.0],   # Mic 3
        [0.0, 0.0, 40.0],      # Mic 4
        [100.0, 0.0, 40.0],    # Mic 5
        [0.0, 100.0, 40.0],    # Mic 6
        [100.0, 100.0, 40.0]   # Mic 7
    ])

    # Couleurs pour visualisation
    COLORS = {
        0: (0, 255, 0),    # Vert
        1: (255, 0, 0),    # Bleu
        2: (0, 0, 255),    # Rouge
        3: (255, 255, 0),  # Cyan
    }


# ============================================================================
# GÉNÉRATION SPECTROGRAMMES
# ============================================================================

def generate_spectrogram(audio: np.ndarray, fs: int, output_path: str) -> tuple:
    """
    Génère et sauvegarde un spectrogramme STFT en PNG.

    Returns:
        (height, width) de l'image
    """
    f, t, Zxx = signal.stft(audio, fs, nperseg=Config.N_FFT,
                            noverlap=Config.N_FFT - Config.HOP_LENGTH)
    S_db = 20 * np.log10(np.abs(Zxx) + 1e-9)

    max_db = np.max(S_db)
    min_val = max_db - Config.DB_THRESHOLD
    S_db[S_db < min_val] = min_val

    img = ((S_db - min_val) / (max_db - min_val) * 255).astype(np.uint8)
    img = np.flipud(img)

    cv2.imwrite(output_path, img)
    return img.shape


# ============================================================================
# DÉTECTION YOLO
# ============================================================================

def detect_on_spectrogram(spectro_path: str, model: YOLO,
                          duration: float) -> list:
    """
    Exécute la détection YOLO sur un spectrogramme.

    Returns:
        Liste de détections avec coordonnées physiques (temps, fréquence)
    """
    results = model.predict(
        source=spectro_path,
        conf=Config.CONF_THRESHOLD,
        iou=0.45,
        verbose=False
    )

    img = cv2.imread(spectro_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape

    t_max = duration
    f_max = Config.FS / 2  # Nyquist

    detections = []
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Conversion pixel -> physique
        t_start = (x1 / img_w) * t_max
        t_end = (x2 / img_w) * t_max

        # Fréquence (attention au flip vertical)
        y1_stft = img_h - y1
        y2_stft = img_h - y2
        f_start = (y2_stft / img_h) * f_max
        f_end = (y1_stft / img_h) * f_max

        t_center = (t_start + t_end) / 2
        f_center = (f_start + f_end) / 2

        detections.append({
            'class_id': cls,
            'confidence': conf,
            'bbox_pixel': [x1, y1, x2, y2],
            't_start': t_start,
            't_end': t_end,
            'f_min': f_start,
            'f_max': f_end,
            't_center': t_center,
            'f_center': f_center
        })

    return detections


# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_detections(detections: list) -> tuple:
    """
    Clusterise les détections par DBSCAN sur (f_center, f_min, f_max).

    Returns:
        (labels, n_clusters)
    """
    if len(detections) == 0:
        return np.array([]), 0

    X = np.array([
        [det['f_center'], det['f_min'], det['f_max']]
        for det in detections
    ])

    # Normalisation par eps_freq
    X_scaled = X / Config.EPS_FREQ

    dbscan = DBSCAN(eps=1.0, min_samples=Config.MIN_SAMPLES, metric='euclidean')
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


def compute_global_cluster_map(clustered_data: dict) -> dict:
    """
    Établit la correspondance entre clusters locaux et identités globales.

    Returns:
        dict: {(mic_id, local_cluster): global_bird_id}
    """
    cluster_centers = []
    cluster_infos = []

    for mic_id, data in clustered_data.items():
        detections = data['detections']
        n_clusters = data['n_clusters']
        if n_clusters == 0:
            continue

        for cluster_id in range(n_clusters):
            cluster_dets = [d for d in detections if d.get('cluster') == cluster_id]
            if len(cluster_dets) == 0:
                continue

            mean_f_center = np.mean([d['f_center'] for d in cluster_dets])
            mean_f_min = np.mean([d['f_min'] for d in cluster_dets])
            mean_f_max = np.mean([d['f_max'] for d in cluster_dets])

            cluster_centers.append([mean_f_center, mean_f_min, mean_f_max])
            cluster_infos.append({'mic_id': mic_id, 'cluster_id': cluster_id})

    if len(cluster_centers) == 0:
        return {}

    cluster_centers = np.array(cluster_centers)

    # Re-clustering global
    X_scaled = cluster_centers / 5000
    dbscan_global = DBSCAN(eps=0.1, min_samples=2, metric='euclidean')
    global_labels = dbscan_global.fit_predict(X_scaled)

    # Construction du mapping
    global_map = {}
    for idx, info in enumerate(cluster_infos):
        global_map[(info['mic_id'], info['cluster_id'])] = int(global_labels[idx])

    return global_map


# ============================================================================
# TDOA & LOCALISATION
# ============================================================================

def get_envelope_band(sig: np.ndarray, fs: int, f_min: float, f_max: float) -> np.ndarray:
    """Calcule l'enveloppe du signal filtré dans une bande de fréquence."""
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

    Returns:
        (delay, score)
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

    Returns:
        (delay, score)
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


def tdoa_residuals(pos: np.ndarray, mic_positions: np.ndarray,
                   delays: np.ndarray, ref_mic_idx: int) -> np.ndarray:
    """Fonction de coût pour la triangulation TDOA."""
    res = []
    d_ref = np.linalg.norm(mic_positions[ref_mic_idx] - pos)

    for i in range(len(mic_positions)):
        if i == ref_mic_idx or np.isnan(delays[i]):
            continue
        d_i = np.linalg.norm(mic_positions[i] - pos)
        d_theo = (d_i - d_ref) / Config.C
        res.append(d_theo - delays[i])

    return np.array(res)


def find_best_initial_guess(mic_positions: np.ndarray, delays: np.ndarray,
                            ref_mic_idx: int) -> np.ndarray:
    """
    Trouve un bon point de départ pour la triangulation via recherche en grille.
    """
    best_pos = np.array([50., 50., 20.])
    best_cost = float('inf')

    # Grille de recherche grossière
    for x in np.linspace(10, 90, 5):
        for y in np.linspace(10, 90, 5):
            for z in np.linspace(5, 35, 3):
                pos = np.array([x, y, z])
                res = tdoa_residuals(pos, mic_positions, delays, ref_mic_idx)
                cost = np.sum(res**2)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

    return best_pos


# ============================================================================
# TRACKING
# ============================================================================

class TrackManager:
    """Gestionnaire de trajectoires avec lissage exponentiel."""

    def __init__(self):
        self.tracks = {}  # {bird_id: [[x, y, z, t], ...]}
        self.smoothed_last_pos = {}  # {bird_id: [x, y, z]}

    def add_point(self, pos: np.ndarray, t: float, bird_id: int):
        if bird_id not in self.tracks:
            self.tracks[bird_id] = []
            self.smoothed_last_pos[bird_id] = pos

        # Lissage exponentiel
        prev = self.smoothed_last_pos[bird_id]
        smooth_pos = Config.ALPHA * pos + (1 - Config.ALPHA) * prev

        self.tracks[bird_id].append(np.append(smooth_pos, t))
        self.smoothed_last_pos[bird_id] = smooth_pos

    def get_smoothed_trajectories(self) -> dict:
        """Applique le lissage Savitzky-Golay sur les trajectoires."""
        smoothed = {}

        for bird_id, points in self.tracks.items():
            if len(points) < 3:
                smoothed[bird_id] = np.array(points)
                continue

            data = np.array(points)
            data = data[data[:, 3].argsort()]  # Tri temporel

            x, y, z = data[:, 0], data[:, 1], data[:, 2]

            if len(x) > 5:
                win = min(7, len(x))
                if win % 2 == 0:
                    win -= 1
                x = savgol_filter(x, win, 2)
                y = savgol_filter(y, win, 2)
                z = savgol_filter(z, win, 2)

            smoothed[bird_id] = np.column_stack([x, y, z, data[:, 3]])

        return smoothed


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class TrajectoryReconstructor:
    """Pipeline complet de reconstruction de trajectoires."""

    def __init__(self, input_dir: str, output_dir: str = None, verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "reconstruction"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.audio_buffers = {}
        self.duration = None
        self.all_detections = {}
        self.clustered_data = {}
        self.global_map = {}
        self.tracker = TrackManager()
        self.debug_data = []  # Pour stocker les infos de debug

        # Charger le modèle YOLO
        print(f"Chargement du modèle YOLO: {Config.MODEL_PATH}")
        self.model = YOLO(Config.MODEL_PATH)

    def load_audio(self):
        """Charge les fichiers audio de tous les micros."""
        print("\n[1/6] Chargement des fichiers audio...")

        for i in range(len(Config.MICROS)):
            path = self.input_dir / f"mix_mic_{i}.wav"
            if not path.exists():
                raise FileNotFoundError(f"Fichier audio manquant: {path}")

            audio, fs = sf.read(str(path))
            self.audio_buffers[i] = audio

            if self.duration is None:
                self.duration = len(audio) / fs

            print(f"  - Micro {i}: {len(audio)} samples ({self.duration:.2f}s)")

        print(f"  Total: {len(self.audio_buffers)} micros chargés")

    def generate_spectrograms(self):
        """Génère les spectrogrammes pour chaque micro."""
        print("\n[2/6] Génération des spectrogrammes...")

        spectro_dir = self.output_dir / "spectrograms"
        spectro_dir.mkdir(exist_ok=True)

        self.spectro_paths = {}

        for mic_id, audio in self.audio_buffers.items():
            path = spectro_dir / f"mic_{mic_id}.png"
            shape = generate_spectrogram(audio, Config.FS, str(path))
            self.spectro_paths[mic_id] = str(path)
            print(f"  - Micro {mic_id}: {shape[1]}x{shape[0]} pixels")

    def detect_all(self):
        """Exécute la détection YOLO sur tous les spectrogrammes."""
        print("\n[3/6] Détection YOLO sur les spectrogrammes...")

        for mic_id, spectro_path in self.spectro_paths.items():
            detections = detect_on_spectrogram(spectro_path, self.model, self.duration)
            self.all_detections[mic_id] = detections
            print(f"  - Micro {mic_id}: {len(detections)} détections")

        total = sum(len(d) for d in self.all_detections.values())
        print(f"  Total: {total} détections")

    def cluster_all(self):
        """Clusterise les détections par micro puis globalement."""
        print("\n[4/6] Clustering des détections...")

        # Clustering local par micro
        for mic_id, detections in self.all_detections.items():
            if len(detections) == 0:
                continue

            labels, n_clusters = cluster_detections(detections)

            for i, det in enumerate(detections):
                det['cluster'] = int(labels[i])

            self.clustered_data[mic_id] = {
                'detections': detections,
                'n_clusters': n_clusters,
                'n_noise': np.sum(labels == -1)
            }

            print(f"  - Micro {mic_id}: {n_clusters} clusters, {np.sum(labels == -1)} bruit")

        # Correspondance globale
        self.global_map = compute_global_cluster_map(self.clustered_data)

        n_birds = len(set(self.global_map.values()) - {-1})
        print(f"  Oiseaux identifiés: {n_birds}")

    def group_events(self) -> dict:
        """Regroupe les détections en événements (chirps) par oiseau."""
        chirp_events = {}

        for mic_id, dets in self.all_detections.items():
            for det in dets:
                local_cluster = det.get('cluster', -1)
                bird_id = self.global_map.get((mic_id, local_cluster), -1)

                if bird_id == -1:
                    continue

                if bird_id not in chirp_events:
                    chirp_events[bird_id] = []

                t_center = (det['t_start'] + det['t_end']) / 2

                # Chercher un événement existant proche temporellement
                found_event = None
                for event in chirp_events[bird_id]:
                    times = [(d['t_start'] + d['t_end']) / 2 for d in event.values()]
                    avg_time = sum(times) / len(times)

                    if abs(t_center - avg_time) < Config.TIME_TOL:
                        found_event = event
                        break

                if found_event is not None:
                    found_event[mic_id] = det
                else:
                    chirp_events[bird_id].append({mic_id: det})

        return chirp_events

    def compute_trajectories(self):
        """Calcule les positions 3D par TDOA et construit les trajectoires."""
        print("\n[5/6] Calcul des trajectoires (TDOA + Triangulation)...")

        chirp_events = self.group_events()

        for bird_id, events in chirp_events.items():
            events_sorted = sorted(events, key=lambda e: list(e.values())[0]['t_start'])

            for event in events_sorted:
                if len(event) < Config.MIN_MICS:
                    continue

                ref_mic = list(event.keys())[0]
                ref_det = event[ref_mic]

                # Bande de fréquence moyenne
                f_min_avg = np.mean([d['f_min'] for d in event.values()])
                f_max_avg = np.mean([d['f_max'] for d in event.values()])

                # Utiliser une FENÊTRE COMMUNE pour tous les micros
                # basée sur le temps moyen des détections
                all_times = []
                for det in event.values():
                    all_times.append(det['t_start'])
                    all_times.append(det['t_end'])

                # Fenêtre étendue pour capturer le signal sur tous les micros
                # (inclut le délai de propagation max ~0.4s pour 140m)
                t_min = min(all_times) - 0.1
                t_max = max(all_times) + 0.1

                # Extraire les signaux sur la même fenêtre temporelle
                s_common = int(max(0, t_min * Config.FS))
                e_common = int(min(len(self.audio_buffers[0]), t_max * Config.FS))

                if e_common - s_common < 1000:  # Au moins ~20ms de signal
                    continue

                sig_ref = self.audio_buffers[ref_mic][s_common:e_common]

                delays = np.full(len(Config.MICROS), np.nan)
                delays[ref_mic] = 0.0

                # Temps de référence (détection YOLO)
                ref_t_center = ref_det['t_center']

                # Calcul TDOA pour chaque micro
                for target_mic, target_det in event.items():
                    if target_mic == ref_mic:
                        continue

                    sig_target = self.audio_buffers[target_mic][s_common:e_common]

                    if len(sig_target) < 100:
                        continue

                    # Méthode 0: Utiliser directement les temps de détection YOLO
                    # La différence de t_center donne le TDOA
                    tau0 = target_det['t_center'] - ref_t_center

                    # Méthode 1: GCC-PHAT (affinage)
                    tau1, score1 = gcc_phat_band(
                        sig_target, sig_ref,
                        Config.FS, f_min_avg, f_max_avg
                    )

                    # Méthode 2: Corrélation d'enveloppe
                    tau2, score2 = compute_delay_envelope(
                        sig_target, sig_ref,
                        Config.FS, f_min_avg, f_max_avg
                    )

                    # Stratégie: privilégier les temps YOLO car ils capturent
                    # bien le délai de propagation, affiner avec GCC si cohérent
                    if abs(tau0) < 0.5:  # Délai YOLO physiquement plausible
                        # Si GCC ou enveloppe confirme (proche de tau0), on fait la moyenne
                        if score1 > Config.MIN_TDOA_SCORE and abs(tau1 - tau0) < 0.05:
                            delays[target_mic] = (tau0 + tau1) / 2
                        elif score2 > 0.1 and abs(tau2 - tau0) < 0.05:
                            delays[target_mic] = (tau0 + tau2) / 2
                        else:
                            # Sinon on fait confiance au temps YOLO
                            delays[target_mic] = tau0
                    elif score2 > 0.1:
                        delays[target_mic] = tau2
                    elif score1 > Config.MIN_TDOA_SCORE:
                        delays[target_mic] = tau1

                # Triangulation
                n_valid_delays = np.sum(~np.isnan(delays))
                if n_valid_delays >= Config.MIN_MICS:
                    # Trouver un bon point de départ
                    if bird_id in self.tracker.smoothed_last_pos:
                        initial_guess = self.tracker.smoothed_last_pos[bird_id]
                    else:
                        # Recherche en grille pour le premier point
                        initial_guess = find_best_initial_guess(
                            Config.MICROS, delays, ref_mic
                        )

                    try:
                        res = least_squares(
                            tdoa_residuals, initial_guess,
                            args=(Config.MICROS, delays, ref_mic),
                            bounds=([0, 0, 0], [100, 100, 50]),
                            ftol=1e-4,
                            max_nfev=200
                        )

                        # Seuil adaptatif basé sur le nombre de micros
                        threshold = Config.COST_THRESHOLD * (8 / n_valid_delays)

                        if res.cost < threshold:
                            t_avg = (ref_det['t_start'] + ref_det['t_end']) / 2
                            self.tracker.add_point(res.x, t_avg, bird_id)

                            if self.verbose:
                                self.debug_data.append({
                                    'bird_id': bird_id,
                                    't': t_avg,
                                    'pos': res.x.tolist(),
                                    'cost': res.cost,
                                    'n_mics': n_valid_delays,
                                    'delays': delays.tolist()
                                })
                    except Exception as e:
                        if self.verbose:
                            print(f"    Erreur triangulation: {e}")

        # Résumé
        for bird_id, points in self.tracker.tracks.items():
            print(f"  - Oiseau {bird_id}: {len(points)} positions")

    def visualize_spectrograms(self):
        """Affiche une grille compacte des spectrogrammes avec les détections."""
        print("\n  Génération de la grille des spectrogrammes...")

        n_mics = len(self.audio_buffers)
        n_cols = 4
        n_rows = int(np.ceil(n_mics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
        axes = axes.ravel()

        # Palette de couleurs pour les oiseaux
        bird_colors = cm.tab10(np.linspace(0, 1, 10))

        for mic_id in range(n_mics):
            ax = axes[mic_id]
            audio = self.audio_buffers[mic_id]

            # Calcul STFT
            f, t, Zxx = signal.stft(audio, Config.FS,
                                    nperseg=Config.N_FFT,
                                    noverlap=Config.N_FFT - Config.HOP_LENGTH)
            S_db = 20 * np.log10(np.abs(Zxx) + 1e-9)

            # Affichage spectrogramme (jusqu'à Nyquist/2 pour plus de clarté)
            freq_max_idx = len(f) // 2
            im = ax.pcolormesh(t, f[:freq_max_idx], S_db[:freq_max_idx],
                               shading='gouraud', cmap='inferno')

            ax.set_title(f"Micro {mic_id}", fontsize=11, fontweight='bold')

            # Dessiner les bounding boxes
            detections = self.all_detections.get(mic_id, [])
            for det in detections:
                local_cluster = det.get('cluster', -1)
                bird_id = self.global_map.get((mic_id, local_cluster), -1)

                t_start, t_end = det['t_start'], det['t_end']
                f_min, f_max = det['f_min'], det['f_max']

                # Couleur selon l'oiseau
                if bird_id >= 0:
                    color = bird_colors[bird_id % 10]
                else:
                    color = (0.5, 0.5, 0.5, 0.5)  # Gris pour le bruit

                rect = Rectangle(
                    (t_start, f_min),
                    t_end - t_start,
                    f_max - f_min,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.9
                )
                ax.add_patch(rect)

                # Label ID oiseau
                if bird_id >= 0:
                    ax.text(t_start + 0.02, f_max + 200, f"{bird_id}",
                            color=color, fontsize=8, fontweight='bold',
                            verticalalignment='bottom')

            ax.set_xlabel("Temps (s)", fontsize=9)
            ax.set_ylabel("Fréquence (Hz)", fontsize=9)
            ax.set_xlim(0, self.duration)
            ax.set_ylim(0, f[freq_max_idx])
            ax.grid(True, alpha=0.2, linestyle='--', color='white')

        # Masquer les axes inutilisés
        for ax in axes[n_mics:]:
            ax.axis('off')

        # Légende globale
        legend_elements = []
        unique_birds = set()
        for (mic_id, local_cluster), bird_id in self.global_map.items():
            if bird_id >= 0:
                unique_birds.add(bird_id)

        for bird_id in sorted(unique_birds):
            color = bird_colors[bird_id % 10]
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                          label=f'Oiseau {bird_id}')
            )

        fig.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(0.99, 0.99), fontsize=10)

        plt.suptitle("Spectrogrammes STFT avec détections YOLO",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Sauvegarde
        save_path = self.output_dir / "spectrograms_detections.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé: {save_path}")

        plt.show()

    def visualize(self, show_ground_truth: bool = True):
        """Génère la visualisation 3D des trajectoires."""
        print("\n[6/6] Génération de la visualisation...")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Micros
        ax.scatter(
            Config.MICROS[:, 0], Config.MICROS[:, 1], Config.MICROS[:, 2],
            c='black', marker='^', s=100, label='Micros'
        )

        # Ground truth si disponible
        if show_ground_truth:
            truth_path = self.input_dir / "trajectory_truth.txt"
            if truth_path.exists():
                try:
                    truth = np.loadtxt(str(truth_path), delimiter=",", skiprows=1)
                    true_ids = np.unique(truth[:, 1])
                    for tid in true_ids:
                        mask = (truth[:, 1] == tid)
                        ax.plot(
                            truth[mask, 2], truth[mask, 3], truth[mask, 4],
                            linestyle='--', color='gray', linewidth=1.5, alpha=0.6,
                            label=f'Vérité Oiseau {int(tid)}'
                        )
                except Exception as e:
                    print(f"  Impossible de charger la vérité terrain: {e}")

        # Trajectoires estimées
        trajectories = self.tracker.get_smoothed_trajectories()
        colors = cm.tab10(np.linspace(0, 1, max(1, len(trajectories))))

        for i, (bird_id, data) in enumerate(trajectories.items()):
            if len(data) < 2:
                continue

            color = colors[i]
            ax.plot(
                data[:, 0], data[:, 1], data[:, 2],
                color=color, linewidth=3, alpha=0.9,
                label=f'Estimé Oiseau {bird_id}'
            )
            ax.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                color=color, s=30, alpha=0.6
            )

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 50)
        ax.set_title('Reconstruction 3D des trajectoires (YOLO + TDOA)', fontsize=14)
        ax.legend(loc='upper left')

        plt.tight_layout()

        # Sauvegarde
        save_path = self.output_dir / "trajectories_3d.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Sauvegardé: {save_path}")

        plt.show()

    def save_results(self):
        """Sauvegarde les résultats en JSON."""
        trajectories = self.tracker.get_smoothed_trajectories()

        results = {
            'config': {
                'input_dir': str(self.input_dir),
                'n_micros': len(Config.MICROS),
                'duration': self.duration
            },
            'detections': {
                mic_id: len(dets) for mic_id, dets in self.all_detections.items()
            },
            'n_birds': len(trajectories),
            'trajectories': {
                str(bird_id): data.tolist()
                for bird_id, data in trajectories.items()
            }
        }

        output_path = self.output_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nRésultats sauvegardés: {output_path}")

    def run(self):
        """Exécute le pipeline complet."""
        print("=" * 60)
        print("RECONSTRUCTION DE TRAJECTOIRES - YOLO + TDOA")
        print("=" * 60)

        self.load_audio()
        self.generate_spectrograms()
        self.detect_all()
        self.cluster_all()
        self.compute_trajectories()
        self.save_results()
        self.visualize_spectrograms()
        self.visualize()

        print("\n" + "=" * 60)
        print("TERMINÉ")
        print("=" * 60)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruction de trajectoires d'oiseaux par YOLO + TDOA"
    )
    parser.add_argument(
        "--input", "-i",
        default="./simu_output_multiple",
        help="Dossier contenant les fichiers mix_mic_*.wav"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Dossier de sortie (défaut: input/reconstruction)"
    )
    parser.add_argument(
        "--no-truth",
        action="store_true",
        help="Ne pas afficher la vérité terrain"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose avec debug TDOA"
    )

    args = parser.parse_args()

    reconstructor = TrajectoryReconstructor(args.input, args.output, verbose=args.verbose)
    reconstructor.run()

    # Afficher debug si verbose
    if args.verbose and reconstructor.debug_data:
        print("\n--- DEBUG TDOA ---")
        for d in reconstructor.debug_data[:10]:  # 10 premiers
            print(f"Bird {d['bird_id']} t={d['t']:.2f}s: pos=({d['pos'][0]:.1f}, {d['pos'][1]:.1f}, {d['pos'][2]:.1f}) cost={d['cost']:.3f}")
            delays_str = [f"{x*1000:.2f}ms" if not np.isnan(x) else "nan" for x in d['delays']]
            print(f"  Delays: {delays_str}")
