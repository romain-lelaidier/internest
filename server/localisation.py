import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import itertools
import os
import csv
import time

# Pour pouvoir envoyer les positions à l'IHM en real time.
from ihm_localisation import notify_position

from utils import micros
from config import CONFIG

def detect_bird_segments(y, sr, f_min=800, n_sigma=3.0, min_duration=0.01):
    """ Détecte les zones d'activité dans le signal audio """
    if len(y) == 0: return []
    sos = signal.butter(4, f_min, 'hp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    hop_length = 512
    
    rms = librosa.feature.rms(y=y_filtered, frame_length=1024, hop_length=hop_length)[0]
    if len(rms) == 0: return []

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
            if (times[i] - start_t) > min_duration:
                segments.append(start_t)
    return segments

class TDOAEngine:
    def __init__(self, max_cost=40.0, v_sound=343.0, use_parabolic=True, use_physical_filter=True, use_multistart=False, nb_pics=1):
        
        self.max_cost = max_cost
        self.v_sound = v_sound
        self.use_parabolic = use_parabolic
        self.use_physical_filter = use_physical_filter
        self.use_multistart = use_multistart
        self.nb_pics = nb_pics
        
        # Gestion des positions micros (Dict ou Array)
        # if isinstance(mic_positions, dict):
        #     self.sorted_ids = sorted(mic_positions.keys())
        #     self.mics_array_full = np.array([mic_positions[i] for i in self.sorted_ids])
        #     self.id_to_idx_map = {mid: i for i, mid in enumerate(self.sorted_ids)}
        # else:
        #     self.mics_array_full = np.array(mic_positions)
        #     self.sorted_ids = list(range(len(mic_positions)))
        #     self.id_to_idx_map = {i: i for i in range(len(mic_positions))}

    def _robust_cross_correlation(self, sig_target, sig_ref):
        """ EXACTEMENT comme ton code original """
        # Normalisation Z-score pour robustesse
        sig_target = (sig_target - np.mean(sig_target)) / (np.std(sig_target) + 1e-9)
        sig_ref = (sig_ref - np.mean(sig_ref)) / (np.std(sig_ref) + 1e-9)
        
        cc = signal.correlate(sig_target, sig_ref, mode='full')
        lags = signal.correlation_lags(len(sig_target), len(sig_ref), mode='full') / CONFIG.SAMPLE_RATE
        return cc, lags

    def _refine_peak_parabolic(self, cc, peak_idx, lags):
        """ Interpolation parabolique """
        if not self.use_parabolic:
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
        
        # Pas de temps (dt)
        if peak_idx + 1 < len(lags):
            dt = lags[peak_idx + 1] - lags[peak_idx]
        else:
            dt = lags[1] - lags[0]
        
        return lags[peak_idx] + delta * dt

    def _is_delay_physically_valid(self, delay, pos_mic_i, pos_mic_j, margin=1.2):
        """ Vérifie si le délai est possible physiquement """
        if not self.use_physical_filter:
            return True
        max_delay = np.linalg.norm(pos_mic_i - pos_mic_j) / self.v_sound
        return abs(delay) <= max_delay * margin

    def _equations_tdoa(self, pos_source, mics_subset, delays_measured):
        """ Fonction de coût pour le solveur """
        residuals = []
        k = 0
        N = len(mics_subset)
        # c est self.v_sound (capturé via self)
        
        for i in range(N - 1):
            d_ref = np.sqrt(np.sum((mics_subset[i] - pos_source) ** 2))
            for j in range(i + 1, N):
                d_i = np.sqrt(np.sum((mics_subset[j] - pos_source) ** 2))
                # Résidu : Différence de distance théorique - Différence de distance mesurée (v*t)
                residuals.append((d_i - d_ref) - (delays_measured[k] * self.v_sound))
                k += 1
        return residuals

    def locate(self, signals):
        """
        Entrée : signals [ (esp, signal) ]
        Sortie : (x,y,z), cost
        """
        # 1. Filtrage des micros valides
        signals = filter(lambda _, signal: len(signal) > 0, signals)
        if len(signals) < 4:
            return None, None

        # Positions des micros actifs pour ce calcul
        active_mics_pos = [ np.array([esp.position for esp, _ in signals]) ]
        candidats_delais = []
        
        # 2. Boucle sur les paires (Cross-Corr)
        # On compare tout le monde (i) avec tout le monde (j)
        # NOTE : Ton code original faisait ref vs les autres. Ici on fait "Full Mesh" ou "Sequential Pairs"
        # Pour coller à ton snippet `equations_tdoa`, il faut une liste linéaire de délais correspondant aux paires (0,1), (0,2)...(1,2)...
        
        for i in range(len(signals) - 1):
            for j in range(i + 1, len(signals)):
                # id_i = available_macs[i]
                # id_j = available_macs[j]
                
                # Cross-Corr
                cc, lags = self._robust_cross_correlation(signals[i][1], signals[j][1])
                
                # Masquage centre (bruit élec)
                center_idx = len(cc) // 2
                cc_masked = cc.copy()
                cc_masked[center_idx - 5 : center_idx + 5] = 0
                
                # Recherche pics
                peaks, _ = signal.find_peaks(cc_masked, distance=int(CONFIG.SAMPLE_RATE * 0.0005))
                
                delays_for_pair = []
                if len(peaks) > 0:
                    sorted_idx = np.argsort(cc_masked[peaks])[::-1]
                    top_peaks = peaks[sorted_idx][:self.nb_pics]
                    
                    for pidx in top_peaks:
                        d = self._refine_peak_parabolic(cc, pidx, lags)
                        # Check physique
                        pos_i = signals[i][0].position
                        pos_j = signals[i][0].position
                        if self._is_delay_physically_valid(d, pos_i, pos_j):
                            delays_for_pair.append(d)
                
                if len(delays_for_pair) == 0:
                    # Fallback (Max arg)
                    idx_max = np.argmax(cc_masked)
                    d_fallback = lags[idx_max]
                    delays_for_pair.append(d_fallback)
                
                candidats_delais.append(delays_for_pair)

        # 3. Optimisation Combinatoire
        combos = list(itertools.product(*candidats_delais))
        if len(combos) > 500: combos = combos[:500]

        best_res = None
        min_cost = 99999.0

        # Points de départ (Multi-Start)
        if self.use_multistart:
            init_points = [[5,5,5], [2,2,2], [8,8,8], [2,8,5], [8,2,5]]
        else:
            init_points = [[2.5, 2.5, 1.0]]

        for combo in combos:
            for init_pos in init_points:
                try:
                    res = least_squares(
                        self._equations_tdoa,
                        init_pos,
                        args=(active_mics_pos, combo),
                        bounds=([0, 0, 0], [7, 7, 3]), # Limites salle
                        xtol=1e-3,
                        ftol=1e-3,
                        max_nfev=100
                    )
                    if res.cost < min_cost:
                        min_cost = res.cost
                        best_res = res.x
                except: pass

        if min_cost < self.max_cost and best_res is not None:
            return best_res, min_cost
            
        return None, None

OUTPUT_CSV = "live_positions.csv"

# Init CSV
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["RPI_Time_us", "X", "| Y", "Z", "Cost"])

tdoa_engine = TDOAEngine()

def localiser(esps, t_start_vad, t_end_vad):
    vad_signals = []
    mic_ids_map = [] # Pour garder le lien index -> mic_id
    # Extraction pour la VAD (Somme des énergies)
    # On a besoin d'un signal de référence temporel, prenons le premier micro valide
    ref_len = 0

    for mac_addr, esp_obj in esps.items():
        # On extrait 2 secondes de signal
        _, _, sig = esp_obj.read_window(t_start_vad, t_end_vad)

        if len(sig) > 0:
            if ref_len == 0: ref_len = len(sig)
            # Sécurité taille pour la somme numpy
            if len(sig) != ref_len:
                sig = np.resize(sig, ref_len)

            vad_signals.append(sig)
            mic_ids_map.append(esp_obj.id)

    # B. DÉTECTION (Sur la somme des 2s)
    if len(vad_signals) > 0:
        # Somme des amplitudes absolues (Moyenne spatiale)
        sum_signal = np.sum(np.abs(np.array(vad_signals)), axis=0)

        # On détecte les segments dans ces 2 secondes
        # detect_bird_segments renvoie des temps RELATIFS (ex: 0.5s depuis le début de la fenêtre)
        detections_relative = detect_bird_segments(sum_signal, sr=48000, n_sigma=3.0)

        print(f"🔎 Analyse VAD (2s): {len(detections_relative)} cris potentiels.")

        # C. BOUCLE SUR CHAQUE CRI DÉTECTÉ
        for t_rel in detections_relative:
            # 1. Calcul de l'instant absolu du cri
            # t_start_vad est en microsecondes, t_rel est en secondes
            t_cri_us = t_start_vad + (t_rel * 1e6)

            # 2. Définition de la petite fenêtre TDOA
            # On commence un peu avant le cri (-0.05s) pour avoir l'attaque
            t_start_tdoa = t_cri_us - (0.05 * 1e6) 
            t_end_tdoa = t_start_tdoa + CONFIG.WINDOW_SIZE_TDOA_US

            # 3. Extraction des fenetres pour le TDOA
            tdoa_signals_dict = {}

            for mac_addr, esp_obj in esps.items():
                # Extraction chirurgicale
                _, _, sig_tdoa = esp_obj.read_window(t_start_tdoa, t_end_tdoa)

                if len(sig_tdoa) > 0 and not np.all(sig_tdoa == 0):
                    tdoa_signals_dict[mac_addr] = sig_tdoa

            print(tdoa_signals_dict)
            # 4. Localisation
            if len(tdoa_signals_dict) >= 4:
                pos, cost = tdoa_engine.locate(tdoa_signals_dict)
                print("output: ", pos, cost)

                if pos is not None:
                    print(f"   >>>  Cri à T+{t_rel:.2f}s >>> X={pos[0]:.1f} | Y={pos[1]:.1f} | Z={pos[2]:.1f}")

                    # Pour l'afficher en temps réel sur l'IHM.
                    notify_position(pos[0], pos[1], pos[2], cost, t_cri_us) # Ca va envoyer à l'IHM si il tourne les pos.
                        
                    with open(OUTPUT_CSV, 'a', newline='') as f:
                        # On enregistre le temps précis du cri
                        csv.writer(f).writerow([t_cri_us, f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}", f"{cost:.2f}"])

def routine_localiser(esps):
    while True:
        t = micros()
        target_t2 = t - CONFIG.BUFFER_DELAY_US
        target_t1 = target_t2 - CONFIG.WINDOW_SIZE_VAD_US        
        localiser(esps, target_t1, target_t2)  
        time.sleep(CONFIG.COMPUTE_INTERVAL_US / 1e6)