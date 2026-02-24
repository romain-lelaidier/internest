import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import itertools
import os
import time
import csv
import sys

# --- 1. CONFIGURATION ---
WATCH_DIR = "./output_wavs"
OUTPUT_CSV = "live_positions.csv"
EXTENSION = ".wav"
VITESSE_SON = 343.0

# Timeout : Combien de temps on attend les retardataires avant de lancer le calcul ?
TIMEOUT_BATCH = 0.5 # secondes

# Géométrie
MICROS = np.array([
    [0,0,0], [100,0,0], [0,100,0], [0,0,100], [100,100,100]
])
N_MICROS = len(MICROS)

# Paramètres Flux & TDOA
STEP_SIZE_SEC = 1.0
FREQ_MIN_OISEAU = 800
SEUIL_DETECT_SIGMA = 0.1
MIN_DURATION = 0.01
L_WINDOW_TDOA = 0.3
NB_PICS = 1
MAX_COST_ACCEPTABLE = 100.0

# --- 2. FONCTIONS DE TRAITEMENT (VAD & TDOA) ---
# (Ces fonctions sont identiques à votre version précédente, juste nettoyées)

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
                segments.append(start_t) # On ne garde que le début pour l'instant
    return segments

def robust_cross_correlation(sig_target, sig_ref, fs):
    cc = signal.correlate(sig_target, sig_ref, mode='full')
    lags = signal.correlation_lags(len(sig_target), len(sig_ref), mode='full') / fs
    return cc, lags

def equations_tdoa(pos_source, mics_subset, delays_measured, c):
    residuals = []
    k = 0
    N = len(mics_subset)
    for i in range(N-1):
        d_ref = np.sqrt(np.sum((mics_subset[i] - pos_source)**2))
        for j in range(i+1, N):
            d_i = np.sqrt(np.sum((mics_subset[j] - pos_source)**2))
            residuals.append((d_i - d_ref) - (delays_measured[k] * c))
            k+=1
    return residuals

def process_localization(t_start, signals, fs):
    idx_start = int(t_start * fs)
    idx_end = idx_start + int(L_WINDOW_TDOA * fs)
    
    # Vérification bounds
    if idx_end >= len(signals[0]): return None
    
    chunks = [s[idx_start:idx_end] for s in signals]
    candidats_delais = []
    
    for i in range(N_MICROS-1):
        for j in range(i+1, N_MICROS):
            cc, lags = robust_cross_correlation(chunks[j], chunks[i], fs)
            # Suppression pic central (bruit électronique)
            center_idx = len(cc)//2
            cc[center_idx-5:center_idx+5] = 0
            
            peaks, _ = signal.find_peaks(cc, distance=int(fs*0.0005))
            if len(peaks) > 0:
                sorted_idx = np.argsort(cc[peaks])[::-1]
                top_peaks = peaks[sorted_idx][:NB_PICS]
                candidats_delais.append(lags[top_peaks])
            else:
                candidats_delais.append([lags[np.argmax(cc)]])

    combos = list(itertools.product(*candidats_delais))
    if len(combos) > 500: combos = combos[:500] # Optimisation vitesse
    
    best_res = None
    min_cost = 99999.0
    
    for combo in combos:
        try:
            res = least_squares(
                equations_tdoa, [50,50,20], 
                args=(MICROS, combo, VITESSE_SON),
                bounds=([0,0,0], [200,200,100]),
                xtol=1e-1, ftol=1e-1 # Moins précis mais plus rapide pour le temps réel
            )
            if res.cost < min_cost:
                min_cost = res.cost
                best_res = res.x
        except: pass
            
    if min_cost < MAX_COST_ACCEPTABLE:
        return best_res, min_cost
    return None

# --- 3. BOUCLE ROBUSTE (TIMEOUT + SILENCE) ---


def main_loop():
    print(f"📡 Démarrage stream (Tolérance panne activée)...")
    
    # Création CSV
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(["Batch_ID", "Global_Time_s", "X", "Y", "Z", "Cost"])
    
    current_batch_id = 0
    
    while True:
        batch_folder = os.path.join(WATCH_DIR, f"batch_{current_batch_id}")
        
        # 1. Attente création du dossier
        if not os.path.exists(batch_folder):
            time.sleep(0.1)
            continue
            
        # 2. Attente intelligente des fichiers (Timeout)
        start_wait = time.time()
        files_found = []
        
        # On boucle tant que le temps n'est pas écoulé ET qu'on n'a pas tout le monde
        while (time.time() - start_wait) < TIMEOUT_BATCH:
            files_found = [i for i in range(N_MICROS) if os.path.exists(os.path.join(batch_folder, f"mic_{i}{EXTENSION}"))]
            
            if len(files_found) == N_MICROS:
                break # Tout le monde est là !
            time.sleep(0.05)
            
        # 3. Vérification post-timeout
        if len(files_found) == 0:
            print(f"⚠️ Batch {current_batch_id} vide ou trop lent. Skip.")
            current_batch_id += 1
            continue
            
        if len(files_found) < N_MICROS:
            missing = [i for i in range(N_MICROS) if i not in files_found]
            print(f"⚠️ Batch {current_batch_id} incomplet. Manque: {missing}. Remplissage silence...")
            
        # 4. Chargement et Remplissage
        signals = []
        fs = 48000
        # On a besoin d'une longueur de référence (celle du premier fichier valide trouvé)
        ref_len = 0
        
        # Première passe pour trouver la longueur
        first_valid_mic = files_found[0]
        y_ref, fs = librosa.load(os.path.join(batch_folder, f"mic_{first_valid_mic}{EXTENSION}"), sr=None)
        ref_len = len(y_ref)
        
        # Deuxième passe : Chargement réel
        for i in range(N_MICROS):
            filename = os.path.join(batch_folder, f"mic_{i}{EXTENSION}")
            if i in files_found:
                # Fichier existe
                y, _ = librosa.load(filename, sr=None)
                # Sécurité taille (parfois write() varie de 1 sample)
                if len(y) != ref_len:
                    y = np.resize(y, ref_len)
                signals.append(y)
            else:
                # Fichier manquant -> Silence
                signals.append(np.zeros(ref_len, dtype=np.float32))

        # 5. Détection Améliorée (SOMME)
        # Si Mic 0 est mort (silence), detect_bird_segments sur signals[0] ne donnerait rien.
        # On fait la somme absolue de tous les signaux pour détecter s'il y a un cri n'importe où.
        sum_signal = np.sum(np.abs(np.array(signals)), axis=0)
        
        # On lance la détection sur ce signal combiné
        t_detections = detect_bird_segments(sum_signal, fs, f_min=FREQ_MIN_OISEAU, n_sigma=SEUIL_DETECT_SIGMA)
        
        # 6. Localisation
        valid_pts = 0
        if len(t_detections) > 0:
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                for t_local in t_detections:
                    res = process_localization(t_local, signals, fs)
                    if res:
                        pos, cost = res
                        t_global = (current_batch_id * STEP_SIZE_SEC) + t_local
                        writer.writerow([current_batch_id, f"{t_global:.3f}", 
                                         f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}", 
                                         f"{cost:.2f}"])
                        valid_pts += 1
                        
        print(f"⚡ Batch {current_batch_id} : {len(files_found)}/{N_MICROS} micros -> {valid_pts} pts")
        
        current_batch_id += 1

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Stop.")
        


