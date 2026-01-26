import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import librosa.display
import itertools
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
DOSSIER = "input_trajectoire" #"input_trajectoire"#"output_5mics_clean"
EXTENSION = ".mp3"             
VITESSE_SON = 343.0
MICROS = np.array([[0,0,0],[100,0,0], [0,100,0], [0,0,100], 
                   [100.0,100.0,100.0] 
                  ])
N_micros = len(MICROS)
# Paramètres de DÉTECTION
FREQ_MIN_OISEAU = 800     # Filtre passe-haut
SEUIL_DETECT_SIGMA = 0.1  # Sensibilité du déclenchement
MIN_DURATION = 0.01       # Durée min d'un cri
PADDING = 0.1             # Marge autour du cri
Delta_t_min = 0.1
# Paramètres TDOA / LOCALISATION
L_WINDOW_TDOA = 0.3       # Taille fenêtre analyse
NB_PICS = 1               # ON GARDE LES 3 MEILLEURS PICS
MAX_COST_ACCEPTABLE = 100.0 # Tolérance erreur géométrique

# --- 2. FONCTIONS DE DÉTECTION (VAD) ---

def detect_bird_segments(y, sr, f_min=1000, n_sigma=3.0):
    """ Repère les zones où il y a de l'énergie dans les hautes fréquences """
    sos = signal.butter(4, f_min, 'hp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    
    hop_length = 512
    rms = librosa.feature.rms(y=y_filtered, frame_length=1024, hop_length=hop_length)[0]
    
    noise_floor = np.median(rms)
    std_noise = np.std(rms)
    threshold = noise_floor + (n_sigma * std_noise)
    
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
            end_t = times[i]
            if (end_t - start_t) > MIN_DURATION:
                segments.append((max(0, start_t - PADDING), end_t + PADDING))
        elif in_segment and times[i] - start_t>Delta_t_min:
            in_segment = False
            end_t = times[i]
            if (end_t - start_t) > MIN_DURATION:
                segments.append((max(0, start_t - PADDING), end_t + PADDING))
                
    return segments, y_filtered, times, rms, threshold

# --- 3. MOTEUR TDOA MULTI-PICS ---

def robust_cross_correlation(sig_target, sig_ref, fs):
    cc = signal.correlate(sig_target, sig_ref, mode='full')
    lags = signal.correlation_lags(len(sig_target), len(sig_ref), mode='full') / fs
    return cc, lags

def equations_tdoa(pos_source, mics_subset, delays_measured, c):
    N = len(mics_subset)
    residuals = []
    k = 0
    for i in range(N-1):
        d_ref = np.sqrt(np.sum((mics_subset[i] - pos_source)**2))
        for j in range(i+1, N):
            d_i = np.sqrt(np.sum((mics_subset[j] - pos_source)**2))
            residuals.append((d_i - d_ref) - (delays_measured[k] * c))
            k+=1
    return residuals

def process_localization_at_time(t_start, signals, fs):
    """ 
    Localisation à un instant T avec stratégie MULTI-PICS 
    """
    idx_start = int(t_start * fs)
    idx_end = idx_start + int(L_WINDOW_TDOA * fs)
    
    chunks = [s[idx_start:idx_end] for s in signals]
    if len(chunks[0]) < int(L_WINDOW_TDOA * fs): return None

    candidats_delais = []
    N_micro = len(chunks)
    
    # Analyse des 6 paires
    for i in range(N_micro-1):
        for j in range(i+1, N_micro):
            cc, lags = robust_cross_correlation(chunks[j], chunks[i], fs)
            
            # A. Suppression de la "Zone Morte" (Pic à 0ms dû au bruit électronique)
            lags_ms = lags * 1000
            center_idx = np.argmax(np.abs(lags_ms) < 0.001)
            width_kill = int(fs * 0.0005) # +/- 0.5ms
            cc[max(0, center_idx-width_kill):min(len(cc), center_idx+width_kill)] = 0
            edge_kill = int(fs * 0.58)
            cc[:max(0,center_idx-edge_kill)] = 0
            cc[min(len(cc),center_idx+edge_kill):]=0
            # B. Recherche des PICS (Multi-Hypothèse)
            # Distance min entre pics = 0.5ms pour éviter les rebonds immédiats
            peaks, _ = signal.find_peaks(cc, distance=int(fs*0.0005))
            
            current_lags = []
            
            if len(peaks) > 0:
                # On trie par hauteur (les plus forts en premier)
                sorted_idx = np.argsort(cc[peaks])[::-1]
                # On garde les N meilleurs
                top_peaks = peaks[sorted_idx][:NB_PICS]
                current_lags = lags[top_peaks]
            
            # C. Fallback : Si on n'a rien trouvé (ou pas assez), on complète avec le max
            if len(current_lags) == 0:
                current_lags = [lags[np.argmax(cc)]]
            
            candidats_delais.append(current_lags)

    # Résolution Combinatoire (Cartesian Product)
    # On teste toutes les combinaisons : (Pic1_Paire1, Pic1_Paire2...)
    combos = list(itertools.product(*candidats_delais))
    
    # Sécurité si trop de combinaisons (ex: bruit énorme générant plein de pics)
    if len(combos) > 2000: combos = combos[:2000]
    
    best_res = None
    min_cost = 99999.0
    
    # On teste tous les scénarios
    for combo in combos:
        # Optimisation : un seul guess central suffit souvent si les délais sont bons
        res = least_squares(
            equations_tdoa, [100,100,50], 
            args=(MICROS, combo, VITESSE_SON),
            bounds=([0,0,0], [200,200,200]),
            xtol=1e-2 # Calcul rapide
        )
        
        if res.cost < min_cost:
            min_cost = res.cost
            best_res = res.x
            
    # Si la meilleure solution mathématique est cohérente géométriquement
    if min_cost < MAX_COST_ACCEPTABLE:
        return best_res, min_cost
    
    return None

# --- 4. MAIN ---
print("1. Chargement des fichiers audio...")
signals = []
for i in range(N_micros):
    f = f"{DOSSIER}/mic_{i}{EXTENSION}"
    if not os.path.exists(f): print(f"Erreur: {f} manquant"); exit()
    y, fs_load = librosa.load(f, sr=None)
    signals.append(y)
fs = fs_load

print("2. Détection automatique des cris (Scan du Mic 0)...")
segments_trouves, _, t_rms, rms, thresh = detect_bird_segments(
    signals[0], fs, f_min=FREQ_MIN_OISEAU, n_sigma=SEUIL_DETECT_SIGMA
)
print(f"-> {len(segments_trouves)} segments trouvés.")

# Affichage detection
plt.figure(figsize=(12, 4))
plt.plot(t_rms, rms, color='gray', alpha=0.6, label='Énergie HF')
plt.axhline(thresh, color='red', linestyle='--', label='Seuil')
for seg in segments_trouves:
    plt.axvspan(seg[0], seg[1], color='green', alpha=0.3)
plt.title("Zones sélectionnées pour l'analyse")
plt.show()

print("\n3. Localisation Multi-Pics (Top 3) sur les segments...")
resultats_finaux = []

for i, (t_start, t_end) in enumerate(segments_trouves):
    print(f"Traitement segment {i+1}/{len(segments_trouves)} (t={t_start:.2f}s)...", end='\r')
    
    res = process_localization_at_time(t_start, signals, fs)
    
    if res:
        pos, cost = res
        resultats_finaux.append([pos[0], pos[1], pos[2], t_start, cost])

# --- 5. RÉSULTATS 3D ---
data = np.array(resultats_finaux)

if len(data) > 0:
    print(f"\n\n✅ {len(data)} positions validées (Coût < {MAX_COST_ACCEPTABLE})")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(MICROS[:,0], MICROS[:,1], MICROS[:,2], c='k', marker='^', s=100, label='Micros')
    
    # Affichage du temps par couleur
    p = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], cmap='turbo', s=60, edgecolor='k')
    
    ax.set_title(f"Localisation Automatique + Multi-Pics ({len(data)} pts)")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    fig.colorbar(p, label="Temps (s)")
    plt.legend()
    plt.show()
    
    # Affichage tabulaire des résultats
    print("\n--- Positions trouvées ---")
    print(f"{'Temps (s)':<10} | {'X':<8} | {'Y':<8} | {'Z':<8} | {'Erreur'}")
    for row in data:
        print(f"{row[3]:<10.2f} | {row[0]:<8.1f} | {row[1]:<8.1f} | {row[2]:<8.1f} | {row[4]:.2f}")
else:
    print("\n❌ Aucune position fiable trouvée.")