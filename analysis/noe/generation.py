import numpy as np
import soundfile as sf
import scipy.signal as signal
import os

# --- CONFIGURATION FLEXIBLE ---
FS = 44100              
DURATION = 5.0          
C = 343.0               
OUTPUT_DIR = "simu_output"

# D'ABORD ON CLEAR LE DOSSIER
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
else:
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

# DÉFINISSEZ ICI VOS N MICROS
# Exemple : 8 Micros (Les coins d'un cube de 100m)
micros = np.array([
    [0.0, 0.0, 0.0],       # Mic 0 (Ref)
    [100.0, 0.0, 0.0],     # Mic 1
    [0.0, 100.0, 0.0],     # Mic 2
    [100.0, 100.0, 0.0],   # Mic 3
    [0.0, 0.0, 40.0],      # Mic 4 (Haut)
    [100.0, 0.0, 40.0],    # Mic 5
    [0.0, 100.0, 40.0],    # Mic 6
    [100.0, 100.0, 40.0]   # Mic 7
])

NB_MICROS = len(micros) # Détection automatique

# --- FONCTIONS GÉNÉRATION ---
def generate_realistic_chirps(fs, total_duration):
    full_signal = np.zeros(int(fs * total_duration))
    # Création d'un 'tweet'
    tweet_len = int(0.12 * fs)
    t = np.linspace(0, 0.12, tweet_len)
    chirp = signal.chirp(t, f0=3500, f1=6500, t1=0.12, method='logarithmic')
    envelope = signal.windows.tukey(tweet_len, alpha=0.3)
    one_tweet = chirp * envelope
    
    # Séquence de tweets
    start_times = [0.5, 1.2, 2.5, 3.1, 4.0]
    for st in start_times:
        idx = int(st * fs)
        if idx + tweet_len < len(full_signal):
            full_signal[idx:idx+tweet_len] += one_tweet
    return full_signal

def get_bird_trajectory(t_total):
    # Trajectoire diagonale courbe
    x = np.linspace(10, 90, len(t_total))
    y = np.linspace(10, 90, len(t_total))
    z = 20 + 10 * np.sin(2 * np.pi * 1.0 * t_total / DURATION) # Vol ondulant
    return np.column_stack((x, y, z))

# --- MAIN ---
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print(f"--- GÉNÉRATION SIMULATION ({NB_MICROS} Micros) ---")

# 1. Source
source_clean = generate_realistic_chirps(FS, DURATION)
t_vec = np.linspace(0, DURATION, int(FS*DURATION))
positions_source = get_bird_trajectory(t_vec)

# 2. Export Vérité Terrain
print("Export vérité terrain...")
step = int(0.05 * FS)
np.savetxt(f"{OUTPUT_DIR}/trajectory_truth.txt", 
           np.column_stack((t_vec[::step], positions_source[::step])), 
           delimiter=",", header="Temps,X,Y,Z", comments='')

# 3. Propagation
print(f"Calcul pour {NB_MICROS} canaux...")
for i, mic_pos in enumerate(micros):
    dists = np.linalg.norm(positions_source - mic_pos, axis=1)
    delays_samples = (dists / C) * FS
    
    # Interpolation
    indices = np.arange(len(source_clean)) - delays_samples
    sig_mic = np.interp(indices, np.arange(len(source_clean)), source_clean, left=0, right=0)
    
    # Atténuation & Bruit
    sig_mic *= (15.0 / np.maximum(dists, 1.0))
    sig_mic += np.random.normal(0, 0.002, len(sig_mic))
    
    # Normalisation
    if np.max(np.abs(sig_mic)) > 0:
        sig_mic = sig_mic / np.max(np.abs(sig_mic)) * 0.9
        
    sf.write(f"{OUTPUT_DIR}/mix_mic_{i}.wav", sig_mic, FS)

print("Terminé.")
