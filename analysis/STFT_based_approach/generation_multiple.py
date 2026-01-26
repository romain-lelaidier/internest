import numpy as np
import soundfile as sf
import scipy.signal as signal
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION GÉNÉRALE ---
FS = 44100              
DURATION = 8.0          # Durée un peu plus longue pour laisser le temps de bouger
C = 343.0               
OUTPUT_DIR = "simu_output_overlap"

# CONFIGURATION DES MICROS (8 Micros - Cube 100m)
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

# CONFIGURATION DES OISEAUX (Ajoutez-en autant que vous voulez)
BIRDS = [
    {
        "id": 0,
        "name": "Moineau Rapide",
        "start": [10, 10, 5],
        "end": [90, 90, 30],
        "freq_shift": 0,       # Décalage de fréquence (Hz)
        "time_offset": 0.0,    # Commence à t=0
        "chirp_interval": 0.8  # Chante souvent
    },
    {
        "id": 1,
        "name": "Mésange Lente",
        "start": [40, 10, 20],
        "end": [10, 10, 40],
        "freq_shift": 1500,     # Voix plus aiguë
        "time_offset": 0.3,    # Commence un peu après (pour éviter le chevauchement parfait)
        "chirp_interval": 0.5  # Chante moins souvent
    },
    {
        "id": 2,
        "name": "Corbeau",
        "start": [50, 50, 10],
        "end": [50, 100, 40],   # Ne bouge pas
        "freq_shift": -1000,   # Voix grave
        "time_offset": 0.5,
        "chirp_interval": 1.0
    }
]

# --- 2. FONCTIONS DE GÉNÉRATION ---

def generate_bird_sound(fs, duration, freq_shift=0, time_offset=0, interval=1.0):
    """ Génère le chant spécifique d'un oiseau """
    full_signal = np.zeros(int(fs * duration))
    
    # Paramètres du "Cui"
    tweet_len = int(0.12 * fs)
    t_tweet = np.linspace(0, 0.12, tweet_len)
    
    # Fréquences de base (3500-6500) + Shift spécifique à l'oiseau
    f0 = 3500 + freq_shift
    f1 = 6500 + freq_shift
    
    chirp = signal.chirp(t_tweet, f0=f0, f1=f1, t1=0.12, method='logarithmic')
    envelope = signal.windows.tukey(tweet_len, alpha=0.3)
    one_tweet = chirp * envelope
    
    # Création de la séquence de chants
    current_t = time_offset
    while current_t < duration - 0.2:
        idx = int(current_t * fs)
        if idx + tweet_len < len(full_signal):
            full_signal[idx:idx+tweet_len] += one_tweet
        
        # Le prochain cri arrive après un intervalle +/- aléatoire
        current_t += interval + np.random.uniform(-0.1, 0.1)
        
    return full_signal

def get_trajectory_points(start_pos, end_pos, nb_points):
    """ Interpole linéairement entre Start et End """
    s = np.array(start_pos)
    e = np.array(end_pos)
    
    # Interpolation pour x, y, z
    # Création d'un tableau (N, 3)
    traj = np.zeros((nb_points, 3))
    for dim in range(3):
        traj[:, dim] = np.linspace(s[dim], e[dim], nb_points)
        
    # Ajoutons une petite courbe sinus pour faire naturel (si l'oiseau bouge)
    dist = np.linalg.norm(e - s)
    if dist > 1.0:
        traj[:, 2] += 5 * np.sin(np.linspace(0, np.pi, nb_points)) # Arche en Z
        
    return traj

# --- 3. EXÉCUTION ---

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
print(f"--- GÉNÉRATION MULTI-OISEAUX ({len(BIRDS)} sources, {len(micros)} micros) ---")

# A. Pré-calcul des sources et trajectoires
sources_audio = []
trajectories = []
t_vec = np.linspace(0, DURATION, int(FS*DURATION))

print("1. Calcul des trajectoires et sons sources...")
truth_data_list = []

for bird in BIRDS:
    # 1. Son
    sig = generate_bird_sound(FS, DURATION, bird['freq_shift'], bird['time_offset'], bird['chirp_interval'])
    sources_audio.append(sig)
    
    # 2. Trajectoire
    traj = get_trajectory_points(bird['start'], bird['end'], len(t_vec))
    trajectories.append(traj)
    
    # 3. Sauvegarde Vérité (Sous-échantillonnée)
    step = int(0.05 * FS)
    t_sub = t_vec[::step]
    traj_sub = traj[::step]
    ids = np.full(len(t_sub), bird['id'])
    
    # Format temporaire : [Temps, ID, X, Y, Z]
    block = np.column_stack((t_sub, ids, traj_sub))
    truth_data_list.append(block)

# Sauvegarde Vérité Terrain Globale
full_truth = np.vstack(truth_data_list)
# On trie par temps pour que ce soit lisible
full_truth = full_truth[full_truth[:, 0].argsort()]
header = "Temps,Bird_ID,X,Y,Z"
np.savetxt(f"{OUTPUT_DIR}/trajectory_truth.txt", full_truth, delimiter=",", header=header, fmt="%.4f")
print(f"   -> Vérité terrain sauvegardée (trajectory_truth.txt)")


# B. Mixage et Propagation
print("2. Propagation et Mixage sur les micros...")

for i, mic_pos in enumerate(micros):
    # Buffer vide pour le micro actuel
    mixed_signal = np.zeros(len(t_vec))
    
    # On ajoute la contribution de CHAQUE oiseau
    for b_idx in range(len(BIRDS)):
        source = sources_audio[b_idx]
        traj = trajectories[b_idx]
        
        # Distances variables
        dists = np.linalg.norm(traj - mic_pos, axis=1)
        
        # Délais
        delays_samples = (dists / C) * FS
        indices_source = np.arange(len(source)) - delays_samples
        
        # Interpolation (Son reçu)
        sig_recu = np.interp(indices_source, np.arange(len(source)), source, left=0, right=0)
        
        # Atténuation (1/Distance)
        # Gain x20 pour être sûr d'entendre
        attenuation = 20.0 / np.maximum(dists, 1.0)
        sig_recu *= attenuation
        
        # Ajout au mix global du micro
        mixed_signal += sig_recu
        
    # Ajout Bruit de fond (Vent)
    noise = np.random.normal(0, 0.003, len(mixed_signal))
    mixed_signal += noise
    
    # Normalisation anti-clipping
    max_val = np.max(np.abs(mixed_signal))
    if max_val > 0:
        mixed_signal = mixed_signal / max_val * 0.95
        
    sf.write(f"{OUTPUT_DIR}/mix_mic_{i}.wav", mixed_signal, FS)
    print(f"   -> Micro {i} généré.")

print("\nTerminé ! Écoutez mix_mic_0.wav, vous devriez entendre plusieurs oiseaux.")

# --- 4. APERÇU VISUEL DU SCÉNARIO ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Micros
ax.scatter(micros[:,0], micros[:,1], micros[:,2], c='k', marker='^', s=100, label='Micros')

# Trajectoires Oiseaux
colors = ['blue', 'red', 'green', 'orange']
for b_idx, bird in enumerate(BIRDS):
    traj = trajectories[b_idx]
    # On plot seulement 1 point sur 100 pour aller vite
    ax.plot(traj[::100,0], traj[::100,1], traj[::100,2], 
            label=f"Oiseau {bird['id']} ({bird['name']})", 
            color=colors[b_idx % len(colors)], linewidth=2)
    
    # Marqueurs début/fin
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', marker='o')
    ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r', marker='x')

ax.set_title(f"Scénario de Simulation : {len(BIRDS)} Oiseaux")
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()