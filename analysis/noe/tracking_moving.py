import numpy as np
import librosa
import scipy.signal as signal
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
VITESSE_SON = 343.0
DOSSIER = "simu_output"
FRAME_LEN = 0.8       
HOP_LEN = 0.2         
FREQ_MIN, FREQ_MAX = 2000, 8000
SEUIL_DETECT = 0.001
SEUIL_COST = 200.0  # Un peu plus haut car + de micros = somme des erreurs + grande

# !!! COPIEZ ICI LA MÊME CONFIG QUE LE GÉNÉRATEUR !!!
micros_total = np.array([
    [0.0, 0.0, 0.0],       # Mic 0
    [100.0, 0.0, 0.0],     # Mic 1
    [0.0, 100.0, 0.0],     # Mic 2
    [100.0, 100.0, 0.0],   # Mic 3
    [0.0, 0.0, 40.0],      # Mic 4
    [100.0, 0.0, 40.0],    # Mic 5
    [0.0, 100.0, 40.0],    # Mic 6
    [100.0, 100.0, 40.0]   # Mic 7
])

NB_MICROS = len(micros_total) # Compte auto

# --- FONCTIONS ---
def get_envelope(sig, fs):
    sos = signal.butter(4, [FREQ_MIN, FREQ_MAX], btype='band', fs=fs, output='sos')
    sig_filt = signal.sosfilt(sos, sig)
    analytic = signal.hilbert(sig_filt)
    env = np.abs(analytic)
    sos_low = signal.butter(4, 50, 'low', fs=fs, output='sos')
    return signal.sosfilt(sos_low, env)

def compute_delay_envelope(sig_tar, sig_ref, fs):
    env_tar = get_envelope(sig_tar, fs)
    env_ref = get_envelope(sig_ref, fs)
    
    # Normalisation
    env_tar = (env_tar - np.mean(env_tar)) / (np.std(env_tar) + 1e-9)
    env_ref = (env_ref - np.mean(env_ref)) / (np.std(env_ref) + 1e-9)
    
    cc = signal.correlate(env_tar, env_ref, mode='full')
    lags = signal.correlation_lags(len(env_tar), len(env_ref), mode='full') / fs
    
    mask = (np.abs(lags) > 0.02) & (np.abs(lags) < 0.45)
    cc[~mask] = 0
    
    if np.max(np.abs(cc)) == 0: return 0
    return lags[np.argmax(np.abs(cc))]

def equations(pos, mics, delays):
    # Cette fonction est déjà compatible N micros !
    # Elle calcule l'erreur pour chaque paire (Mic_i vs Mic_0)
    res = []
    d0 = np.sqrt(np.sum((mics[0] - pos)**2))
    
    # On itère sur tous les micros sauf le 0 (qui est la ref)
    for i in range(1, len(mics)):
        d_i = np.sqrt(np.sum((mics[i] - pos)**2))
        # Résidu = (Delta Distance Modèle) - (Delta Distance Mesuré)
        res.append((d_i - d0) - (delays[i-1] * VITESSE_SON))
    return res

# --- MAIN ---
print(f"--- TRACKING SUR {NB_MICROS} MICROS ---")

signaux = []
fs = None

# Chargement dynamique
for i in range(NB_MICROS):
    path = f"{DOSSIER}/mix_mic_{i}.wav"
    if not os.path.exists(path):
        print(f"ERREUR : {path} manquant (Vérifiez votre config micros)"); exit()
    y, sr = librosa.load(path, sr=None)
    fs = sr
    signaux.append(y)

n_frames = int((len(signaux[0]) - FRAME_LEN*fs) / (HOP_LEN*fs))
traj = []
last_pos = np.array([50., 50., 10.]) 

print(f"Analyse de {n_frames} fenêtres...")

for i in range(n_frames):
    start = int(i * HOP_LEN * fs)
    end = start + int(FRAME_LEN * fs)
    t_curr = start/fs + FRAME_LEN/2
    
    # 1. Detection Volume (Mic 0)
    env_ref = get_envelope(signaux[0][start:end], fs)
    if np.max(env_ref) < SEUIL_DETECT: continue 
    
    # 2. TDOA (1 vers Tous)
    delays = []
    # Boucle dynamique : de 1 à N-1
    for m in range(1, NB_MICROS):
        d = compute_delay_envelope(signaux[m][start:end], signaux[0][start:end], fs)
        delays.append(d)
    
    # 3. Solver
    # Le solver va recevoir (NB_MICROS - 1) équations. 
    # Plus il y en a, plus la position 'x' sera robuste.
    res = least_squares(equations, last_pos, args=(micros_total, delays), 
                        bounds=([0,0,0], [150,150,50]))
    
    if res.cost < SEUIL_COST:
        print(f"✅ t={t_curr:.2f}s | Cost={res.cost:.1f} | Pos={res.x.astype(int)}")
        last_pos = res.x
        traj.append(np.append(res.x, t_curr))

# --- VISUALISATION ---
if len(traj) > 0:
    d = np.array(traj)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Micros Dynamique
    ax.scatter(micros_total[:,0], micros_total[:,1], micros_total[:,2], 
               c='k', marker='^', s=80, label=f'{NB_MICROS} Micros')
    
    # Plot Trajectoire
    ax.plot(d[:,0], d[:,1], d[:,2], c='gray', alpha=0.5)
    p = ax.scatter(d[:,0], d[:,1], d[:,2], c=d[:,3], cmap='plasma', s=50, label='Mesure')
    
    # Vérité terrain
    try:
        truth = np.loadtxt(f"{DOSSIER}/trajectory_truth.txt", delimiter=",", skiprows=1)
        ax.plot(truth[:,1], truth[:,2], truth[:,3], 'g--', linewidth=2, label='Vérité')
    except: pass

    fig.colorbar(p, label='Temps')
    ax.legend()
    ax.set_title(f"Tracking avec {NB_MICROS} microphones")
    plt.show()
else:
    print("Rien détecté.")
