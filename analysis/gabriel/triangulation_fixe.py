import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import itertools
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
VITESSE_SON = 343.0
DOSSIER = "output_5mics_clean"
NB_PICS_PAR_MICRO = 2  # On garde les 2 meilleurs pics par micro

# Positions des micros
micros = np.array([
    [0.0, 0.0, 0.0],    # Mic 1 (Ref)
    [100.0, 0.0, 0.0],  # Mic 2
    [0.0, 100.0, 0.0],  # Mic 3
    [0.0, 0.0, 100.0]    # Mic 4
])
N_micro = len(micros)
# --- 2. FONCTIONS ---

def gcc_phat(sig_target, sig_ref, fs, max_tau=None):
    n = len(sig_target) + len(sig_ref)
    SIG_TARGET = np.fft.rfft(sig_target, n=n)
    SIG_REF = np.fft.rfft(sig_ref, n=n)
    R = SIG_TARGET * np.conj(SIG_REF)
    cc = np.fft.irfft(R / np.abs(R), n=n)
    cc = np.fft.fftshift(cc)
    if max_tau:
        center = len(cc) // 2
        shift = int(max_tau * fs)
        cc = cc[center - shift : center + shift]
        lags = np.arange(-shift, shift) / fs
    else:
        lags = np.fft.fftshift(np.fft.fftfreq(n, 1/fs))
    return cc, lags

# def equations_tdoa(pos_source, mics, delays_measured, c):
#     d_ref = np.sqrt(np.sum((mics[0] - pos_source)**2))
#     residuals = []
#     for i in range(1, len(mics)):
#         d_i = np.sqrt(np.sum((mics[i] - pos_source)**2))
#         modele_diff = d_i - d_ref
#         mesure_diff = delays_measured[i-1] * c
#         residuals.append(modele_diff - mesure_diff)
#     return residuals
    
def equations_tdoa(pos_source, mics_subset, delays_measured, c):
    N = len(mics_subset)
    residuals = []
    k = 0
    for i in range(N-1):
        d_ref = np.sqrt(np.sum((mics_subset[i] - pos_source)**2))
        for j in range(i+1,N):
            d_i = np.sqrt(np.sum((mics_subset[j] - pos_source)**2))
            residuals.append((d_i - d_ref) - (delays_measured[k] * c))
            k+=1
    return residuals

def get_dynamic_threshold(data_array, n_sigma=3.0):
    median = np.median(data_array)
    mad = np.median(np.abs(data_array - median))
    sigma_noise = 1.4826 * mad
    # On assure un seuil minimal pour éviter les problèmes sur le silence parfait
    return max(median + (n_sigma * sigma_noise), 1e-4) 

# --- 3. CHARGEMENT ---
print("--- Chargement ---")
signaux = {}
y_ref, sr = librosa.load(f"{DOSSIER}/mic_0.mp3", sr=None)
signaux[0] = y_ref

# --- 3. TRAITEMENT DU SIGNAL ---
print("--- Analyse TDOA ---")
MAX_LAG = 0.6
candidats_delais = [] 


for i in range(1, 4):
    y_target, _ = librosa.load(f"{DOSSIER}/mic_{i}.mp3", sr=fs)
    signaux[i] = y_target

for i in range(N_micro-1):
    y_ref = signaux[i]
    for j in range(i+1,N_micro):
        y_target = signaux[j]
        cc, lags = gcc_phat(y_target, y_ref, fs, max_tau=MAX_LAG)
        
        # On prend les pics
        peaks, properties = signal.find_peaks(cc, height=0.05, distance=10)
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        top_peaks = peaks[sorted_indices][:NB_PICS_PAR_MICRO]
        
        found_delays = lags[top_peaks]
        candidats_delais.append(found_delays)
        print(f"Mic 0 vs Mic {i} : {len(found_delays)} pics détectés -> {np.round(found_delays, 5)}")

candidats_delais_groupe = []
visu_data_groupe = []


# --- 4. ANALYSE DES COMBINAISONS ---

combinations = list(itertools.product(*candidats_delais))
print(f"\nNombre total de combinaisons à tester : {len(combinations)}")

tous_les_resultats = []

for idx, combo in enumerate(combinations):
    # Estimation initiale
    guess = [100.0, 100.0, 50.0]
    
    # Résolution
    res = least_squares(
        equations_tdoa, 
        guess, 
        args=(micros, combo, VITESSE_SON),
        bounds=([0, 0, 0], [200, 200, 200])
    )
    
    # On stocke tout : index, coût, position trouvée
    tous_les_resultats.append({
        "id": idx,
        "cost": res.cost,
        "pos": res.x,
        "delays": combo
    })

# --- 5. AFFICHAGE ET PLOT ---

# Tri par coût croissant (les meilleurs en premier)
tous_les_resultats.sort(key=lambda x: x["cost"])

print("\n--- TABLEAU DES COÛTS (Top 10) ---")
print(f"{'Rang':<5} | {'Coût (Erreur)':<15} | {'Position Estimée (x, y, z)':<35} | {'Type'}")
print("-" * 70)

couts_pour_plot = []
labels_pour_plot = []

for i, res in enumerate(tous_les_resultats):
    c = res["cost"]
    p = res["pos"]
    
    couts_pour_plot.append(c)
    labels_pour_plot.append(str(i))
    
    # Interprétation simple
    # Un coût < 0.1 (arbitraire mais robuste) est souvent une vraie solution
    type_sol = "✅ VRAI OISEAU ?" if c < 0.1 else "❌ CHIMÈRE"
    
    # On affiche tout, ou juste les premiers si la liste est longue
    if i < 10: 
        print(f"{i+1:<5} | {c:.6f}        | [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]         | {type_sol}")
    elif i == 10:
        print("... (suite masquée) ...")

# --- GRAPHIQUE ---
plt.figure(figsize=(10, 6))
plt.bar(range(len(couts_pour_plot)), couts_pour_plot, color=['green' if c < 0.1 else 'red' for c in couts_pour_plot])
plt.yscale('log') # Echelle log car la différence est souvent énorme
plt.axhline(y=0.1, color='blue', linestyle='--', label='Seuil de tolérance')
plt.title("Comparaison des Coûts pour chaque combinaison de délais")
plt.xlabel("Index de la combinaison (Trié)")
plt.ylabel("Coût Résiduel (Échelle Log)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()