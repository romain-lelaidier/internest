import numpy as np
import scipy.signal as signal
from scipy.optimize import least_squares
import librosa
import itertools
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
VITESSE_SON = 343.0
DOSSIER = "output_5mics_clean"
NB_PICS_PAR_MICRO = 2   # Nombre de pics à tester par paire
SEUIL_COST = 5.0        # Tolérance (plus large car mp3)

# VOTRE CONFIGURATION (5 Micros)
micros_total = np.array([
    [0.0, 0.0, 0.0],     # Mic 0
    [100.0, 0.0, 0.0],   # Mic 1
    [0.0, 100.0, 0.0],   # Mic 2
    [10.0, 20.0, 10.0],  # Mic 3
    [70.0, 80.0, 10.0]   # Mic 4
])

# --- 2. FONCTIONS (VOTRE CODE) ---

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

def equations_subset(pos_source, mics_subset, delays_measured, c):
    # Le premier micro du sous-groupe sert de référence locale
    d_ref = np.sqrt(np.sum((mics_subset[0] - pos_source)**2))
    residuals = []
    for i in range(1, len(mics_subset)):
        d_i = np.sqrt(np.sum((mics_subset[i] - pos_source)**2))
        modele_diff = d_i - d_ref
        mesure_diff = delays_measured[i-1] * c
        residuals.append(modele_diff - mesure_diff)
    return residuals

# --- 3. CHARGEMENT AUDIO ---
print("--- Chargement des 5 fichiers MP3 ---")
signaux = {} # Dictionnaire pour stocker les signaux {0: sig0, 1: sig1...}
fs_global = None

for i in range(5):
    try:
        # Chargement MP3
        y, sr = librosa.load(f"{DOSSIER}/mic_{i}.mp3", sr=None)
        signaux[i] = y
        if fs_global is None: fs_global = sr
    except FileNotFoundError:
        print(f"ERREUR : mic_{i}.mp3 introuvable.")
        exit()

print(f"Signaux chargés (Freq: {fs_global} Hz)")

# --- 4. BOUCLE PRINCIPALE (LES 5 GROUPES) ---

# On génère toutes les combinaisons de 4 micros parmi [0, 1, 2, 3, 4]
# Ex: (0,1,2,3), (0,1,2,4), (0,1,3,4), (0,2,3,4), (1,2,3,4)
all_subsets = list(itertools.combinations(range(5), 4))

print(f"\nLancement du vote sur {len(all_subsets)} groupes de micros...")

nuage_points_valides = []

for subset_idx, micros_indices in enumerate(all_subsets):
    # micros_indices est un tuple, ex: (0, 1, 2, 4)
    # Le micro de RÉFÉRENCE locale sera le premier de la liste (ex: 0, ou 1 pour le dernier groupe)
    ref_idx = micros_indices[0]
    target_idxs = micros_indices[1:] # Les 3 autres
    
    print(f"\n--- Groupe {subset_idx+1}/5 : Micros {micros_indices} (Ref: Mic {ref_idx}) ---")
    
    # Récupération des positions et signaux
    mics_subset = micros_total[list(micros_indices)]
    y_ref_local = signaux[ref_idx]
    
    # 4A. CALCUL TDOA LOCAL
    candidats_delais_groupe = []
    
    for t_idx in target_idxs:
        y_target = signaux[t_idx]
        
        # GCC-PHAT entre la cible et la ref locale
        cc, lags = gcc_phat(y_target, y_ref_local, fs_global, max_tau=0.6)
        
        peaks, props = signal.find_peaks(cc, height=0.05, distance=10)
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        top_peaks = peaks[sorted_indices][:NB_PICS_PAR_MICRO]
        
        candidats_delais_groupe.append(lags[top_peaks])
        
    # 4B. SOLVER POUR CE GROUPE
    combos_possibles = list(itertools.product(*candidats_delais_groupe))
    
    solutions_groupe = 0
    for combo in combos_possibles:
        # On essaie de résoudre
        res = least_squares(
            equations_subset, 
            [100, 100, 50], # Guess initial
            args=(mics_subset, combo, VITESSE_SON),
            bounds=([0, 0, 0], [200, 200, 200])
        )
        
        if res.cost < SEUIL_COST:
            nuage_points_valides.append(res.x)
            solutions_groupe += 1
            
    print(f"   -> {solutions_groupe} solutions trouvées.")

# --- 5. RÉSULTAT DU VOTE (CLUSTERING) ---

print("\n" + "="*50)
print(f" TOTAL : {len(nuage_points_valides)} positions candidates cumulées")
print("="*50)

if len(nuage_points_valides) > 0:
    points_array = np.array(nuage_points_valides)
    
    # DBSCAN pour trouver les zones de forte densité (le "Vote")
    # eps=4.0 : Tolérance de 4 mètres (un peu large pour compenser le MP3)
    # min_samples=3 : Il faut qu'au moins 3 solutions tombent au même endroit
    clustering = DBSCAN(eps=4.0, min_samples=3).fit(points_array)
    
    unique_labels = set(clustering.labels_)
    
    final_results = []
    
    for k in unique_labels:
        if k == -1: continue # Bruit
        
        cluster_points = points_array[clustering.labels_ == k]
        centroid = np.mean(cluster_points, axis=0)
        votes = len(cluster_points)
        
        final_results.append(centroid)
        
        print(f"✅ OISEAU CONFIRMÉ (Cluster {k})")
        print(f"   Position : [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
        print(f"   Force du vote : {votes} solutions convergentes")
        print("-" * 30)

    # --- 6. PLOT FINAL ---
    if len(final_results) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Micros
        ax.scatter(micros_total[:,0], micros_total[:,1], micros_total[:,2], c='k', marker='^', s=80, label='Micros')
        
        # Nuage de points (bruit + validés) en rouge transparent
        ax.scatter(points_array[:,0], points_array[:,1], points_array[:,2], c='red', alpha=1, s=10, label='Candidats (tous)')
        
        # Résultats finaux en vert
        final_results = np.array(final_results)
        ax.scatter(final_results[:,0], final_results[:,1], final_results[:,2], c='green', s=200, edgecolor='k', label='Oiseaux (Consensus)')
        
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        plt.show()

else:
    print("Aucune solution convergente trouvée. Vérifiez SEUIL_COST ou la géométrie.")
