import numpy as np
import soundfile as sf
import librosa
import os

# --- 1. CONFIGURATION ---
FS = 44100
DUREE_SIMULATION = 15.0  # On allonge un peu pour profiter du spectacle
VITESSE_SON = 343.0
DOSSIER_OUT = "input_trajectoire_complexe"

# Vos fichiers (Adaptez les noms si besoin)
FILE_1 = r"chant oiseau/Canard_colvert.mp3"   #Mésange_bleue_chant #Oie_cendree #cri_cygne_tubercule #Bergeronnette_grise
FILE_2 = r"chant oiseau/Merle_noir_chant.mp3"   #Canard_colvert #Merle_noir_chant
FILE_3 = r"chant oiseau/Oie_cendree.mp3"

# Positions des Micros (Tétraèdre)
micros = np.array([
    [0.0, 0.0, 0.0],
    [100.0, 0.0, 0.0],
    [0.0, 100.0, 0.0], 
    [0.0, 0.0, 100.0],  
    [100.0, 100.0, 100.0],  # Mic 3
    
])

# --- 2. DÉFINITION DES DEUX TRAJECTOIRES ---

def get_pos_cygne(t):
    # CERCLE : Le cygne tourne tranquillement
    angle = (2 * np.pi / 50.0) * t 
    x = 50 + 40 * np.cos(angle)
    y = 50 + 40 * np.sin(angle)
    z = 40
    return np.array([x, y, z])

def get_pos_oie(t):
    # LIGNE DROITE : L'oie traverse le terrain (Passage rapide)
    # Départ à (-20, -20), Arrivée à (120, 120)
    vitesse = 3 # m/s
    x = 20 + t * vitesse
    y = 20 + t * vitesse
    z = 60 - t * 1 # Elle descend doucement
    return np.array([x, y, z])

def get_pos_3(t):
    x=20
    z = 30+20*np.sin(t)
    y = 10+3*t
    return np.array([x, y, z])


# --- 3. CHARGEMENT AUDIO ---
print("--- PRÉPARATION ---")
if not os.path.exists(DOSSIER_OUT): os.makedirs(DOSSIER_OUT)

def charger_et_boucler(path, fs, duree_voulue):
    if not os.path.exists(path):
        print(f"ERREUR: {path} introuvable."); exit()
    
    y, _ = librosa.load(path, sr=fs, mono=True)
    nb_samples = int(fs * duree_voulue)
    
    # Bouclage si trop court
    if len(y) < nb_samples:
        tile = int(np.ceil(nb_samples / len(y)))
        y = np.tile(y, tile)
    
    return y[:nb_samples]

print(f"Chargement du Cygne : {FILE_1}")
sig_cygne = charger_et_boucler(FILE_1, FS, DUREE_SIMULATION)

print(f"Chargement de l'Oie : {FILE_2}")
sig_oie = charger_et_boucler(FILE_2, FS, DUREE_SIMULATION)

print(f"Chargement de 3: {FILE_3}")
sig_3 = charger_et_boucler(FILE_3, FS, DUREE_SIMULATION)
# --- 4. MOTEUR DE SPATIALISATION (MIXAGE) ---
print("--- CALCUL DE PROPAGATION ---")

nb_total_samples = len(sig_cygne)
mic_buffers = [np.zeros(nb_total_samples) for _ in range(len(micros))]
t_vector = np.linspace(0, DUREE_SIMULATION, nb_total_samples)

BLOC_SIZE = 256 # Petit bloc pour fluidité
nb_blocs = nb_total_samples // BLOC_SIZE

for b in range(nb_blocs):
    idx_start = b * BLOC_SIZE
    idx_end = idx_start + BLOC_SIZE
    t_now = t_vector[idx_start]
    
    # A. TRAITEMENT CYGNE
    pos_cygne = get_pos_cygne(t_now)
    chunk_cygne = sig_cygne[idx_start:idx_end]
    
    for m in range(len(micros)):
        dist = np.linalg.norm(micros[m] - pos_cygne)
        delay_samp = int((dist / VITESSE_SON) * FS)
        atten = 10.0 / max(dist, 1.0)
        
        start_w = idx_start + delay_samp
        end_w = start_w + BLOC_SIZE
        
        if end_w < nb_total_samples:
            mic_buffers[m][start_w:end_w] += chunk_cygne * atten

    # B. TRAITEMENT OIE (On ajoute par dessus -> Mixage)
    pos_oie = get_pos_oie(t_now)
    chunk_oie = sig_oie[idx_start:idx_end]
    
    for m in range(len(micros)):
        dist = np.linalg.norm(micros[m] - pos_oie)
        delay_samp = int((dist / VITESSE_SON) * FS)
        atten = 10.0 / max(dist, 1.0)
        
        start_w = idx_start + delay_samp
        end_w = start_w + BLOC_SIZE
        
        if end_w < nb_total_samples:
            mic_buffers[m][start_w:end_w] += chunk_oie * atten

    pos_3 = get_pos_3(t_now)
    chunk_3 = sig_3[idx_start:idx_end]
    
    for m in range(len(micros)):
        dist = np.linalg.norm(micros[m] - pos_3)
        delay_samp = int((dist / VITESSE_SON) * FS)
        atten = 10.0 / max(dist, 1.0)
        
        start_w = idx_start + delay_samp
        end_w = start_w + BLOC_SIZE
        
        if end_w < nb_total_samples:
            mic_buffers[m][start_w:end_w] += chunk_3 * atten

# --- 5. EXPORT ---
print("--- SAUVEGARDE ---")
for i in range(len(micros)):
    sig = mic_buffers[i]
    # Normalisation pour éviter saturation
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig)) * 0.9
        
    sf.write(f"{DOSSIER_OUT}/mic_{i}.mp3", sig, FS)
    print(f"Mic {i} généré.")
