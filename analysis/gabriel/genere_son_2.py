import numpy as np
import librosa
import soundfile as sf
import os

# --- 1. CONFIGURATION ---
VITESSE_SON = 343.0
FS = 44100
DOSSIER_SORTIE = "output_5mics_clean" # Je change le nom du dossier pour ne pas mélanger

# Noms de vos fichiers audio sources
FILE_SOURCE_1 = "chant oiseau\cri_cygne_tubercule.mp3"
FILE_SOURCE_2 = "chant oiseau\Oie_cendree.mp3"

# Positions des 5 Microphones
micros = np.array([
    [0.0, 0.0, 0.0],     # Mic 0 (Ref)
    [100.0, 0.0, 0.0],   # Mic 1
    [0.0, 100.0, 0.0],   # Mic 2
    [0.0, 0.0, 100.0],    # Mic 3
    [100.0, 100.0, 100.0],     # Mic 4
])

# Positions des Oiseaux
sources_config = [
    {"pos": np.array([30.0, 80.0, 50.0]),  "file": FILE_SOURCE_1},
    {"pos": np.array([50.0, 20.0, 10.0]), "file": FILE_SOURCE_2}
]

# --- 2. FONCTIONS ---

def charger_ou_generer(fichier, fs, duration_sec=2.0):
    if os.path.exists(fichier):
        print(f"Chargement de {fichier}...")
        y, _ = librosa.load(fichier, sr=fs, mono=True)
        return y
    else:
        print(f"ATTENTION : {fichier} introuvable. Génération d'un son synthétique.")
        t = np.linspace(0, duration_sec, int(fs*duration_sec))
        freq = 440 if "1" in fichier else 880
        return np.sin(2*np.pi*freq*t) * np.exp(-t)

# --- 3. GÉNÉRATION ---

if not os.path.exists(DOSSIER_SORTIE):
    os.makedirs(DOSSIER_SORTIE)

audio_data = []
for src in sources_config:
    audio_data.append(charger_ou_generer(src["file"], FS))

# Calcul longueur max
max_samples = 0
for i, mic in enumerate(micros):
    for j, src in enumerate(sources_config):
        dist = np.linalg.norm(mic - src["pos"])
        delay_s = dist / VITESSE_SON
        len_tot = int(delay_s * FS) + len(audio_data[j])
        if len_tot > max_samples:
            max_samples = len_tot

max_samples += 1000 
print(f"\nLongueur des fichiers générés : {max_samples/FS:.2f} secondes")
print("Génération SANS bruit blanc...")

signaux_mixes = []

for i, mic in enumerate(micros):
    buffer_mic = np.zeros(max_samples)
    
    for j, src in enumerate(sources_config):
        dist = np.linalg.norm(mic - src["pos"])
        delay_samples = int((dist / VITESSE_SON) * FS)
        attenuation = 1.0 / max(dist, 0.5) 
        
        sound = audio_data[j]
        end_idx = delay_samples + len(sound)
        
        buffer_mic[delay_samples:end_idx] += sound * attenuation
        
    # --- MODIFICATION ICI : PAS DE BRUIT ---
    # bruit = np.random.normal(0, 0.001, max_samples)
    # buffer_mic += bruit
    
    signaux_mixes.append(buffer_mic)

# --- 4. SAUVEGARDE ---

max_val_global = np.max([np.max(np.abs(sig)) for sig in signaux_mixes])
if max_val_global == 0: max_val_global = 1.0 # Sécurité anti-crash

print(f"Normalisation (1/{max_val_global:.4f})")

for i, sig in enumerate(signaux_mixes):
    sig_norm = sig / max_val_global
    filename = f"{DOSSIER_SORTIE}/mic_{i}.mp3"
    sf.write(filename, sig_norm, FS)
    print(f" -> Sauvegardé : {filename}")

print("\nFichiers propres prêts dans le dossier 'output_5mics_clean'.")
print("Pensez à mettre à jour la variable 'DOSSIER' dans votre script de résolution !")