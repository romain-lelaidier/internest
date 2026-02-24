import numpy as np
import os
import time
import struct
from scipy.io.wavfile import write
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
INPUT_FOLDER = '/home/pi/udp_servers/packets'
OUTPUT_FOLDER = './output_wavs'
SAMPLE_RATE = 48000

# PARAMÈTRES DE LA FENÊTRE GLISSANTE
CHUNK_DURATION = 2.0       # Durée de chaque fichier WAV (fenêtre d'analyse)
OVERLAP_DURATION = 1.0     # Durée du chevauchement (ce qui est commun entre 2 fichiers)

# Vérification de sécurité
if OVERLAP_DURATION >= CHUNK_DURATION:
    print("❌ Erreur : L'overlap doit être strictement inférieur à la durée du chunk.")
    exit()

# Calcul des tailles en échantillons
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)
SAMPLES_OVERLAP = int(SAMPLE_RATE * OVERLAP_DURATION)
SAMPLES_STEP = SAMPLES_PER_CHUNK - SAMPLES_OVERLAP # C'est le "pas" d'avancement (ce qu'on supprime du buffer)

# Variables globales
buffer_audio = np.array([], dtype=np.int16)
next_expected_ts = None
batch_counter = 0

# Création des dossiers
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)

# --- FONCTIONS ---

def read_binary_sample(filepath):
    """ Lit le fichier .bin brut (int16) en gérant les erreurs d'alignement """
    try:
        with open(filepath, "rb") as f:
            data = f.read()
            
        if len(data) % 2 != 0:
            # print(f"⚠️ Fichier {os.path.basename(filepath)} impair. Tronquage.")
            data = data[:-1] 
            
        return np.frombuffer(data, dtype='<i2')
        
    except Exception as e:
        print(f"❌ Erreur lecture {filepath}: {e}")
        return np.array([], dtype='<i2')

def parse_filename(filepath):
    try:
        base = os.path.basename(filepath).replace('.bin', '')
        parts = base.split('_')
        ts = int(parts[-1])
        return ts
    except:
        return None

def flush_buffer():
    """ 
    Gère la fenêtre glissante :
    1. Vérifie si on a assez de données pour faire une fenêtre (2s)
    2. Écrit le fichier
    3. Ne supprime que la partie 'vieux' (STEP) et garde l'overlap pour le prochain tour
    """
    global buffer_audio, batch_counter
    
    # Tant qu'on a assez de matière pour faire un fichier complet de 2s
    while len(buffer_audio) >= SAMPLES_PER_CHUNK:
        
        # 1. On COPIE la fenêtre de 2s (sans la retirer du buffer pour l'instant)
        chunk = buffer_audio[:SAMPLES_PER_CHUNK]
        
        # 2. Écriture
        # On nomme le dossier batch_X pour garder l'ordre chronologique
        folder_batch = os.path.join(OUTPUT_FOLDER, f"batch_{batch_counter}")
        if not os.path.exists(folder_batch): os.makedirs(folder_batch)
        
        filename = os.path.join(folder_batch, "mic_0.wav")
        write(filename, SAMPLE_RATE, chunk)
        
        # print(f"💾 Batch {batch_counter} écrit (2s).")
        
        # 3. AVANCEMENT (Sliding)
        # On supprime seulement le "Pas" (Step) du début du buffer.
        # Le reste (Overlap) devient le début du nouveau buffer.
        buffer_audio = buffer_audio[SAMPLES_STEP:]
        
        batch_counter += 1

class SingleMicHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.bin'): return
        time.sleep(0.2)
        self.ingest_file(event.src_path)

    def ingest_file(self, filepath):
        global buffer_audio, next_expected_ts

        ts_us = parse_filename(filepath)
        if ts_us is None: return

        new_data = read_binary_sample(filepath)
        if len(new_data) == 0: return

        if next_expected_ts is None:
            next_expected_ts = ts_us
            print(f"⏱️  Démarrage à {ts_us} us")

        # GESTION DES TROUS
        delta = ts_us - next_expected_ts
        if delta > 2000: 
            seconds_missing = delta / 1_000_000
            samples_missing = int(seconds_missing * SAMPLE_RATE)
            print(f"⚠️ Gap: {seconds_missing:.3f}s -> Silence.")
            silence = np.zeros(samples_missing, dtype=np.int16)
            buffer_audio = np.concatenate((buffer_audio, silence))
        
        buffer_audio = np.concatenate((buffer_audio, new_data))
        
        duration_us = (len(new_data) / SAMPLE_RATE) * 1_000_000
        next_expected_ts = ts_us + duration_us
        
        flush_buffer()

# --- MAIN ---

if __name__ == "__main__":
    print(f"👀  Surveillance {INPUT_FOLDER}...")
    print(f"⏱️  Config: Fichiers de {CHUNK_DURATION}s avec chevauchement de {OVERLAP_DURATION}s.")
    print(f"🚀  Un nouveau fichier sera généré toutes les {CHUNK_DURATION - OVERLAP_DURATION} secondes.")
    
    observer = Observer()
    observer.schedule(SingleMicHandler(), INPUT_FOLDER, recursive=False)
    observer.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()