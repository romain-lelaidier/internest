import numpy as np
import os
import time
import struct
from scipy.io.wavfile import write
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
# Sur PC pour le test: './input_packets'
# Sur Pi: '/home/pi/udp_servers/packets'
INPUT_FOLDER = './input_packets' 
OUTPUT_FOLDER = './output_wavs'
SAMPLE_RATE = 48000

# PARAMÈTRES DE LA FENÊTRE GLISSANTE
CHUNK_DURATION = 2.0       
OVERLAP_DURATION = 1.0     

if OVERLAP_DURATION >= CHUNK_DURATION:
    print("❌ Erreur : L'overlap doit être strictement inférieur à la durée du chunk.")
    exit()

SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)
SAMPLES_OVERLAP = int(SAMPLE_RATE * OVERLAP_DURATION)
SAMPLES_STEP = SAMPLES_PER_CHUNK - SAMPLES_OVERLAP 

# --- ÉTAT GLOBAL ---
mic_buffers = {}       
mic_next_ts = {}       
mic_batch_counters = {} 
mac_to_id = {}         # Mapping Adresse MAC -> ID Micro (0, 1, 2...)

# --- FONCTIONS ---

def read_binary_sample(filepath):
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        if len(data) % 2 != 0:
            data = data[:-1] 
        return np.frombuffer(data, dtype='<i2')
    except Exception as e:
        print(f"❌ Erreur lecture {filepath}: {e}")
        return np.array([], dtype='<i2')

def parse_metadata(filepath):
    """ Parse le format {mac}_{timestamp}.bin """
    try:
        base = os.path.basename(filepath).replace('.bin', '')
        parts = base.split('_')
        if len(parts) < 2: return None, None
        
        mac = parts[0]
        ts = int(parts[1])
        return mac, ts
    except:
        return None, None

def get_or_assign_id(mac):
    """ Assigne un ID (0 à 4) à une adresse MAC par ordre de découverte """
    global mac_to_id
    if mac not in mac_to_id:
        new_id = len(mac_to_id)
        if new_id >= 5: # Limite à 5 micros selon notre config
            print(f"⚠️ Micro ignoré [{mac}] (Déjà 5 micros assignés)")
            return None
        mac_to_id[mac] = new_id
        print(f"🔗 Nouveau mapping : ESP [{mac}] -> Micro ID {new_id}")
    return mac_to_id[mac]

def flush_buffer(mic_id):
    global mic_buffers, mic_batch_counters
    
    buffer = mic_buffers[mic_id]
    counter = mic_batch_counters[mic_id]
    
    while len(buffer) >= SAMPLES_PER_CHUNK:
        chunk = buffer[:SAMPLES_PER_CHUNK]
        folder_batch = os.path.join(OUTPUT_FOLDER, f"batch_{counter}")
        os.makedirs(folder_batch, exist_ok=True)
        
        filename = os.path.join(folder_batch, f"mic_{mic_id}.wav")
        write(filename, SAMPLE_RATE, chunk)
        
        buffer = buffer[SAMPLES_STEP:]
        counter += 1
    
    mic_buffers[mic_id] = buffer
    mic_batch_counters[mic_id] = counter

class MultiMicHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.bin'): return
        # Petit délai pour s'assurer que l'OS a fini d'écrire le fichier
        time.sleep(0.05)
        self.ingest_file(event.src_path)

    def ingest_file(self, filepath):
        global mic_buffers, mic_next_ts, mic_batch_counters

        mac, ts_us = parse_metadata(filepath)
        if mac is None: return

        mic_id = get_or_assign_id(mac)
        if mic_id is None: return

        # Initialisation si nouveau micro
        if mic_id not in mic_buffers:
            mic_buffers[mic_id] = np.array([], dtype=np.int16)
            mic_next_ts[mic_id] = None
            mic_batch_counters[mic_id] = 0

        new_data = read_binary_sample(filepath)
        if len(new_data) == 0: return

        if mic_next_ts[mic_id] is None:
            mic_next_ts[mic_id] = ts_us
            print(f"⏱️  [Mic {mic_id}] Démarrage flux")

        # GESTION DES TROUS (Gap > 5ms)
        delta = ts_us - mic_next_ts[mic_id]
        if delta > 5000: 
            seconds_missing = delta / 1_000_000
            samples_missing = int(seconds_missing * SAMPLE_RATE)
            if samples_missing > 0:
                print(f"⚠️ [Mic {mic_id}] Gap: {seconds_missing:.3f}s -> Silence ajouté.")
                silence = np.zeros(samples_missing, dtype=np.int16)
                mic_buffers[mic_id] = np.concatenate((mic_buffers[mic_id], silence))
        
        mic_buffers[mic_id] = np.concatenate((mic_buffers[mic_id], new_data))
        
        duration_us = (len(new_data) / SAMPLE_RATE) * 1_000_000
        mic_next_ts[mic_id] = ts_us + duration_us
        
        flush_buffer(mic_id)

# --- MAIN ---

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)

    print(f"👀 Surveillance {INPUT_FOLDER}...")
    print(f"⏱️ Config: Fichiers de {CHUNK_DURATION}s (Overlap {OVERLAP_DURATION}s)")
    
    observer = Observer()
    observer.schedule(MultiMicHandler(), INPUT_FOLDER, recursive=True)
    observer.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()