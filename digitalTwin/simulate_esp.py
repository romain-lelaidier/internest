import os
import time
import numpy as np
import librosa
import shutil

# CONFIGURATION
INPUT_AUDIO_FILES = [
    "sim_files/sim_mic_0.wav", 
    "sim_files/sim_mic_1.wav", 
    "sim_files/sim_mic_2.wav", 
    "sim_files/sim_mic_3.wav",
    "sim_files/sim_mic_4.wav"
]
TARGET_DIR = "./input_packets"
# Fausse adresses MAC pour la simulation
MACS = ["AA:BB:CC:DD:EE:00", "AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02", "AA:BB:CC:DD:EE:03", "AA:BB:CC:DD:EE:04"]

CHUNK_SIZE_SAMPLES = 4800  # 0.1 seconde par paquet
SAMPLE_RATE = 48000
SEND_INTERVAL = 0.1  # Simulation temps réel

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ Erreur nettoyage {file_path}: {e}")

def main():
    print("🚀 Préparation de la simulation ESP32 (Format MAC_TIMESTAMP)...")
    
    # 1. Chargement des audio
    signals = []
    for f in INPUT_AUDIO_FILES:
        if not os.path.exists(f):
            print(f"❌ Fichier introuvable : {f}")
            return
        y, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        y_int16 = (y * 32767).astype('<i2')
        signals.append(y_int16)

    # 2. Préparation des dossiers
    ensure_dir(TARGET_DIR)

    print(f"🟢 Démarrage de l'envoi de paquets...")

    # 3. Boucle d'envoi
    cursor = 0
    max_len = min([len(s) for s in signals])
    start_time_us = int(time.time() * 1_000_000)

    try:
        while cursor < max_len:
            current_ts_us = start_time_us + int((cursor / SAMPLE_RATE) * 1_000_000)
            
            for i, signal_arr in enumerate(signals):
                end = min(cursor + CHUNK_SIZE_SAMPLES, max_len)
                chunk = signal_arr[cursor:end]
                
                if len(chunk) == 0: continue
                
                # Nouveau format : {mac}_{timestamp}.bin
                filename = f"{MACS[i]}_{current_ts_us}.bin"
                filepath = os.path.join(TARGET_DIR, filename)
                
                with open(filepath, "wb") as f:
                    f.write(chunk.tobytes())
            
            # print(f"📤 Envoyé paquets à {current_ts_us} us")
            cursor += CHUNK_SIZE_SAMPLES
            time.sleep(SEND_INTERVAL)

    except KeyboardInterrupt:
        print("Arrêt de la simulation.")

if __name__ == "__main__":
    main()