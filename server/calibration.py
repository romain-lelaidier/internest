import socket
import time
import numpy as np
import os

# --- IMPORTS LOCAUX ---
from esp import ESP
from utils import micros
from config import SAMPLE_RATE
from esp_controller import ESPController
from audio_analyzer import detect_buzz_in_audio
from position_calculator import PositionCalculator

# --- CONFIGURATION ---
PORT_AUDIO = 8002
POSITIONS_FILE = "mic_positions.csv"

# Configuration des paquets (identique à main.py)
ESP_ID_LENGTH = 6
ESP_TIME_LENGTH = 8
PACKET_LENGTH = 1024 * 8 * 3

def run_step_by_step_calibration(esps):
    """
    Fonction principale appelée par main.py au démarrage.
    Retourne un dictionnaire {MAC: [x, y, z]} ou None.
    """
    print("\n" + "="*50)
    print("   MODE CALIBRATION INTERACTIVE")
    print("="*50)
    
    macs_list = list(esps.keys())
    
    # 3. Initialisation des outils
    controller = ESPController() # Pour envoyer les Buzz
    all_tdoa = {} # Pour stocker les résultats

    # 4. Boucle de Tir
    for i, buzzer_mac in enumerate(macs_list):
        print(f"\n[{i+1}/{len(macs_list)}] Test Emetteur : {buzzer_mac}")
        
        # --- Interaction Utilisateur ---
        user_input = input(f"👉 Appuyez sur ENTRÉE pour faire buzzer {buzzer_mac} (ou 'q' pour quitter)... ")
        if user_input.lower() == 'q':
            # sock.close()
            return None

        # --- Action ---
        buzz_sent_time = controller.buzz(buzzer_mac)
        
        # --- Analyse ---
        print("   🔍 Analyse du signal...")
        all_tdoa[buzzer_mac] = {}
        
        # On définit la fenêtre d'analyse (les 2 dernières secondes reçues)
        t_now = micros()
        t_end_us = t_now
        t_start_us = t_end_us - (2.0 * 1e6)
        
        for receiver_mac in macs_list:
            # Extraction audio depuis l'objet ESP
            _, _, sig_int16 = esps[receiver_mac].read_window(t_start_us, t_end_us)
            
            if len(sig_int16) == 0: continue

            # Conversion Float pour l'analyseur
            sig_float = sig_int16.astype(np.float32) / 32768.0
            
            # Détection du Buzz
            sample_idx = detect_buzz_in_audio(sig_float, sample_rate=SAMPLE_RATE, threshold=0.25)
            
            if sample_idx is not None:
                # Calcul du temps absolu d'arrivée
                detection_time_abs = (t_start_us / 1e6) + (sample_idx / SAMPLE_RATE)
                
                all_tdoa[buzzer_mac][receiver_mac] = detection_time_abs
        t_ref = min(all_tdoa[buzzer_mac])
        for receiver_mac in macs_list:
            all_tdoa[buzzer_mac][receiver_mac] = all_tdoa[buzzer_mac][receiver_mac] - t_ref

    # 5. Calcul Final
    print("\n🧮 Calcul de la géométrie...")
    if len(all_tdoa) < 3:
        print("❌ Pas assez de données.")
        return None

    calc = PositionCalculator(macs_list)
    try:
        # Calcul MDS + Optimisation
        new_positions_raw = calc.calculate_positions(all_tdoa)
        calc.print_positions()
        
        # Sauvegarde
        try:
            if mac in esps:
                esps[mac].position = new_positions_raw[mac] 
                print(f"position sauvegardée à {new_positions_raw[mac]} pour l'esp {mac} dans la RAM")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde : {e}")

        return new_positions_raw

    except Exception as e:
        print(f"❌ Erreur Calcul Position: {e}")
        return None