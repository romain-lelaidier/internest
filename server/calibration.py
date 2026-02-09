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

def ingest_packet(esps, message):
    """ Décode un paquet UDP et remplit l'objet ESP correspondant """
    if len(message) != ESP_ID_LENGTH + ESP_TIME_LENGTH + PACKET_LENGTH:
        return

    # Décodage MAC
    mac_bytes = message[0 : ESP_ID_LENGTH]
    mac = ':'.join([f"{b:02x}" for b in mac_bytes]).upper()
    
    # Décodage Temps et Audio
    esp_time = int.from_bytes(message[ESP_ID_LENGTH : ESP_ID_LENGTH + ESP_TIME_LENGTH], 'little')
    samples = np.frombuffer(message[ESP_ID_LENGTH + ESP_TIME_LENGTH : ], dtype='<i2')

    # Création / Mise à jour ESP
    if mac not in esps:
        esps[mac] = ESP(mac)
    
    esps[mac].receive_packet(esp_time, samples)

def listen_and_buffer(sock, esps, duration_sec):
    """ Écoute le port UDP pendant X secondes et remplit les objets ESP """
    start = time.time()
    # On vide le buffer système pour ne pas traiter de vieux paquets
    sock.settimeout(0.001)
    while True:
        try: sock.recvfrom(40960)
        except: break
    
    sock.settimeout(0.1) # Timeout court pour la boucle
    
    while time.time() - start < duration_sec:
        try:
            message, _ = sock.recvfrom(ESP_ID_LENGTH + ESP_TIME_LENGTH + PACKET_LENGTH)
            ingest_packet(esps, message)
        except socket.timeout:
            continue
        except Exception:
            pass

def run_step_by_step_calibration():
    """
    Fonction principale appelée par main.py au démarrage.
    Retourne un dictionnaire {MAC: [x, y, z]} ou None.
    """
    print("\n" + "="*50)
    print("   MODE CALIBRATION INTERACTIVE")
    print("="*50)

    # 1. Initialisation du socket temporaire (le main n'a pas encore pris le port)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', PORT_AUDIO))
    except Exception as e:
        print(f"❌ Erreur socket calibration: {e}")
        return None

    # 2. Découverte rapide des micros connectés
    print("⏳ Découverte des micros (3s)...")
    esps = {} # Dictionnaire local temporaire
    listen_and_buffer(sock, esps, 3.0)
    
    macs_list = list(esps.keys())
    if len(macs_list) < 4:
        print(f"⚠️ Seulement {len(macs_list)} micros trouvés. Il en faut au moins 4 (idéalement 5+).")
        # On peut continuer pour tester, mais le calcul échouera probablement
    else:
        print(f"✅ {len(macs_list)} micros trouvés : {macs_list}")

    # 3. Initialisation des outils
    controller = ESPController() # Pour envoyer les Buzz
    all_tdoa = {} # Pour stocker les résultats

    # 4. Boucle de Tir
    for i, buzzer_mac in enumerate(macs_list):
        print(f"\n[{i+1}/{len(macs_list)}] Test Emetteur : {buzzer_mac}")
        
        # --- Interaction Utilisateur ---
        user_input = input(f"👉 Appuyez sur ENTRÉE pour faire buzzer {buzzer_mac} (ou 'q' pour quitter)... ")
        if user_input.lower() == 'q':
            sock.close()
            return None

        # --- Action ---
        buzz_sent_time = controller.buzz(buzzer_mac)
        
        # --- Capture ---
        print("   🎧 Écoute...")
        # On écoute 1.5s, c'est suffisant pour capturer le buzz et la réverb
        listen_and_buffer(sock, esps, 1.5)
        
        # --- Analyse ---
        print("   🔍 Analyse du signal...")
        all_tdoa[buzzer_mac] = {}
        
        # On définit la fenêtre d'analyse (les 2 dernières secondes reçues)
        t_now = micros()
        t_end_us = t_now
        t_start_us = t_end_us - (2.0 * 1e6)
        
        for receiver_mac in macs_list:
            if receiver_mac not in esps: continue
            
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
                
                # Time of Flight
                tof = detection_time_abs - buzz_sent_time
                
                # Filtrage basique (le son ne remonte pas le temps)
                if tof > 0.001: 
                    all_tdoa[buzzer_mac][receiver_mac] = detection_time_abs
                    print(f"     ✅ Reçu par {receiver_mac} (+{tof*1000:.1f}ms)")
                else:
                    print(f"     ❌ Reçu par {receiver_mac} (Temps incohérent)")

    sock.close() # On libère le port 8002 pour le programme principal

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
        
        # Sauvegarde CSV
        try:
            if mac in esps:
                esps[mac].position = pos 
                print(f"position sauvegardée pour l'esp {mac} dans la RAM")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde : {e}")

        return new_positions_raw

    except Exception as e:
        print(f"❌ Erreur Calcul Position: {e}")
        return None