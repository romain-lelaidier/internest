"""
Module d'analyse en continu pour birdnet.

Comment ça marche ?

Un thread par ESP, lancé via start_birdnet(mac, esp) depuis main.py. Chaque thread :                                                              
                                                                                                                                                    
  1. Attend du nouvel audio : poll toutes les 0.5s, vérifie si buffer_end_t - last_analyzed_t >= BIRDNET_WINDOW_S (en µs)                           
  2. Analyse dès que la fenêtre est pleine : lit la fenêtre non analysée, avance last_analyzed_t, analyse avec 
  RecordingBuffer (pas de fichier WAV temporaire)                                                                                                                                       
  3. Gère arrivées/départs :
    - [ARRIVEE] quand une espèce apparaît pour la première fois
    - [DEPART] quand une espèce n'a pas été détectée depuis SPECIES_TIMEOUT_S (vérifié aussi pendant l'attente)
"""

import time
import threading
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer  # ça nous permet de ne pas passer par des wav temporaires.

from esp import THEORETICAL_SAMPLE_RATE
from ihm import notify_arrival, notify_departure

BIRDNET_WINDOW_S = 5.0          # durée de la fenêtre d'analyse (secondes)
BIRDNET_MIN_CONFIDENCE = 0.5    # seuil de confiance minimum
SPECIES_TIMEOUT_S = 30.0        # durée sans détection → départ
POLL_INTERVAL_S = 0.5           # fréquence de vérification du buffer
AFFICHAGE_IHM = True            # si False, on ne notifie pas l'IHM

# Chargement du modèle BirdNET (une seule fois à l'import)
print("Chargement du modele BirdNET...")
analyzer = Analyzer()
print("Modele BirdNET charge.")

# Threads actifs par adresse MAC pour les ESPs.
_esp_threads = {}


def _esp_loop(mac, esp):
    """Boucle d'analyse BirdNET dédiée à un ESP."""
    active_species = {}     # species_name -> last_seen (time.time())
    last_analyzed_t = None  # fin de la dernière fenêtre analysée (µs, timeline ESP)

    while True:
        # pas encore de données
        if esp.buffer_end_t is None:
            time.sleep(POLL_INTERVAL_S)
            continue

        # initialisation : on part du buffer actuel, on attend du nouvel audio
        if last_analyzed_t is None:
            last_analyzed_t = esp.buffer_end_t
            time.sleep(POLL_INTERVAL_S)
            continue

        # pas assez de nouvel audio non analysé → on attend
        unanalyzed_us = esp.buffer_end_t - last_analyzed_t
        if unanalyzed_us < BIRDNET_WINDOW_S * 1e6:
            # vérifier les départs en attendant
            now = time.time()
            departed = [sp for sp, ts in active_species.items() if now - ts > SPECIES_TIMEOUT_S]
            for sp in departed:
                print(f"Espèce {sp} partie de {mac} (dernière détection il y a {now - active_species[sp]:.1f}s)")
                if AFFICHAGE_IHM:
                    notify_departure(mac, sp)
                del active_species[sp]
            time.sleep(POLL_INTERVAL_S)
            continue

        # lire la fenêtre non analysée
        t1 = last_analyzed_t
        t2 = t1 + BIRDNET_WINDOW_S * 1e6
        _, _, samples = esp.read_window(t1, t2)
        last_analyzed_t = t2

        if len(samples) < THEORETICAL_SAMPLE_RATE:
            continue

        # analyse BirdNET directement en mémoire
        now = time.time()
        try:
            recording = RecordingBuffer(
                analyzer, samples, THEORETICAL_SAMPLE_RATE,
                min_conf=BIRDNET_MIN_CONFIDENCE
            )
            print(f"Analyse BirdNET pour {mac} de {t1} à {t2} ({len(samples)} samples)")
            recording.analyze()


            for det in recording.detections:
                species = det['common_name']
                is_new = species not in active_species
                if is_new:
                    print(f"Espèce {species} arrivée sur {mac} (confidence {det['confidence']:.2f})")
                    if AFFICHAGE_IHM:
                        notify_arrival(mac, species, det['confidence'])
                else:
                    print(f"Espèce {species} toujours sur {mac} (confidence {det['confidence']:.2f})")
                active_species[species] = now
        except Exception as e:
            print(f"Erreur BirdNET pour {mac}: {e}")

        # vérifier les départs
        departed = [sp for sp, ts in active_species.items() if now - ts > SPECIES_TIMEOUT_S]
        for sp in departed:
            print(f"Espèce {sp} partie de {mac} (dernière détection il y a {now - active_species[sp]:.1f}s)")
            if AFFICHAGE_IHM:
                notify_departure(mac, sp)
            del active_species[sp]


def start_birdnet(mac, esp):
    """Démarre un thread BirdNET pour un ESP (si pas déjà actif)."""
    if mac in _esp_threads and _esp_threads[mac].is_alive():
        return
    t = threading.Thread(target=_esp_loop, args=(mac, esp), daemon=True)
    t.start()
    _esp_threads[mac] = t
    print(f"Analyse BirdNET demarree pour {mac}")
