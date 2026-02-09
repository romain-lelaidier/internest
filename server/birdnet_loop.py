"""
Module d'analyse en continu pour birdnet.

Comment ça marche ?

Un thread par ESP, lancé via start_birdnet(mac, esp) depuis main.py. Chaque thread :                                                              
                                                                                                                                                    
  1. Attend du nouvel audio : poll toutes les 0.5s, vérifie si buffer_end_t - last_analyzed_t >= CONFIG.BIRDNET_WINDOW_S (en µs)                           
  2. Analyse dès que la fenêtre est pleine : lit la fenêtre non analysée, avance last_analyzed_t, analyse avec 
  RecordingBuffer (pas de fichier WAV temporaire)                                                                                                                                       
  3. Gère arrivées/départs :
    - [ARRIVEE] quand une espèce apparaît pour la première fois
    - [DEPART] quand une espèce n'a pas été détectée depuis CONFIG.SPECIES_TIMEOUT_S (vérifié aussi pendant l'attente)
"""

import time
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer  # ça nous permet de ne pas passer par des wav temporaires.

from config import CONFIG
from ihm_birdnet import notify_esp, notify_arrival, notify_departure

# Chargement du modèle BirdNET (une seule fois à l'import)
print("Chargement du modele BirdNET...")
analyzer = Analyzer()
print("Modele BirdNET chargé.")

def _esp_loop(mac, esp):
    """Boucle d'analyse BirdNET dédiée à un ESP."""
    if CONFIG.AFFICHAGE_IHM:
        notify_esp(mac)
    active_species = {}     # species_name -> last_seen (time.time())
    last_analyzed_t = None  # fin de la dernière fenêtre analysée (µs, timeline ESP)

    while True:
        # pas encore de données
        if esp.buffer_end_t is None:
            time.sleep(CONFIG.POLL_INTERVAL_S)
            continue

        # initialisation : on part du buffer actuel, on attend du nouvel audio
        if last_analyzed_t is None:
            last_analyzed_t = esp.buffer_end_t
            time.sleep(CONFIG.POLL_INTERVAL_S)
            continue

        # pas assez de nouvel audio non analysé → on attend
        unanalyzed_us = esp.buffer_end_t - last_analyzed_t
        if unanalyzed_us < CONFIG.BIRDNET_WINDOW_S * 1e6:
            # vérifier les départs en attendant
            now = time.time()
            departed = [sp for sp, ts in active_species.items() if now - ts > CONFIG.SPECIES_TIMEOUT_S]
            for sp in departed:
                print(f"<<<Espèce {sp} partie de {mac} (dernière détection il y a {now - active_species[sp]:.1f}s)")
                if CONFIG.AFFICHAGE_IHM:
                    notify_departure(mac, sp)
                del active_species[sp]
            time.sleep(CONFIG.POLL_INTERVAL_S)
            continue

        # lire la fenêtre non analysée
        t1 = last_analyzed_t
        t2 = t1 + CONFIG.BIRDNET_WINDOW_S * 1e6
        _, _, samples = esp.read_window(t1, t2)
        samples = samples.copy()  # copie pour éviter les conflits TFLite / buffer partagé
        last_analyzed_t = t2

        if len(samples) < CONFIG.SAMPLE_RATE:
            continue

        # analyse BirdNET directement en mémoire
        now = time.time()
        try:
            recording = RecordingBuffer(
                analyzer, samples, CONFIG.SAMPLE_RATE,
                min_conf=CONFIG.BIRDNET_MIN_CONFIDENCE
            )
            print(f"Analyse BirdNET pour {mac} de {t1} à {t2} ({len(samples)} samples)")
            recording.analyze()

            for det in recording.detections:
                species = det['common_name']
                is_new = species not in active_species
                if is_new:
                    print(f">>> Espèce {species} arrivée sur {mac} (confidence {det['confidence']:.2f})")
                    if CONFIG.AFFICHAGE_IHM:
                        notify_arrival(mac, species, det['confidence'])
                else:
                    print(f"=== Espèce {species} toujours sur {mac} (confidence {det['confidence']:.2f})")
                active_species[species] = now
        except Exception as e:
            print(f"Erreur BirdNET pour {mac}: {e}")

        # vérifier les départs
        departed = [sp for sp, ts in active_species.items() if now - ts > CONFIG.SPECIES_TIMEOUT_S]
        for sp in departed:
            print(f"<<< Espèce {sp} partie de {mac} (dernière détection il y a {now - active_species[sp]:.1f}s)")
            if CONFIG.AFFICHAGE_IHM:
                notify_departure(mac, sp)
            del active_species[sp]
