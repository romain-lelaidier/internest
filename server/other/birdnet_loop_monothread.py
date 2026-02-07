"""
Module d'analyse en continu pour birdnet (version monothread).

Un seul thread BirdNET qui round-robin sur tous les ESP enregistrés.
Une seule analyse à la fois.
"""

import time
import threading
from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer

from esp import THEORETICAL_SAMPLE_RATE
from server.ihm_birdnet import notify_arrival, notify_departure

BIRDNET_WINDOW_S = 5.0         # durée de la fenêtre d'analyse (secondes)
BIRDNET_MIN_CONFIDENCE = 0.5    # seuil de confiance minimum
SPECIES_TIMEOUT_S = 30.0        # durée sans détection → départ
POLL_INTERVAL_S = 0.5           # fréquence de vérification entre chaque round-robin
AFFICHAGE_IHM = False            # si False, on ne notifie pas l'IHM

# Chargement du modèle BirdNET (une seule fois à l'import)
print("Chargement du modele BirdNET...")
analyzer = Analyzer()
print("Modele BirdNET charge.")

# ESP enregistrés et état par ESP
_esps = {}                      # { mac: esp }
_esp_state = {}                 # { mac: { last_analyzed_t, active_species } }
_thread = None


def register_esp(mac, esp):
    """Enregistre un ESP pour l'analyse BirdNET."""
    if mac not in _esps:
        _esps[mac] = esp
        _esp_state[mac] = {
            'last_analyzed_t': None,
            'active_species': {}     # species_name -> last_seen (time.time())
        }
        print(f"ESP {mac} enregistre pour BirdNET")


def _check_departures(mac, active_species):
    """Vérifie et traite les départs d'espèces pour un ESP."""
    now = time.time()
    departed = [sp for sp, ts in active_species.items() if now - ts > SPECIES_TIMEOUT_S]
    for sp in departed:
        print(f"Espece {sp} partie de {mac} (derniere detection il y a {now - active_species[sp]:.1f}s)")
        if AFFICHAGE_IHM:
            notify_departure(mac, sp)
        del active_species[sp]


def _try_analyze(mac, esp, state):
    """Tente une analyse BirdNET pour un ESP. Retourne True si une analyse a été faite."""
    active_species = state['active_species']

    # pas encore de données
    if esp.buffer_end_t is None:
        return False

    # initialisation
    if state['last_analyzed_t'] is None:
        state['last_analyzed_t'] = esp.buffer_end_t
        return False

    # vérifier les départs dans tous les cas
    _check_departures(mac, active_species)

    # pas assez de nouvel audio non analysé
    unanalyzed_us = esp.buffer_end_t - state['last_analyzed_t']
    if unanalyzed_us < BIRDNET_WINDOW_S * 1e6:
        return False

    # lire la fenêtre non analysée
    t1 = state['last_analyzed_t']
    t2 = t1 + BIRDNET_WINDOW_S * 1e6
    _, _, samples = esp.read_window(t1, t2)
    state['last_analyzed_t'] = t2

    if len(samples) < THEORETICAL_SAMPLE_RATE:
        return False

    # analyse BirdNET
    now = time.time()
    try:
        recording = RecordingBuffer(
            analyzer, samples, THEORETICAL_SAMPLE_RATE,
            min_conf=BIRDNET_MIN_CONFIDENCE
        )
        print(f"Analyse BirdNET pour {mac} de {t1} a {t2} ({len(samples)} samples)")
        recording.analyze()

        for det in recording.detections:
            species = det['common_name']
            is_new = species not in active_species
            if is_new:
                print(f"Espece {species} arrivee sur {mac} (confidence {det['confidence']:.2f})")
                if AFFICHAGE_IHM:
                    notify_arrival(mac, species, det['confidence'])
            else:
                print(f"Espece {species} toujours sur {mac} (confidence {det['confidence']:.2f})")
            active_species[species] = now
    except Exception as e:
        print(f"Erreur BirdNET pour {mac}: {e}")

    # vérifier les départs après analyse
    _check_departures(mac, active_species)
    return True


def _main_loop():
    """Boucle principale : round-robin sur tous les ESP."""
    while True:
        if not _esps:
            time.sleep(POLL_INTERVAL_S)
            continue

        did_work = False
        for mac in list(_esps.keys()):
            esp = _esps[mac]
            state = _esp_state[mac]
            if _try_analyze(mac, esp, state):
                did_work = True

        # si aucun ESP n'avait assez d'audio, on attend un peu
        if not did_work:
            time.sleep(POLL_INTERVAL_S)


def start_birdnet():
    """Lance le thread unique d'analyse BirdNET."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_main_loop, daemon=True)
    _thread.start()
    print("Thread BirdNET monothread demarre")
