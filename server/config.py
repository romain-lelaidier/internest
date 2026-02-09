class CONFIG:
    # GENERAL
    PORT_AUDIO = 8002               # port d'écoute UDP pour les paquets audio
    PORT_SYNC = 8001                # port d'écoute UDP pour les paquets audio
    PORT_BUZZER = 8003              # port d'écoute UDP pour les paquets audio
    PORT_IHM = 8008                 # port d'écoute HTTP pour l'affichage web

    # ESPS
    ESP_ID_LENGTH = 6               # nombre d'octets pour identifier un ESP
    ESP_TIME_LENGTH = 8             # nombre d'octets pour le timestamp d'un paquet
    PACKET_LENGTH = 1024*8*3        # nombre d'octets de data d'un paquet (sans les metadata)
    SAMPLE_RATE = 48000             # sample rate théorique des ESP
    MAX_WINDOW_S = 20               # durée maximale d'enregistrement (au-delà, on oublie les paquets)

    # BIRDNET_LOOP
    BIRDNET_WINDOW_S = 5.0          # durée de la fenêtre d'analyse (secondes)
    BIRDNET_MIN_CONFIDENCE = 0.5    # seuil de confiance minimum
    SPECIES_TIMEOUT_S = 30.0        # durée sans détection → départ
    POLL_INTERVAL_S = 0.5           # fréquence de vérification du buffer
    AFFICHAGE_IHM = True            # si False, on ne notifie pas l'IHM

    # LOCALIZATION LOOP
    COMPUTE_INTERVAL_US = 1.5 * 1e6   # 1.5 secondes
    WINDOW_SIZE_VAD_US = 2.0 * 1e6    # Fenêtre large pour la détection (2s)
    WINDOW_SIZE_TDOA_US = 0.3 * 1e6   # Fenêtre courte pour le TDOA (0.3s)
    BUFFER_DELAY_US = 0.5 * 1e6       # On regarde 0.5s en arrière (pour être sûr que les paquets sont arrivés)
