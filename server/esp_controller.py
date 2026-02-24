"""
Contrôleur pour communiquer avec les ESPs via UDP.
Envoie la MAC de l'ESP cible en broadcast, seul celui-ci buzze.
"""

import socket
import time
import os
from typing import List

from config import UDP_PORT

# Fichier contenant les MACs des ESPs
MACS_FILE = os.path.join(os.path.dirname(__file__), "macs.txt")


def load_macs() -> List[str]:
    """Charge les adresses MAC depuis macs.txt"""
    macs = []
    if os.path.exists(MACS_FILE):
        with open(MACS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    macs.append(line.upper())
    return macs


class ESPController:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.macs = load_macs()

    def buzz(self, mac: str) -> float:
        """
        Envoie la MAC en broadcast. L'ESP avec cette MAC buzze.
        Retourne le timestamp de l'envoi.
        """
        timestamp = time.time()

        try:
            # Broadcast avec la MAC cible
            self.sock.sendto(mac.encode('utf-8'), ("255.255.255.255", UDP_PORT))
            print(f"[{timestamp:.6f}] Buzz -> {mac}")
        except socket.error as e:
            print(f"Erreur: {e}")

        return timestamp

    def close(self):
        self.sock.close()


def run_buzz_sequence(wait_between_s: float = 3.0) -> dict:
    """
    Fait buzzer chaque ESP séquentiellement.
    Retourne {mac: buzz_time}
    """
    controller = ESPController()
    buzz_timestamps = {}

    try:
        macs = controller.macs

        if not macs:
            print("ERREUR: Aucune MAC dans macs.txt")
            return {}

        print(f"=== Séquence de calibration ({len(macs)} ESPs) ===\n")

        for i, mac in enumerate(macs):
            print(f"[{i+1}/{len(macs)}] ", end="")
            buzz_time = controller.buzz(mac)
            buzz_timestamps[mac] = buzz_time

            time.sleep(wait_between_s)

        print("\n=== Terminé ===")

    finally:
        controller.close()

    return buzz_timestamps


if __name__ == "__main__":
    macs = load_macs()
    print(f"MACs chargées ({len(macs)}):")
    for mac in macs:
        print(f"  {mac}")

    if macs:
        print()
        response = input("Lancer la séquence ? (o/n) ")
        if response.lower() == 'o':
            timestamps = run_buzz_sequence()
            print("\nTimestamps:")
            for mac, ts in timestamps.items():
                print(f"  {mac}: {ts:.6f}")
