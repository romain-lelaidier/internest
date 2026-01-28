import socket
from CONFIG import *

"""
Pour pouvoir stream lors de la démo, les différents audio en se connectant au même WIFI que la Raspberry
On dispatch chaque ESP sur un port différent pour pouvoir récuperer chaque flux de façon distincte

Ecouter le flux N°3 (Port 5003)
ffplay -f s8 -ar 48000 -ac 1 udp://@:5003
"""

PC_TARGET_IP = ""  # <--- On mettra l'IP d'un PC la dessus.
BASE_OUTPUT_PORT = 5000        # Le premier ESP sera sur 5001, le 2ème sur 5002...

def dispatch_streams():
    # Socket d'écoute (Reçoit les ESP)
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind(("", localPort)) # Port 8888 (config.py)
    
    # Socket d'envoi vers le PC
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Dictionnaire pour mémoriser quel IP correspond à quel port de sortie
    # Format : { 'IP': 5001, 'IP': 5002 }...
    known_sources = {}
    next_port = BASE_OUTPUT_PORT + 1
    
    print(f"Listening for ESPs on port {localPort}...")
    
    try:
        while True:
            # Réception (Buffer large pour éviter les pertes)
            data, addr = sock_in.recvfrom(4096)
            ip_source = addr[0]
            
            # Si c'est un nouvel ESP inconnu, on lui attribue un port de sortie
            if ip_source not in known_sources:
                known_sources[ip_source] = next_port
                print(f"New ESP detected [{ip_source}] -> Redirecting to PC port {next_port}")
                next_port += 1
            
            # Identification du port cible
            target_port = known_sources[ip_source]
            
            # NETTOYAGE (Pour écoute audio) ou PASS-THROUGH (Pour triangulation)
            # Pour écouter le son, on enlève la date (8 octets)
            if len(data) > BYTES_FOR_DATE:
                audio_payload = data[BYTES_FOR_DATE:]
                sock_out.sendto(audio_payload, (PC_TARGET_IP, target_port))
                
    except KeyboardInterrupt:
        print("\nArrêt du dispatcher.")
        print("Mapping final des ESPs :")
        for ip, port in known_sources.items():
            print(f"  ESP {ip} -> Port PC {port}")

if __name__ == "__main__":
    dispatch_streams()
