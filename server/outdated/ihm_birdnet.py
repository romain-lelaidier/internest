"""
IHM légère pour la détection BirdNET.

Serveur Flask sur le port 8008 qui centralise les arrivées et départs
d'espèces pour chaque ESP.

Usage depuis main.py :
    from ihm import start_ihm
    start_ihm()

Usage depuis birdnet_loop.py :
    from ihm import notify_arrival, notify_departure

Debug standalone :
    python ihm.py
"""

import time
import io
import csv
import threading
from collections import deque
from flask import Flask, jsonify, request, Response

from config import CONFIG

app = Flask(__name__)

# État partagé (thread-safe grâce au GIL pour les opérations atomiques)
active_species = {}   # { mac: { species_name: { confidence, last_seen } } }
events = deque(maxlen=100)

def notify_esp(mac):
    """Appelé quand un ESP se connecte, pour qu'il apparaisse dans l'IHM même sans détection."""
    if mac not in active_species:
        active_species[mac] = {}


def notify_arrival(mac, species, confidence):
    """Appelé par birdnet_loop quand une espèce arrive."""
    if mac not in active_species:
        active_species[mac] = {}
    active_species[mac][species] = {
        'confidence': confidence,
        'last_seen': time.time()
    }
    events.appendleft({
        'type': 'arrivee',
        'mac': mac,
        'species': species,
        'confidence': confidence,
        'time': time.time()
    })

def notify_departure(mac, species):
    """Appelé par birdnet_loop quand une espèce repart."""
    if mac in active_species:
        active_species[mac].pop(species, None)
    events.appendleft({
        'type': 'depart',
        'mac': mac,
        'species': species,
        'time': time.time()
    })

# --- Routes ---
HTML = open("./index.html", "r").read()

@app.route('/')
def index():
    return HTML

@app.route('/api/status')
def api_status():
    return jsonify({
        'now': time.time(),
        'confidence': CONFIG.BIRDNET_MIN_CONFIDENCE,
        'active': {
            mac: {sp: info for sp, info in species.items()}
            for mac, species in active_species.items()
        },
        'events': list(events)[:30]
    })

@app.route('/api/set_confidence', methods=['POST'])
def api_set_confidence():
    data = request.get_json()
    val = float(data.get('value', CONFIG.BIRDNET_MIN_CONFIDENCE))
    CONFIG.BIRDNET_MIN_CONFIDENCE = val
    print(f"Seuil BirdNET mis a jour: {val}")
    return jsonify({'ok': True, 'value': val})


@app.route('/api/clear_events', methods=['POST'])
def api_clear_events():
    events.clear()
    return jsonify({'ok': True})


@app.route('/api/export_csv')
def api_export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['timestamp', 'type', 'mac', 'species', 'confidence'])
    for ev in events:
        writer.writerow([
            ev.get('time', ''),
            ev.get('type', ''),
            ev.get('mac', ''),
            ev.get('species', ''),
            ev.get('confidence', '')
        ])
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=detections.csv'}
    )


def start_ihm():
    """Lance le serveur Flask dans un thread daemon."""
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=CONFIG.PORT_IHM, debug=False),
        daemon=True
    )
    t.start()
    print(f"IHM demarree sur http://0.0.0.0:{CONFIG.PORT_IHM}")

if __name__ == '__main__':
    notify_arrival('aa:bb:cc:dd:ee:01', 'Merle noir', 0.92)
    notify_arrival('aa:bb:cc:dd:ee:01', 'Mesange bleue', 0.78)
    notify_arrival('aa:bb:cc:dd:ee:02', 'Rouge-gorge familier', 0.85)
    notify_departure('aa:bb:cc:dd:ee:02', 'Pinson des arbres')
    notify_arrival('aa:bb:cc:dd:ee:01', 'Tourterelle turque', 0.61)

    print(f"Mode debug — http://localhost:{CONFIG.PORT_IHM}")
    app.run(host='0.0.0.0', port=CONFIG.PORT_IHM, debug=True)
