"""
viz_server_v2.py — IHM v2 : positions 3D + espèces BirdNET + journal.

Deux colonnes :
  - Gauche : Plotly 3D scatter (positions estimées + micros + vérité terrain)
  - Droite : barre BirdNET + espèces actives + journal arrivées/départs

Communique avec triangulation_mvt_stream_v3.py via :
  - live_positions.csv   (positions, lu en tail)
  - live_species.json    (espèces, lu en polling)

Port : 8080
"""

from flask import Flask, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO
import os
import json
import threading
import time

# --- CONFIG ---
HTTP_PORT = 8080
CSV_FILE = "live_positions.csv"
SPECIES_FILE = "live_species.json"
SIM_FILES_DIR = "sim_files"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# --- Tail CSV (positions) → socket.io ---
def tail_csv():
    print(f"[IHM v2] Surveillance de {CSV_FILE}...")
    last_size = 0
    while True:
        if not os.path.exists(CSV_FILE):
            socketio.sleep(0.5)
            last_size = 0
            continue

        current_size = os.path.getsize(CSV_FILE)
        if current_size < last_size:
            last_size = 0

        if current_size > last_size:
            with open(CSV_FILE, "r") as f:
                f.seek(last_size)
                lines = f.readlines()
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if parts[0] == "Batch_ID":
                        continue
                    try:
                        data = {
                            "id": int(parts[0]),
                            "t": float(parts[1]),
                            "x": float(parts[2]),
                            "y": float(parts[3]),
                            "z": float(parts[4]),
                            "cost": float(parts[5]),
                            "bird_id": int(parts[6]) if len(parts) > 6 else 0
                        }
                        socketio.emit('new_position', data)
                    except Exception as e:
                        print(f"[IHM v2] Erreur parsing : {e}")
                last_size = f.tell()
        socketio.sleep(0.1)


# --- Routes ---
@app.route('/')
def index():
    return render_template('index_v2.html')


@app.route('/ground_truth')
def get_ground_truth():
    return send_from_directory(SIM_FILES_DIR, 'ground_truth.json')


@app.route('/api/species')
def api_species():
    """Lit live_species.json produit par triangulation_mvt_stream_v3.py."""
    if not os.path.exists(SPECIES_FILE):
        return jsonify({'active': [], 'events': [], 'birdnet': {'status': 'idle'}})
    try:
        with open(SPECIES_FILE) as f:
            data = json.load(f)
        # Recalculer le status cooldown côté serveur
        birdnet = data.get('birdnet', {})
        now = time.time()
        if birdnet.get('status') == 'cooldown':
            remaining = max(0, birdnet.get('cooldown_end', 0) - now)
            total = birdnet.get('cooldown_total', 5.0)
            progress = 1.0 - (remaining / total) if total > 0 else 1.0
            if remaining <= 0:
                birdnet['status'] = 'idle'
                progress = 1.0
                remaining = 0
            birdnet['progress'] = round(progress, 3)
            birdnet['remaining'] = round(remaining, 1)
        elif birdnet.get('status') == 'analyzing':
            birdnet['progress'] = -1
            birdnet['remaining'] = 0
        else:
            birdnet['progress'] = 1.0
            birdnet['remaining'] = 0

        data['birdnet'] = birdnet
        return jsonify(data)
    except Exception:
        return jsonify({'active': [], 'events': [], 'birdnet': {'status': 'idle'}})


if __name__ == '__main__':
    t = threading.Thread(target=tail_csv, daemon=True)
    t.start()

    print(f"[IHM v2] http://localhost:{HTTP_PORT}")
    socketio.run(app, host='0.0.0.0', port=HTTP_PORT, debug=False, allow_unsafe_werkzeug=True)
