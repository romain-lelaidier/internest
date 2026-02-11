"""
IHM unifiee pour postproc_2 : positions 3D + journal d'especes.

Serveur Flask sur le port 8010 avec :
  - Colonne gauche : Plotly 3D scatter (positions estimees + ESPs)
  - Colonne droite : especes actives + journal arrivees/departs

Template HTML : templates/ihm_postproc2.html

Usage depuis main.py :
    from ihm_postproc2 import start_ihm_postproc2
    start_ihm_postproc2()
"""

import time
import threading
from flask import Flask, jsonify, request as flask_request, render_template

import postproc_2 as pp2

app = Flask(__name__)

IHM_PORT = 8010


# --- Routes ---

@app.route('/')
def index():
    return render_template('ihm_postproc2.html')


@app.route('/api/positions')
def api_positions():
    offset = int(flask_request.args.get('offset', 0))
    mics = {}
    for mac, esp in pp2.ihm_esps.items():
        pos = esp.position
        if pos is not None and len(pos) >= 3:
            mics[mac] = [float(pos[0]), float(pos[1]), float(pos[2])]
    if offset == 0:
        print(f"[IHM] api/positions: {len(pp2.ihm_esps)} esps, {len(mics)} mics, {len(pp2.ihm_positions)} pos")
    return jsonify({
        'mics': mics,
        'positions': pp2.ihm_positions[offset:],
        'total': len(pp2.ihm_positions)
    })


@app.route('/api/species')
def api_species():
    now = time.time()
    active = []
    for sp, info in pp2.ihm_species.items():
        active.append({
            'species': sp,
            'confidence': info['confidence'],
            'ago': round(now - info['last_seen'], 1)
        })
    return jsonify({
        'active': active,
        'events': pp2.ihm_events[-100:]
    })


@app.route('/api/birdnet_status')
def api_birdnet_status():
    now = time.time()
    b = pp2.ihm_birdnet
    status = b['status']
    progress = 0.0
    remaining = 0.0
    if status == 'cooldown':
        remaining = max(0, b['cooldown_end'] - now)
        total = b['cooldown_total']
        progress = 1.0 - (remaining / total) if total > 0 else 1.0
        if remaining <= 0:
            status = 'idle'
            progress = 1.0
    elif status == 'analyzing':
        progress = -1  # indeterminate
    return jsonify({
        'status': status,
        'progress': round(progress, 3),
        'remaining': round(remaining, 1)
    })


@app.route('/api/clear', methods=['POST'])
def api_clear():
    target = flask_request.json.get('target', 'all') if flask_request.json else 'all'
    if target in ('positions', 'all'):
        pp2.ihm_positions.clear()
    if target in ('journal', 'all'):
        pp2.ihm_events.clear()
        pp2.ihm_species.clear()
    print(f"[IHM] clear: {target}")
    return jsonify({'ok': True})


def start_ihm_postproc2():
    """Lance le serveur Flask dans un thread daemon."""
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=IHM_PORT, debug=False),
        daemon=True
    )
    t.start()
    print(f"IHM PostProc2 demarree sur http://0.0.0.0:{IHM_PORT}")


if __name__ == '__main__':
    import math
    from types import SimpleNamespace

    # Simuler des ESPs
    fake_esps = {}
    coords = [[0,0,0], [0,5,0], [7,0,0], [6,5,2.3], [4,3,1]]
    for i, c in enumerate(coords):
        mac = f"aa:bb:cc:dd:ee:{i:02d}"
        esp = SimpleNamespace(position=c, mac=mac)
        fake_esps[mac] = esp

    pp2.ihm_esps = fake_esps

    # Simuler des positions
    now = time.time()
    for i in range(30):
        angle = i * 0.3
        pp2.ihm_positions.append({
            'x': 4 + 2 * math.cos(angle),
            'y': 3 + 2 * math.sin(angle),
            'z': 1 + i * 0.05,
            'time': now - 30 + i
        })

    # Simuler des especes
    pp2.ihm_species['Merle noir'] = {'confidence': 0.87, 'last_seen': now - 3}
    pp2.ihm_species['Mesange bleue'] = {'confidence': 0.72, 'last_seen': now - 12}
    pp2.ihm_events.append({'time': now - 60, 'type': 'arrivee', 'species': 'Pinson des arbres'})
    pp2.ihm_events.append({'time': now - 45, 'type': 'depart', 'species': 'Pinson des arbres'})
    pp2.ihm_events.append({'time': now - 20, 'type': 'arrivee', 'species': 'Merle noir'})
    pp2.ihm_events.append({'time': now - 12, 'type': 'arrivee', 'species': 'Mesange bleue'})

    print(f"Mode debug — http://localhost:{IHM_PORT}")
    app.run(host='0.0.0.0', port=IHM_PORT, debug=True, use_reloader=False)
