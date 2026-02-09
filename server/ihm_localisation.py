"""
IHM 3D pour la localisation en temps réel.

Serveur Flask sur le port 8009 avec un plot Plotly.js 3D.
Les positions sont alimentées par localisation.py via notify_position().

Usage depuis main.py :
    from ihm_localisation import start_ihm_localisation, set_esps
    set_esps(esps)
    start_ihm_localisation()

Usage depuis localisation.py :
    from ihm_localisation import notify_position

Debug standalone :
    python ihm_localisation.py
"""

import time
import threading
from flask import Flask, jsonify, request

app = Flask(__name__)

IHM_LOC_PORT = 8009

# État partagé
positions = []      # [{ x, y, z, cost, time }]
_esps = {}          # référence vers le dict esps de main.py


def set_esps(esps):
    """Enregistre la référence vers le dict esps pour lire les coordonnées."""
    global _esps
    _esps = esps


def notify_position(x, y, z, cost, t):
    """Appelé par localisation.py quand une position est calculée."""
    positions.append({
        'x': float(x), 'y': float(y), 'z': float(z),
        'cost': float(cost),
        'time': float(t)
    })


# --- Routes ---

@app.route('/')
def index():
    return HTML


@app.route('/api/positions')
def api_positions():
    offset = int(request.args.get('offset', 0))
    # construire les positions micros dynamiquement depuis les ESP
    mics = {}
    for mac, esp in _esps.items():
        if esp.coordinates is not None:
            mics[mac] = esp.coordinates
    return jsonify({
        'mics': mics,
        'positions': positions[offset:],
        'total': len(positions)
    })


def start_ihm_localisation():
    """Lance le serveur Flask dans un thread daemon."""
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=IHM_LOC_PORT, debug=False),
        daemon=True
    )
    t.start()
    print(f"IHM Localisation demarree sur http://0.0.0.0:{IHM_LOC_PORT}")


# --- Template HTML ---

HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>InterNest - Localisation 3D</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { margin: 0; padding: 0; font-family: monospace; background: #1a1a2e; color: #e0e0e0; }
        #chart { width: 100vw; height: 100vh; }
        #status {
            position: absolute; top: 10px; left: 10px;
            background: rgba(22, 33, 62, 0.9); padding: 12px 16px;
            border-radius: 8px; z-index: 100;
        }
        #status h2 { margin: 0 0 8px 0; color: #00d4aa; font-size: 16px; }
        #status p { margin: 4px 0; font-size: 13px; }
        .val { color: #00d4aa; }
    </style>
</head>
<body>

    <div id="status">
        <h2>InterNest - Localisation 3D</h2>
        <p>Points: <span class="val" id="nb-points">0</span></p>
        <p>Derniere position: <span class="val" id="last-pos">-</span></p>
    </div>

    <div id="chart"></div>

    <script>
    let offset = 0;
    let nbMics = 0;

    // trace 0 : positions estimées
    // trace 1 : micros (ESP) — mis à jour dynamiquement
    const traceBird = {
        x: [], y: [], z: [],
        mode: 'markers',
        type: 'scatter3d',
        marker: {
            color: [],
            size: 4,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                thickness: 8, len: 0.3, x: 0.95, y: 0.5,
                title: { text: 'temps (s)', side: 'top', font: { size: 10, color: '#ccc' } },
                tickfont: { size: 10, color: '#ccc' }
            }
        },
        name: 'Positions'
    };

    const traceMicros = {
        x: [], y: [], z: [],
        mode: 'markers+text',
        type: 'scatter3d',
        marker: { color: '#f87171', size: 5, symbol: 'diamond' },
        text: [],
        textposition: 'top center',
        textfont: { size: 10, color: '#f87171' },
        name: 'ESPs'
    };

    const layout = {
        scene: {
            xaxis: { range: [0, 15], title: 'X (m)' },
            yaxis: { range: [0, 15], title: 'Y (m)' },
            zaxis: { range: [0, 15], title: 'Z (m)' },
            camera: { eye: { x: 1.5, y: 1.5, z: 1.0 } }
        },
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#1a1a2e',
        font: { color: '#ccc' },
        showlegend: true,
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' }
    };

    // trace 0 = positions, trace 1 = micros
    Plotly.newPlot('chart', [traceBird, traceMicros], layout);

    function updateMics(mics) {
        const macs = Object.keys(mics);
        if (macs.length === nbMics) return;
        nbMics = macs.length;

        Plotly.restyle('chart', {
            x: [macs.map(mac => mics[mac][0])],
            y: [macs.map(mac => mics[mac][1])],
            z: [macs.map(mac => mics[mac][2])],
            text: [macs]
        }, [1]);
    }

    function refresh() {
        fetch('/api/positions?offset=' + offset)
            .then(r => r.json())
            .then(data => {
                // mettre à jour les micros si nouveaux ESP
                updateMics(data.mics);

                const pts = data.positions;
                if (pts.length === 0) return;

                // temps relatif pour la colorbar
                const t0 = (positions_t0 !== null) ? positions_t0 : pts[0].time;
                if (positions_t0 === null) positions_t0 = t0;

                Plotly.extendTraces('chart', {
                    x: [pts.map(p => p.x)],
                    y: [pts.map(p => p.y)],
                    z: [pts.map(p => p.z)],
                    'marker.color': [pts.map(p => (p.time - t0) / 1e6)]
                }, [0]);

                offset = data.total;

                const last = pts[pts.length - 1];
                document.getElementById('nb-points').innerText = data.total;
                document.getElementById('last-pos').innerText =
                    '[' + last.x.toFixed(1) + ', ' + last.y.toFixed(1) + ', ' + last.z.toFixed(1) + ']' +
                    ' (cout: ' + last.cost.toFixed(2) + ')';
            });
    }

    let positions_t0 = null;

    refresh();
    setInterval(refresh, 1000);
    </script>
</body>
</html>"""


if __name__ == '__main__':
    import math
    from types import SimpleNamespace

    # simuler des ESP avec coordonnées
    fake_esps = {}
    coords = [[0,0,0], [10,0,0], [0,10,0], [0,0,10], [10,10,10]]
    for i, c in enumerate(coords):
        mac = f"aa:bb:cc:dd:ee:{i:02d}"
        esp = SimpleNamespace(coordinates=c, mac=mac, id=i)
        fake_esps[mac] = esp

    set_esps(fake_esps)

    # simuler des positions en spirale
    t0 = time.time() * 1e6
    for i in range(50):
        angle = i * 0.3
        notify_position(
            7 + 3 * math.cos(angle),
            7 + 3 * math.sin(angle),
            2 + i * 0.1,
            5.0 + i * 0.2,
            t0 + i * 1.5e6
        )

    print(f"Mode debug — http://localhost:{IHM_LOC_PORT}")
    app.run(host='0.0.0.0', port=IHM_LOC_PORT, debug=True)
