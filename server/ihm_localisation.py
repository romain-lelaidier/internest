"""
IHM 3D pour la localisation en temps réel.

Serveur Flask sur le port 8009 avec un plot Plotly.js 3D.
Les positions sont alimentées par localisation.py via notify_position().

Usage depuis main.py :
    from ihm_localisation import start_ihm_localisation, set_mic_positions
    set_mic_positions(MIC_POSITIONS)
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
mic_positions = {}  # { id: [x, y, z] }


def set_mic_positions(mics):
    """Enregistre les positions des micros pour l'affichage."""
    global mic_positions
    mic_positions = {str(k): v for k, v in mics.items()}


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
    return jsonify({
        'mics': mic_positions,
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
    let micsInitialized = false;

    // trace 0 : micros (ajouté dynamiquement au premier fetch)
    // trace 1 : positions estimées
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

    Plotly.newPlot('chart', [traceBird], layout);

    function initMics(mics) {
        const ids = Object.keys(mics);
        if (ids.length === 0) return;

        const traceMicros = {
            x: ids.map(id => mics[id][0]),
            y: ids.map(id => mics[id][1]),
            z: ids.map(id => mics[id][2]),
            mode: 'markers+text',
            type: 'scatter3d',
            marker: { color: '#f87171', size: 5, symbol: 'diamond' },
            text: ids.map(id => 'M' + id),
            textposition: 'top center',
            textfont: { size: 10, color: '#f87171' },
            name: 'Micros'
        };

        Plotly.addTraces('chart', traceMicros);
        micsInitialized = true;
    }

    function refresh() {
        fetch('/api/positions?offset=' + offset)
            .then(r => r.json())
            .then(data => {
                // init micros une seule fois
                if (!micsInitialized) initMics(data.mics);

                const pts = data.positions;
                if (pts.length === 0) return;

                // temps relatif pour la colorbar (depuis le premier point)
                const t0 = (positions_t0 !== null) ? positions_t0 : pts[0].time;
                if (positions_t0 === null) positions_t0 = t0;

                Plotly.extendTraces('chart', {
                    x: [pts.map(p => p.x)],
                    y: [pts.map(p => p.y)],
                    z: [pts.map(p => p.z)],
                    'marker.color': [pts.map(p => (p.time - t0) / 1e6)]
                }, [0]);

                offset = data.total;

                // mise a jour status
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

    # positions micros fictives
    set_mic_positions({
        0: [0, 0, 0],
        1: [10, 0, 0],
        2: [0, 10, 0],
        3: [0, 0, 10],
        4: [10, 10, 10]
    })

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
