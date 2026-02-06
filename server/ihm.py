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
import threading
from collections import deque
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

IHM_PORT = 8008

# État partagé (thread-safe grâce au GIL pour les opérations atomiques)
active_species = {}   # { mac: { species_name: { confidence, last_seen } } }
events = deque(maxlen=100)


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

@app.route('/')
def index():
    now = time.time()
    return render_template_string(HTML, active=active_species, events=list(events)[:30], now=now)


@app.route('/api/status')
def api_status():
    return jsonify({
        'active': {
            mac: {sp: info for sp, info in species.items()}
            for mac, species in active_species.items()
        },
        'events': list(events)[:30]
    })


def start_ihm():
    """Lance le serveur Flask dans un thread daemon."""
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=IHM_PORT, debug=False),
        daemon=True
    )
    t.start()
    print(f"IHM demarree sur http://0.0.0.0:{IHM_PORT}")


# --- Template HTML ---

HTML = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <title>InterNest</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
        h1 { color: #00d4aa; }
        h2 { color: #7b8cde; margin-top: 24px; }
        .esp-block { background: #16213e; padding: 12px 16px; border-radius: 8px; margin: 8px 0; }
        .species { padding: 4px 0; }
        .confidence { color: #00d4aa; }
        .event { padding: 3px 0; font-size: 14px; }
        .arrivee { color: #4ade80; }
        .depart { color: #f87171; }
        .time { color: #888; }
        .empty { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>InterNest</h1>

    <h2>Especes actives</h2>
    {% if active %}
        {% for mac, species in active.items() %}
        <div class="esp-block">
            <strong>{{ mac }}</strong>
            {% if species %}
                {% for sp, info in species.items() %}
                <div class="species">
                    {{ sp }} — <span class="confidence">{{ "%.0f"|format(info.confidence * 100) }}%</span>
                </div>
                {% endfor %}
            {% else %}
                <div class="empty">aucune espece</div>
            {% endif %}
        </div>
        {% endfor %}
    {% else %}
        <p class="empty">aucun ESP connecte</p>
    {% endif %}

    <h2>Evenements recents</h2>
    {% if events %}
        {% for ev in events %}
        <div class="event">
            <span class="time">{{ "%.0f"|format(now - ev.time) }}s</span>
            {% if ev.type == 'arrivee' %}
                <span class="arrivee">[ARRIVEE]</span>
            {% else %}
                <span class="depart">[DEPART]</span>
            {% endif %}
            {{ ev.species }} sur <strong>{{ ev.mac }}</strong>
            {% if ev.confidence is defined and ev.type == 'arrivee' %}
                (<span class="confidence">{{ "%.0f"|format(ev.confidence * 100) }}%</span>)
            {% endif %}
        </div>
        {% endfor %}
    {% else %}
        <p class="empty">aucun evenement</p>
    {% endif %}
</body>
</html>
"""


if __name__ == '__main__':
    notify_arrival('aa:bb:cc:dd:ee:01', 'Merle noir', 0.92)
    notify_arrival('aa:bb:cc:dd:ee:01', 'Mesange bleue', 0.78)
    notify_arrival('aa:bb:cc:dd:ee:02', 'Rouge-gorge familier', 0.85)
    notify_departure('aa:bb:cc:dd:ee:02', 'Pinson des arbres')
    notify_arrival('aa:bb:cc:dd:ee:01', 'Tourterelle turque', 0.61)

    print(f"Mode debug — http://localhost:{IHM_PORT}")
    app.run(host='0.0.0.0', port=IHM_PORT, debug=True)
