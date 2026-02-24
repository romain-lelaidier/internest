"""
Minimal web interface for InterNest bird detection.

Provides a Flask server with SSE (Server-Sent Events) for real-time
updates that can be accessed from any device on the same network.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Generator
from collections import deque
from queue import Queue, Empty

from flask import Flask, Response, render_template_string


# HTML template for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InterNest - Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #00d9ff;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .esp-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .esp-card {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            min-width: 280px;
            max-width: 350px;
            flex: 1;
            border: 1px solid #0f3460;
        }
        .esp-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 8px;
        }
        .esp-title {
            font-weight: bold;
            color: #00d9ff;
            font-size: 1.1em;
        }
        .esp-mac {
            font-size: 0.75em;
            color: #888;
        }
        .buffer-container {
            margin-bottom: 12px;
        }
        .buffer-bar {
            height: 20px;
            background: #0f3460;
            border-radius: 10px;
            overflow: hidden;
        }
        .buffer-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .buffer-text {
            font-size: 0.8em;
            color: #888;
            margin-top: 4px;
        }
        .birds-section {
            margin-bottom: 12px;
        }
        .birds-title {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 6px;
        }
        .bird-item {
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            background: #0f3460;
            border-radius: 5px;
            margin-bottom: 4px;
        }
        .bird-name {
            color: #00ff88;
        }
        .bird-confidence {
            color: #00d9ff;
        }
        .no-birds {
            color: #555;
            font-style: italic;
            font-size: 0.9em;
        }
        .events-section {
            max-height: 120px;
            overflow-y: auto;
        }
        .event-item {
            font-size: 0.85em;
            padding: 4px 0;
            border-bottom: 1px solid #0f3460;
        }
        .event-arrival {
            color: #00ff88;
        }
        .event-departure {
            color: #ff6b6b;
        }
        .event-detection {
            color: #ffd93d;
        }
        .event-time {
            color: #666;
            margin-right: 8px;
        }
        .stats-panel {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            border: 1px solid #0f3460;
        }
        .stats-title {
            font-weight: bold;
            color: #00d9ff;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #00d9ff;
        }
        .stat-label {
            font-size: 0.8em;
            color: #888;
        }
        .event-log {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid #0f3460;
            max-height: 200px;
            overflow-y: auto;
        }
        .log-title {
            font-weight: bold;
            color: #00d9ff;
            margin-bottom: 10px;
        }
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-active {
            background: #00ff88;
            box-shadow: 0 0 8px #00ff88;
        }
        .status-inactive {
            background: #555;
        }
    </style>
</head>
<body>
    <h1>InterNest - Detection Multi-ESP</h1>

    <div class="esp-grid" id="esp-grid">
        <div class="esp-card">
            <div style="text-align: center; color: #555; padding: 30px;">
                En attente de donnees ESP...
            </div>
        </div>
    </div>

    <div class="stats-panel">
        <div class="stats-title">Statistiques</div>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value" id="stat-esps">0</div>
                <div class="stat-label">ESP actifs</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-detections">0</div>
                <div class="stat-label">Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-chunks">0</div>
                <div class="stat-label">Chunks analyses</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-elapsed">0s</div>
                <div class="stat-label">Duree</div>
            </div>
        </div>
    </div>

    <div class="event-log">
        <div class="log-title">Evenements recents</div>
        <div id="event-log"></div>
    </div>

    <script>
        // State
        const state = {
            esps: {},
            totalDetections: 0,
            totalChunks: 0,
            startTime: Date.now(),
            events: []
        };

        // Connect to SSE stream
        const evtSource = new EventSource('/stream');

        evtSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleEvent(data);
        };

        evtSource.onerror = function() {
            console.log('SSE connection error, reconnecting...');
        };

        function handleEvent(data) {
            switch(data.type) {
                case 'new_esp':
                    addEsp(data.esp_id, data.mac_address);
                    break;
                case 'buffer_update':
                    updateBuffer(data.esp_id, data.buffer_seconds, data.files_received);
                    break;
                case 'chunk_analyzed':
                    onChunkAnalyzed(data.esp_id);
                    break;
                case 'detection':
                    onDetection(data.esp_id, data.species, data.confidence);
                    break;
                case 'arrival':
                    onArrival(data.esp_id, data.species, data.confidence);
                    break;
                case 'departure':
                    onDeparture(data.esp_id, data.species, data.duration);
                    break;
            }
            updateStats();
        }

        function addEsp(espId, macAddress) {
            if (state.esps[espId]) return;

            state.esps[espId] = {
                mac: macAddress,
                bufferSeconds: 0,
                filesReceived: 0,
                chunksAnalyzed: 0,
                birds: {},
                events: []
            };

            renderEspGrid();
            addEventLog(`ESP ${espId} connecte (${macAddress})`, 'arrival');
        }

        function updateBuffer(espId, bufferSeconds, filesReceived) {
            if (!state.esps[espId]) addEsp(espId, '');

            state.esps[espId].bufferSeconds = bufferSeconds;
            state.esps[espId].filesReceived = filesReceived;

            renderEspGrid();
        }

        function onChunkAnalyzed(espId) {
            if (!state.esps[espId]) return;

            state.esps[espId].chunksAnalyzed++;
            state.totalChunks++;

            renderEspGrid();
        }

        function onDetection(espId, species, confidence) {
            if (!state.esps[espId]) return;

            state.esps[espId].birds[species] = confidence;
            state.totalDetections++;

            const esp = state.esps[espId];
            esp.events.unshift({
                type: 'detection',
                species: species,
                confidence: confidence,
                time: new Date()
            });
            if (esp.events.length > 10) esp.events.pop();

            renderEspGrid();
        }

        function onArrival(espId, species, confidence) {
            if (!state.esps[espId]) return;

            state.esps[espId].birds[species] = confidence;

            const esp = state.esps[espId];
            esp.events.unshift({
                type: 'arrival',
                species: species,
                confidence: confidence,
                time: new Date()
            });
            if (esp.events.length > 10) esp.events.pop();

            addEventLog(`ESP ${espId}: ${species} (${(confidence*100).toFixed(0)}%)`, 'arrival');
            renderEspGrid();
        }

        function onDeparture(espId, species, duration) {
            if (!state.esps[espId]) return;

            delete state.esps[espId].birds[species];

            const esp = state.esps[espId];
            esp.events.unshift({
                type: 'departure',
                species: species,
                duration: duration,
                time: new Date()
            });
            if (esp.events.length > 10) esp.events.pop();

            addEventLog(`ESP ${espId}: ${species} parti (${duration.toFixed(0)}s)`, 'departure');
            renderEspGrid();
        }

        function addEventLog(message, type) {
            state.events.unshift({
                message: message,
                type: type,
                time: new Date()
            });
            if (state.events.length > 20) state.events.pop();

            renderEventLog();
        }

        function renderEspGrid() {
            const grid = document.getElementById('esp-grid');
            const espIds = Object.keys(state.esps).sort((a, b) => a - b);

            if (espIds.length === 0) return;

            let html = '';
            for (const espId of espIds) {
                const esp = state.esps[espId];
                const progress = Math.min(esp.bufferSeconds / 15.0, 1.0) * 100;

                let birdsHtml = '';
                const birdNames = Object.keys(esp.birds);
                if (birdNames.length > 0) {
                    birdsHtml = birdNames.map(name => `
                        <div class="bird-item">
                            <span class="bird-name">${name.substring(0, 25)}</span>
                            <span class="bird-confidence">${(esp.birds[name]*100).toFixed(0)}%</span>
                        </div>
                    `).join('');
                } else {
                    birdsHtml = '<div class="no-birds">Aucun oiseau detecte</div>';
                }

                let eventsHtml = '';
                if (esp.events.length > 0) {
                    eventsHtml = esp.events.slice(0, 5).map(e => {
                        const timeStr = e.time.toLocaleTimeString('fr-FR');
                        let cls = 'event-detection';
                        let text = '';
                        if (e.type === 'arrival') {
                            cls = 'event-arrival';
                            text = `&#9654; ${e.species.substring(0, 20)}`;
                        } else if (e.type === 'departure') {
                            cls = 'event-departure';
                            text = `&#9664; ${e.species.substring(0, 20)}`;
                        } else {
                            text = `&#128038; ${e.species.substring(0, 20)}`;
                        }
                        return `<div class="event-item ${cls}">
                            <span class="event-time">${timeStr}</span>${text}
                        </div>`;
                    }).join('');
                }

                html += `
                    <div class="esp-card">
                        <div class="esp-header">
                            <div>
                                <span class="status-indicator status-active"></span>
                                <span class="esp-title">ESP ${espId}</span>
                            </div>
                            <span class="esp-mac">${esp.mac || ''}</span>
                        </div>
                        <div class="buffer-container">
                            <div class="buffer-bar">
                                <div class="buffer-fill" style="width: ${progress}%"></div>
                            </div>
                            <div class="buffer-text">
                                ${esp.bufferSeconds.toFixed(1)}s / 15s | ${esp.filesReceived} fichiers | ${esp.chunksAnalyzed} chunks
                            </div>
                        </div>
                        <div class="birds-section">
                            <div class="birds-title">Oiseaux detectes</div>
                            ${birdsHtml}
                        </div>
                        <div class="events-section">
                            ${eventsHtml}
                        </div>
                    </div>
                `;
            }

            grid.innerHTML = html;
        }

        function renderEventLog() {
            const log = document.getElementById('event-log');

            log.innerHTML = state.events.map(e => {
                const timeStr = e.time.toLocaleTimeString('fr-FR');
                const cls = e.type === 'arrival' ? 'event-arrival' :
                           e.type === 'departure' ? 'event-departure' : '';
                return `<div class="event-item ${cls}">
                    <span class="event-time">${timeStr}</span>${e.message}
                </div>`;
            }).join('');
        }

        function updateStats() {
            const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
            const mins = Math.floor(elapsed / 60);
            const secs = elapsed % 60;

            document.getElementById('stat-esps').textContent = Object.keys(state.esps).length;
            document.getElementById('stat-detections').textContent = state.totalDetections;
            document.getElementById('stat-chunks').textContent = state.totalChunks;
            document.getElementById('stat-elapsed').textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        }

        // Update elapsed time every second
        setInterval(updateStats, 1000);
    </script>
</body>
</html>
"""


@dataclass
class WebESPStatus:
    """Status tracking for a single ESP in web display."""
    esp_id: int
    mac_address: str = ""
    buffer_seconds: float = 0.0
    files_received: int = 0
    chunks_analyzed: int = 0
    current_birds: Dict[str, float] = field(default_factory=dict)


class WebDisplay:
    """
    Minimal web server for InterNest bird detection.

    Uses Flask with Server-Sent Events (SSE) for real-time updates.
    Can be accessed from any device on the same network.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port

        # Flask app
        self.app = Flask(__name__)

        # Event queue for SSE
        self._event_queue: Queue = Queue()
        self._clients: List[Queue] = []
        self._clients_lock = threading.Lock()

        # Setup routes
        self._setup_routes()

        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route('/stream')
        def stream():
            def event_stream():
                # Create a queue for this client
                client_queue = Queue()
                with self._clients_lock:
                    self._clients.append(client_queue)

                try:
                    while True:
                        try:
                            # Wait for an event
                            data = client_queue.get(timeout=30)
                            yield f"data: {json.dumps(data)}\n\n"
                        except Empty:
                            # Send keepalive
                            yield ": keepalive\n\n"
                except GeneratorExit:
                    pass
                finally:
                    with self._clients_lock:
                        if client_queue in self._clients:
                            self._clients.remove(client_queue)

            return Response(
                event_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )

    def _broadcast(self, event_data: dict):
        """Broadcast an event to all connected clients."""
        with self._clients_lock:
            for client_queue in self._clients:
                try:
                    client_queue.put_nowait(event_data)
                except:
                    pass

    def start(self):
        """Start the web server in a background thread."""
        if self._running:
            return

        self._running = True
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self._server_thread.start()

    def _run_server(self):
        """Run the Flask server."""
        # Disable Flask's default logging for cleaner output
        import logging as log
        log.getLogger('werkzeug').setLevel(log.WARNING)

        self.app.run(
            host=self.host,
            port=self.port,
            threaded=True,
            use_reloader=False
        )

    def stop(self):
        """Stop the web server."""
        self._running = False
        # Flask doesn't have a clean shutdown, but since it's daemon thread, it will stop

    # Event handlers (called by the pipeline)

    def on_new_esp(self, esp_id: int, mac_address: str):
        """Called when a new ESP is discovered."""
        self._broadcast({
            'type': 'new_esp',
            'esp_id': esp_id,
            'mac_address': mac_address
        })

    def update_buffer(self, esp_id: int, buffer_seconds: float, files_received: int):
        """Update buffer status for an ESP."""
        self._broadcast({
            'type': 'buffer_update',
            'esp_id': esp_id,
            'buffer_seconds': buffer_seconds,
            'files_received': files_received
        })

    def on_chunk_analyzed(self, esp_id: int):
        """Called when a chunk is analyzed."""
        self._broadcast({
            'type': 'chunk_analyzed',
            'esp_id': esp_id
        })

    def on_detection(self, esp_id: int, species: str, confidence: float):
        """Called when a bird is detected."""
        self._broadcast({
            'type': 'detection',
            'esp_id': esp_id,
            'species': species,
            'confidence': confidence
        })

    def on_arrival(self, esp_id: int, species: str, confidence: float):
        """Called when a new bird arrives."""
        self._broadcast({
            'type': 'arrival',
            'esp_id': esp_id,
            'species': species,
            'confidence': confidence
        })

    def on_departure(self, esp_id: int, species: str, duration: float):
        """Called when a bird departs."""
        self._broadcast({
            'type': 'departure',
            'esp_id': esp_id,
            'species': species,
            'duration': duration
        })

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
