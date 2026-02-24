"""
Real-time bird detection pipeline for ESP32 audio streams.

Processes .bin audio files as they arrive from ESP32 devices,
buffers audio per device, and runs BirdNET analysis on complete chunks.
"""

import time
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from datetime import datetime
from threading import Thread, Event
from queue import Queue

import numpy as np
import soundfile as sf

from .config import (
    BirdNETConfig, StreamingConfig, FilterConfig,
    Detection, AudioChunk
)
from .bin_reader import BinAudioReader
from .file_watcher import BinFileWatcher, ESPAudioBuffer, QueuedFile
from .birdnet_analyzer import BirdNETAnalyzerWrapper
from .detection_tracker import DetectionTracker
from .presence_tracker import PresenceTracker

logger = logging.getLogger(__name__)


@dataclass
class ESPDetectionEvent:
    """A detection event from a specific ESP device."""
    esp_id: int
    detection: Detection
    timestamp: datetime = field(default_factory=datetime.now)
    chunk_index: int = 0


@dataclass
class RealtimePipelineConfig:
    """Configuration for the real-time pipeline."""
    watch_dir: Path

    # Audio settings
    sample_rate: int = 44100  # 44.1kHz from ESP32
    chunk_duration: float = 15.0  # seconds
    overlap_duration: float = 3.0  # seconds

    # BirdNET settings
    min_confidence: float = 0.25
    confidence_threshold: float = 0.5
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Presence tracking
    departure_timeout: float = 30.0

    # Processing
    process_existing: bool = True
    stabilization_delay: float = 0.5


class RealtimeBirdDetector:
    """
    Real-time bird detector for multiple ESP32 audio streams.

    Features:
    - Monitors a directory for new .bin audio files
    - Maintains per-ESP audio buffers
    - Runs BirdNET analysis on 30-second windows
    - Tracks bird presence with arrival/departure events
    - Thread-safe detection event queue
    """

    def __init__(self, config: RealtimePipelineConfig):
        self.config = config

        # Initialize components
        self.bin_reader = BinAudioReader(sample_rate=config.sample_rate)
        self.file_watcher: Optional[BinFileWatcher] = None

        # Per-ESP buffers (created dynamically when new ESP is discovered)
        self.esp_buffers: Dict[int, ESPAudioBuffer] = {}

        # BirdNET analyzer
        birdnet_config = BirdNETConfig(
            min_confidence=config.min_confidence,
            latitude=config.latitude,
            longitude=config.longitude,
            date=datetime.now()
        )
        self.analyzer = BirdNETAnalyzerWrapper(birdnet_config)

        # Detection tracking
        filter_config = FilterConfig(confidence_threshold=config.confidence_threshold)
        self.detection_tracker = DetectionTracker(filter_config)

        # Per-ESP presence trackers (created dynamically)
        self.presence_trackers: Dict[int, PresenceTracker] = {}

        # Event queue for external consumers
        self.detection_queue: Queue[ESPDetectionEvent] = Queue()

        # Callbacks
        self._on_detection: Optional[Callable[[ESPDetectionEvent], None]] = None
        self._on_arrival: Optional[Callable[[int, str, float], None]] = None
        self._on_departure: Optional[Callable[[int, str, float], None]] = None
        self._on_buffer_update: Optional[Callable[[int, float, int], None]] = None  # esp_id, buffer_secs, files
        self._on_chunk_analyzed: Optional[Callable[[int], None]] = None  # esp_id

        # State
        self._running = False
        self._stop_event = Event()
        self._processing_thread: Optional[Thread] = None
        self._temp_files: List[Path] = []

        # Statistics
        self._stats = {
            'files_processed': 0,
            'chunks_analyzed': 0,
            'total_detections': 0,
            'start_time': None,
            'per_esp_chunks': {}  # Created dynamically
        }

        # Callback for new ESP discovered
        self._on_new_esp: Optional[Callable[[int, str], None]] = None  # esp_id, mac_address
        self._notified_esps: set = set()  # Track which ESPs we've notified about

    def _get_or_create_esp(self, esp_id: int):
        """Get or create buffer and tracker for an ESP."""
        if esp_id not in self.esp_buffers:
            self.esp_buffers[esp_id] = ESPAudioBuffer(
                esp_id=esp_id,
                sample_rate=self.config.sample_rate,
                target_duration=self.config.chunk_duration,
                overlap_duration=self.config.overlap_duration
            )
            self.presence_trackers[esp_id] = PresenceTracker(
                departure_timeout=self.config.departure_timeout,
                silent=True
            )
            self._stats['per_esp_chunks'][esp_id] = 0
            logger.info(f"Créé buffer et tracker pour ESP {esp_id}")

    def on_new_esp(self, callback: Callable[[int, str], None]):
        """Set callback when a new ESP is discovered (esp_id, mac_address)."""
        self._on_new_esp = callback

    def on_detection(self, callback: Callable[[ESPDetectionEvent], None]):
        """Set callback for detection events."""
        self._on_detection = callback

    def on_arrival(self, callback: Callable[[int, str, float], None]):
        """Set callback for bird arrival events (esp_id, species, confidence)."""
        self._on_arrival = callback

    def on_departure(self, callback: Callable[[int, str, float], None]):
        """Set callback for bird departure events (esp_id, species, duration)."""
        self._on_departure = callback

    def on_buffer_update(self, callback: Callable[[int, float, int], None]):
        """Set callback for buffer updates (esp_id, buffer_seconds, files_count)."""
        self._on_buffer_update = callback

    def on_chunk_analyzed(self, callback: Callable[[int], None]):
        """Set callback when a chunk is analyzed (esp_id)."""
        self._on_chunk_analyzed = callback

    def start(self):
        """Start the real-time detection pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        logger.info(f"Starting real-time pipeline, watching: {self.config.watch_dir}")
        self._stats['start_time'] = datetime.now()

        # Initialize file watcher
        self.file_watcher = BinFileWatcher(
            watch_dir=self.config.watch_dir,
            process_existing=self.config.process_existing,
            stabilization_delay=self.config.stabilization_delay
        )
        self.file_watcher.start()

        # Start processing thread
        self._running = True
        self._stop_event.clear()
        self._processing_thread = Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()

        logger.info("Pipeline started")

    def stop(self):
        """Stop the pipeline gracefully."""
        if not self._running:
            return

        logger.info("Stopping pipeline...")
        self._stop_event.set()
        self._running = False

        # Stop file watcher
        if self.file_watcher:
            self.file_watcher.stop()

        # Wait for processing thread
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        # Process remaining buffers
        self._finalize_buffers()

        # Cleanup temp files
        self._cleanup_temp_files()

        logger.info("Pipeline stopped")

    def _processing_loop(self):
        """Main processing loop (runs in thread)."""
        while not self._stop_event.is_set():
            queued = self.file_watcher.get_next_file(timeout=0.5)
            if queued is None:
                # Check for ready chunks even without new files
                self._process_ready_chunks()
                continue

            try:
                self._process_file(queued)
                self.file_watcher.mark_processed(queued)
            except Exception as e:
                logger.error(f"Error processing {queued.info.path.name}: {e}")

    def _process_file(self, queued: QueuedFile):
        """Process a single .bin file."""
        info = queued.info
        logger.debug(f"Processing: {info.path.name}")

        # Get ESP ID from MAC address mapping
        esp_id = self.file_watcher.get_esp_id(info.mac_address)

        # Ensure buffer and tracker exist for this ESP
        self._get_or_create_esp(esp_id)

        # Notify new ESP if callback is set
        if esp_id not in self._notified_esps:
            self._notified_esps.add(esp_id)
            if self._on_new_esp:
                self._on_new_esp(esp_id, info.mac_address)

        # Read audio data
        audio = self.bin_reader.read_file(info.path)

        # Add to ESP buffer
        buffer = self.esp_buffers[esp_id]
        buffer.add_audio(audio, info.timestamp)
        self._stats['files_processed'] += 1

        # Notify buffer update
        if self._on_buffer_update:
            self._on_buffer_update(esp_id, buffer.buffer_duration, self._stats['files_processed'])

        # Process any ready chunks
        self._process_ready_chunks()

    def _process_ready_chunks(self):
        """Process all ready chunks from all ESP buffers."""
        for esp_id, buffer in self.esp_buffers.items():
            while buffer.has_chunk_ready():
                chunk_audio = buffer.get_chunk()
                if chunk_audio is not None:
                    self._analyze_chunk(esp_id, chunk_audio, buffer.chunks_emitted)

    def _analyze_chunk(self, esp_id: int, audio: np.ndarray, chunk_index: int):
        """Analyze an audio chunk with BirdNET."""
        logger.debug(f"Analyzing chunk {chunk_index} from ESP{esp_id}")

        # Save to temp file for BirdNET
        temp_path = self._save_temp_wav(audio)

        try:
            # Create AudioChunk for analysis
            chunk = AudioChunk(
                data=audio,
                start_time=chunk_index * (self.config.chunk_duration - self.config.overlap_duration),
                end_time=chunk_index * (self.config.chunk_duration - self.config.overlap_duration) + self.config.chunk_duration,
                sample_rate=self.config.sample_rate,
                chunk_index=chunk_index
            )

            # Analyze with BirdNET (using the temp file directly)
            detections = self._analyze_with_file(temp_path, chunk)

            # Filter detections
            filtered = self.detection_tracker.process_detections(detections)

            # Update stats
            self._stats['chunks_analyzed'] += 1
            self._stats['per_esp_chunks'][esp_id] += 1
            self._stats['total_detections'] += len(filtered)

            # Notify chunk analyzed
            if self._on_chunk_analyzed:
                self._on_chunk_analyzed(esp_id)

            # Update presence tracker for this ESP
            tracker = self.presence_trackers.get(esp_id)
            if tracker:
                # Get arrivals/departures before update
                previously_present = set(tracker.present_birds.keys())

                tracker.update(filtered, chunk.end_time)

                # Check for arrivals
                currently_present = set(tracker.present_birds.keys())
                new_arrivals = currently_present - previously_present

                for species in new_arrivals:
                    if self._on_arrival:
                        detection = next((d for d in filtered if d.common_name == species), None)
                        conf = detection.confidence if detection else 0.5
                        self._on_arrival(esp_id, species, conf)

            # Emit detection events
            for detection in filtered:
                event = ESPDetectionEvent(
                    esp_id=esp_id,
                    detection=detection,
                    chunk_index=chunk_index
                )
                self.detection_queue.put(event)

                if self._on_detection:
                    self._on_detection(event)

        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except Exception:
                pass

    def _analyze_with_file(self, wav_path: Path, chunk: AudioChunk) -> List[Detection]:
        """Analyze a WAV file with BirdNET."""
        import sys
        import os

        try:
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer

            # Get or create analyzer (suppress prints during init)
            if not hasattr(self, '_birdnet_analyzer'):
                # Suppress birdnetlib prints
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        self._birdnet_analyzer = Analyzer()
                    finally:
                        sys.stdout = old_stdout

            # Suppress prints during analysis
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    recording = Recording(
                        self._birdnet_analyzer,
                        str(wav_path),
                        lat=self.config.latitude,
                        lon=self.config.longitude,
                        date=datetime.now(),
                        min_conf=self.config.min_confidence
                    )
                    recording.analyze()
                finally:
                    sys.stdout = old_stdout

            # Convert to Detection objects
            detections = []
            for d in recording.detections:
                detection = Detection(
                    common_name=d['common_name'],
                    scientific_name=d['scientific_name'],
                    confidence=d['confidence'],
                    start_time=chunk.start_time + d['start_time'],
                    end_time=chunk.start_time + d['end_time'],
                    chunk_index=chunk.chunk_index
                )
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"BirdNET analysis error: {e}")
            return []

    def _save_temp_wav(self, audio: np.ndarray) -> Path:
        """Save audio to a temporary WAV file."""
        fd, path = tempfile.mkstemp(suffix='.wav')
        path = Path(path)

        sf.write(str(path), audio, self.config.sample_rate)
        self._temp_files.append(path)

        return path

    def _finalize_buffers(self):
        """Process remaining audio in all buffers."""
        logger.info("Finalizing remaining buffers...")

        for esp_id, buffer in self.esp_buffers.items():
            remaining = buffer.get_remaining()
            if remaining is not None and len(remaining) >= self.config.sample_rate * 3:
                # At least 3 seconds of audio
                self._analyze_chunk(esp_id, remaining, buffer.chunks_emitted)

            # Finalize presence tracker
            tracker = self.presence_trackers.get(esp_id)
            if tracker:
                # Get departures
                for species in list(tracker.present_birds.keys()):
                    if self._on_departure:
                        presence = tracker.present_birds[species]
                        duration = presence.total_duration
                        self._on_departure(esp_id, species, duration)

                tracker.finalize()

    def _cleanup_temp_files(self):
        """Remove temporary WAV files."""
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        self._temp_files.clear()

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        watcher_stats = self.file_watcher.get_stats() if self.file_watcher else {}

        buffer_stats = {}
        for esp_id, buffer in self.esp_buffers.items():
            buffer_stats[f'esp{esp_id}'] = buffer.get_stats()

        presence_stats = {}
        for esp_id, tracker in self.presence_trackers.items():
            presence_stats[f'esp{esp_id}'] = tracker.get_summary()

        return {
            **self._stats,
            'watcher': watcher_stats,
            'buffers': buffer_stats,
            'presence': presence_stats,
            'queue_size': self.detection_queue.qsize()
        }

    def get_present_birds(self, esp_id: Optional[int] = None) -> Dict[str, dict]:
        """
        Get currently present birds.

        Args:
            esp_id: Specific ESP to query (None = all ESPs)

        Returns:
            Dict mapping species names to presence info
        """
        if esp_id is not None:
            tracker = self.presence_trackers.get(esp_id)
            if tracker:
                return {
                    species: {
                        'first_seen': p.first_seen,
                        'last_seen': p.last_seen,
                        'detection_count': p.detection_count,
                        'avg_confidence': p.avg_confidence
                    }
                    for species, p in tracker.present_birds.items()
                }
            return {}

        # Aggregate all ESPs
        all_birds = {}
        for tracker in self.presence_trackers.values():
            for species, p in tracker.present_birds.items():
                if species not in all_birds:
                    all_birds[species] = {
                        'esp_ids': [],
                        'first_seen': p.first_seen,
                        'last_seen': p.last_seen,
                        'detection_count': 0,
                        'max_confidence': 0
                    }
                all_birds[species]['esp_ids'].append(tracker)
                all_birds[species]['detection_count'] += p.detection_count
                all_birds[species]['max_confidence'] = max(
                    all_birds[species]['max_confidence'],
                    p.avg_confidence
                )
                all_birds[species]['first_seen'] = min(
                    all_birds[species]['first_seen'],
                    p.first_seen
                )
                all_birds[species]['last_seen'] = max(
                    all_birds[species]['last_seen'],
                    p.last_seen
                )

        return all_birds

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
