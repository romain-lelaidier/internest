"""
Watcher qui regarde les nouveaux fichiers .bin audio qui arrivent.
Les fichiers .bin sont dans /home/pi/udp_servers/packets sur la Raspberry Pi.

Les fichiers sont mis dans une queue et analysés dans l'ordre d'arrivée.
"""

import time
import threading
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, TYPE_CHECKING
from datetime import datetime
import logging

import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .bin_reader import BinAudioReader, BinFileInfo

logger = logging.getLogger(__name__)


@dataclass
class QueuedFile:
    """Création de la file"""
    info: BinFileInfo
    queued_at: datetime = field(default_factory=datetime.now)
    processing_started: Optional[datetime] = None


class BinFileHandler(FileSystemEventHandler):
    """
    On fera gaffe a bien attendre un peu pour que le fichier .bin soit stabilisé.
    """

    def __init__(
        self,
        file_queue: Queue,
        bin_reader: BinAudioReader,
        esp_filter: Optional[List[int]] = None,
        stabilization_delay: float = 0.5
    ):
        """
        Initialize the file handler.

        Args:
            file_queue: Queue to add new files to
            bin_reader: BinAudioReader instance for parsing files
            esp_filter: Only process files from these ESP IDs (None = all)
            stabilization_delay: Seconds to wait for file to stabilize after creation
        """
        super().__init__()
        self.file_queue = file_queue
        self.bin_reader = bin_reader
        self.esp_filter = set(esp_filter) if esp_filter else None
        self.stabilization_delay = stabilization_delay
        self._pending_files: Dict[Path, float] = {}  # path -> last_modified_time
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() != '.bin':
            return

        logger.debug(f"Nouveau fichier détecté: {path.name}")
        self._schedule_processing(path)

    def on_modified(self, event):
        """Handle file modification events (for files still being written)."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() != '.bin':
            return

        with self._lock:
            if path in self._pending_files:
                # File is still being written, update timestamp
                self._pending_files[path] = time.time()

    def _schedule_processing(self, path: Path):
        """Schedule a file for processing after stabilization delay."""
        with self._lock:
            self._pending_files[path] = time.time()

        # Start a thread to wait for stabilization
        thread = threading.Thread(
            target=self._wait_and_queue,
            args=(path,),
            daemon=True
        )
        thread.start()

    def _wait_and_queue(self, path: Path):
        """Wait for file to stabilize, then add to queue."""
        while True:
            time.sleep(self.stabilization_delay)

            with self._lock:
                if path not in self._pending_files:
                    # Already processed or removed
                    return

                last_modified = self._pending_files[path]
                if time.time() - last_modified >= self.stabilization_delay:
                    # File has stabilized
                    del self._pending_files[path]
                    break

        # Parse file info
        info = self.bin_reader.get_file_info(path)
        if info is None:
            logger.warning(f"On a pas pu parser le fichier : {path.name}")
            return

        # Check ESP filter
        if self.esp_filter and info.esp_id not in self.esp_filter:
            logger.debug(f"Ignorer le fichier de l'ESP{info.esp_id} (not in filter)")
            return

        # Add to queue
        queued = QueuedFile(info=info)
        self.file_queue.put(queued)
        logger.info(f"Queued: {path.name} (ESP{info.esp_id}, {info.duration_seconds:.2f}s)")


class BinFileWatcher:
    """
    Watches a directory for new .bin audio files from ESP32 devices.

    Provides an iterator interface for processing files as they arrive.
    """

    def __init__(
        self,
        watch_dir: Path,
        esp_filter: Optional[List[int]] = None,
        process_existing: bool = True,
        stabilization_delay: float = 0.5    # attendre pour la stabilisation des fichiers
    ):
        """
        Initialize the file watcher.

        Args:
            watch_dir: Directory to watch for .bin files
            esp_filter: Only process files from these ESP IDs (None = all)
            process_existing: If True, process existing files in directory first
            stabilization_delay: Seconds to wait for file to stabilize
        """
        self.watch_dir = Path(watch_dir)
        self.esp_filter = esp_filter
        self.process_existing = process_existing
        self.stabilization_delay = stabilization_delay

        self.file_queue: Queue[QueuedFile] = Queue()
        self.bin_reader = BinAudioReader()

        self._observer: Optional[Observer] = None
        self._handler: Optional[BinFileHandler] = None
        self._running = False
        self._stats = {
            'files_queued': 0,
            'files_processed': 0,
            'bytes_processed': 0,
            'start_time': None
        }

    def start(self):
        """Start watching the directory."""
        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Le dossier à surveiller n'existe pas: {self.watch_dir}")

        logger.info(f"Début du watchdog sur le dossier : {self.watch_dir}")
        self._stats['start_time'] = datetime.now()

        # Process existing files first
        if self.process_existing:
            self._queue_existing_files()

        # Set up watchdog
        self._handler = BinFileHandler(
            file_queue=self.file_queue,
            bin_reader=self.bin_reader,
            esp_filter=self.esp_filter,
            stabilization_delay=self.stabilization_delay
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        self._running = True

        logger.info("File watcher started")

    def stop(self):
        """Stop watching the directory."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None

        self._running = False
        logger.info("Le watcher a été arrêté.")

    def _queue_existing_files(self):
        """Queue existing .bin files in the watch directory."""
        existing_files = sorted(
            self.watch_dir.glob('*.bin'),
            key=lambda p: p.stat().st_mtime
        )

        for path in existing_files:
            info = self.bin_reader.get_file_info(path)
            if info is None:
                continue

            if self.esp_filter and info.esp_id not in self.esp_filter:
                continue

            queued = QueuedFile(info=info)
            self.file_queue.put(queued)
            self._stats['files_queued'] += 1

        logger.info(f"Mise de {self._stats['files_queued']} fichiers existants dans la queue.")

    def get_next_file(self, timeout: Optional[float] = None) -> Optional[QueuedFile]:
        """
        Get the next file from the queue.

        Args:
            timeout: Seconds to wait for a file (None = block forever)

        Returns:
            QueuedFile or None if timeout
        """
        try:
            queued = self.file_queue.get(timeout=timeout)
            queued.processing_started = datetime.now()
            return queued
        except Empty:
            return None

    def mark_processed(self, queued: QueuedFile):
        """Mark a file as processed (for statistics)."""
        self._stats['files_processed'] += 1
        self._stats['bytes_processed'] += queued.info.size_bytes
        self.file_queue.task_done()

    def iter_files(
        self,
        timeout: float = 1.0,
        on_file: Optional[Callable[[QueuedFile], None]] = None
    ):
        """
        Iterate over files as they arrive.

        Args:
            timeout: Seconds to wait between checks
            on_file: Optional callback called for each file

        Yields:
            QueuedFile objects
        """
        while self._running:
            queued = self.get_next_file(timeout=timeout)
            if queued is None:
                continue

            if on_file:
                on_file(queued)

            yield queued

            self.mark_processed(queued)

    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return {
            **self._stats,
            'queue_size': self.file_queue.qsize(),
            'is_running': self._running
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class ESPAudioBuffer:
    """
    Per-ESP audio buffer for accumulating audio chunks.

    Buffers audio data from a single ESP device until enough
    data is available for analysis (e.g., 30 seconds).
    """

    def __init__(
        self,
        esp_id: int,
        sample_rate: int = 48000,
        target_duration: float = 10.0,
        overlap_duration: float = 3.0
    ):
        """
        Initialize the audio buffer.

        Args:
            esp_id: ESP device ID
            sample_rate: Audio sample rate
            target_duration: Target chunk duration in seconds
            overlap_duration: Overlap between consecutive chunks
        """
        self.esp_id = esp_id
        self.sample_rate = sample_rate
        self.target_samples = int(target_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)

        self._buffer: List[float] = []
        self._total_samples_received = 0
        self._chunks_emitted = 0

        # Track timestamps
        self._first_timestamp: Optional[int] = None
        self._last_timestamp: Optional[int] = None

    def add_audio(self, audio: np.ndarray, timestamp: int):
        """
        Add audio samples to the buffer.

        Args:
            audio: Numpy array of audio samples
            timestamp: Unix timestamp of this audio segment
        """
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._last_timestamp = timestamp

        self._buffer.extend(audio.tolist())
        self._total_samples_received += len(audio)

    def has_chunk_ready(self) -> bool:
        """Check if a full chunk is available."""
        return len(self._buffer) >= self.target_samples

    def get_chunk(self) -> Optional[np.ndarray]:
        """
        Get a chunk of audio for analysis.

        Returns:
            Numpy array of audio samples, or None if not enough data
        """
        if not self.has_chunk_ready():
            return None

        # Extract chunk
        chunk = np.array(self._buffer[:self.target_samples], dtype=np.float32)

        # Keep overlap for next chunk
        self._buffer = self._buffer[self.target_samples - self.overlap_samples:]
        self._chunks_emitted += 1

        return chunk

    def get_remaining(self) -> Optional[np.ndarray]:
        """Get any remaining audio in the buffer (for finalization)."""
        if len(self._buffer) == 0:
            return None

        chunk = np.array(self._buffer, dtype=np.float32)
        self._buffer = []

        return chunk

    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        return len(self._buffer) / self.sample_rate

    @property
    def chunks_emitted(self) -> int:
        """Number of chunks emitted so far."""
        return self._chunks_emitted

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'esp_id': self.esp_id,
            'buffer_samples': len(self._buffer),
            'buffer_duration': self.buffer_duration,
            'total_samples': self._total_samples_received,
            'chunks_emitted': self._chunks_emitted,
            'first_timestamp': self._first_timestamp,
            'last_timestamp': self._last_timestamp
        }
