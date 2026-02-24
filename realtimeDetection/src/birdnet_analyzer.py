"""BirdNET-Analyzer wrapper for bird species identification."""

import logging
from typing import List, Optional
from datetime import datetime

from .config import Detection, AudioChunk, BirdNETConfig
from .audio_stream import AudioStreamProcessor

logger = logging.getLogger(__name__)


class BirdNETAnalyzerWrapper:
    """
    Wrapper for BirdNET-Analyzer providing chunked analysis.

    Uses birdnetlib for clean Python API access to BirdNET models.
    """

    def __init__(self, config: BirdNETConfig):
        self.config = config
        self._analyzer = None
        self._load_analyzer()

    def _load_analyzer(self):
        """Load the BirdNET analyzer model."""
        import sys
        import os

        try:
            from birdnetlib.analyzer import Analyzer
            logger.info("Loading BirdNET analyzer model...")

            # Suppress birdnetlib prints during model loading
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    self._analyzer = Analyzer()
                finally:
                    sys.stdout = old_stdout

            logger.info("BirdNET analyzer loaded successfully")
        except ImportError:
            raise ImportError(
                "birdnetlib is not installed. "
                "Please install it with: pip install birdnetlib"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BirdNET analyzer: {e}")

    def analyze_file(self, audio_path: str, chunk_start_time: float = 0.0) -> List[Detection]:
        """
        Analyze an audio file and return detections.

        Args:
            audio_path: Path to audio file (should be 3-second chunk)
            chunk_start_time: Start time of this chunk in the original recording

        Returns:
            List of Detection objects
        """
        from birdnetlib import Recording

        # Create recording with optional geographic filtering
        recording_kwargs = {
            'min_conf': self.config.min_confidence
        }

        if self.config.latitude is not None and self.config.longitude is not None:
            recording_kwargs['lat'] = self.config.latitude
            recording_kwargs['lon'] = self.config.longitude

        if self.config.date is not None:
            recording_kwargs['date'] = self.config.date
        elif self.config.week is not None:
            recording_kwargs['week'] = self.config.week

        recording = Recording(
            self._analyzer,
            audio_path,
            **recording_kwargs
        )

        # Run analysis
        recording.analyze()

        # Convert to Detection objects with adjusted timestamps
        detections = []
        for det in recording.detections:
            detection = Detection(
                common_name=det.get('common_name', 'Unknown'),
                scientific_name=det.get('scientific_name', 'Unknown'),
                confidence=det.get('confidence', 0.0),
                start_time=chunk_start_time + det.get('start_time', 0.0),
                end_time=chunk_start_time + det.get('end_time', 0.0),
                chunk_index=0  # Will be set by caller
            )
            detections.append(detection)

        return detections

    def analyze_chunk(
        self,
        chunk: AudioChunk,
        stream_processor: AudioStreamProcessor
    ) -> List[Detection]:
        """
        Analyze an audio chunk.

        Args:
            chunk: AudioChunk to analyze
            stream_processor: AudioStreamProcessor for temp file handling

        Returns:
            List of Detection objects
        """
        # Save chunk to temporary file
        temp_path = stream_processor.chunk_to_temp_file(chunk)

        try:
            # Analyze the chunk
            detections = self.analyze_file(temp_path, chunk.start_time)

            # Set chunk index on all detections
            for det in detections:
                det.chunk_index = chunk.chunk_index

            return detections

        except Exception as e:
            logger.warning(f"Error analyzing chunk {chunk.chunk_index}: {e}")
            return []

    @property
    def is_loaded(self) -> bool:
        """Check if the analyzer is loaded."""
        return self._analyzer is not None
