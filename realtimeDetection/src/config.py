"""Configuration dataclasses for the bird detection pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import json

if TYPE_CHECKING:
    import numpy as np


@dataclass
class AudioConfig:
    """Audio input configuration."""
    input_path: str
    sample_rate: int = 48000


@dataclass
class BirdNETConfig:
    """BirdNET analyzer configuration."""
    min_confidence: float = 0.25
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: Optional[datetime] = None
    week: Optional[int] = None


@dataclass
class StreamingConfig:
    """Audio streaming/chunking configuration."""
    chunk_duration_seconds: float = 30.0  # 30s windows
    overlap_seconds: float = 3.0  # Small overlap
    simulate_realtime: bool = False
    realtime_speed: float = 1.0


@dataclass
class OutputConfig:
    """Output configuration."""
    mode: str = "cli"  # "cli", "json", "csv", "all"
    json_path: Optional[str] = None
    csv_path: Optional[str] = None
    show_summary: bool = True


@dataclass
class FilterConfig:
    """Detection filtering configuration."""
    confidence_threshold: float = 0.5
    species_whitelist: List[str] = field(default_factory=list)
    species_blacklist: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    audio: AudioConfig
    birdnet: BirdNETConfig
    streaming: StreamingConfig
    output: OutputConfig
    filtering: FilterConfig

    @classmethod
    def from_json(cls, path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Parse date if present
        birdnet_data = data.get('birdnet', {})
        if 'date' in birdnet_data and birdnet_data['date']:
            birdnet_data['date'] = datetime.fromisoformat(birdnet_data['date'])

        return cls(
            audio=AudioConfig(**data.get('audio', {})),
            birdnet=BirdNETConfig(**birdnet_data),
            streaming=StreamingConfig(**data.get('streaming', {})),
            output=OutputConfig(**data.get('output', {})),
            filtering=FilterConfig(**data.get('filtering', {}))
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'audio': {
                'input_path': self.audio.input_path,
                'sample_rate': self.audio.sample_rate
            },
            'birdnet': {
                'min_confidence': self.birdnet.min_confidence,
                'latitude': self.birdnet.latitude,
                'longitude': self.birdnet.longitude,
                'date': self.birdnet.date.isoformat() if self.birdnet.date else None,
                'week': self.birdnet.week
            },
            'streaming': {
                'chunk_duration_seconds': self.streaming.chunk_duration_seconds,
                'overlap_seconds': self.streaming.overlap_seconds,
                'simulate_realtime': self.streaming.simulate_realtime,
                'realtime_speed': self.streaming.realtime_speed
            },
            'output': {
                'mode': self.output.mode,
                'json_path': self.output.json_path,
                'csv_path': self.output.csv_path,
                'show_summary': self.output.show_summary
            },
            'filtering': {
                'confidence_threshold': self.filtering.confidence_threshold,
                'species_whitelist': self.filtering.species_whitelist,
                'species_blacklist': self.filtering.species_blacklist
            }
        }


@dataclass
class Detection:
    """A single bird detection result."""
    common_name: str
    scientific_name: str
    confidence: float
    start_time: float  # seconds from start of recording
    end_time: float  # seconds from start of recording
    chunk_index: int  # which chunk this came from

    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            'common_name': self.common_name,
            'scientific_name': self.scientific_name,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'timestamp': f"{int(self.start_time // 60):02d}:{self.start_time % 60:05.2f}"
        }


@dataclass
class AudioChunk:
    """An audio chunk for analysis."""
    data: Any  # numpy.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    chunk_index: int


@dataclass
class AnalysisResult:
    """Result of a complete analysis."""
    detections: List[Detection]
    duration: float
    config: PipelineConfig

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            'detections': [d.to_dict() for d in self.detections],
            'duration_seconds': self.duration,
            'num_detections': len(self.detections),
            'unique_species': len(set(d.common_name for d in self.detections))
        }
