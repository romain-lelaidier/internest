"""Audio streaming and chunking for real-time analysis."""

import numpy as np
import librosa
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Iterator, Tuple, Optional
from dataclasses import dataclass

from .config import AudioChunk, StreamingConfig


@dataclass
class AudioInfo:
    """Information about loaded audio."""
    duration_seconds: float
    sample_rate: int
    num_samples: int
    channels: int


class AudioStreamProcessor:
    """
    Processes audio files in overlapping chunks to simulate real-time streaming.

    BirdNET analyzes 3-second segments, so we use overlapping chunks
    to ensure no vocalizations at chunk boundaries are missed.
    """

    def __init__(self, config: StreamingConfig, target_sample_rate: int = 48000):
        self.config = config
        self.target_sample_rate = target_sample_rate
        self._temp_dir = None

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int, AudioInfo]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, OGG)

        Returns:
            Tuple of (signal, sample_rate, info)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load with librosa (handles multiple formats)
        signal, sr = librosa.load(
            audio_path,
            sr=self.target_sample_rate,
            mono=True
        )

        info = AudioInfo(
            duration_seconds=len(signal) / sr,
            sample_rate=sr,
            num_samples=len(signal),
            channels=1
        )

        return signal, sr, info

    def stream_file(self, audio_path: str) -> Iterator[AudioChunk]:
        """
        Yield audio chunks from a file, simulating real-time stream.

        Args:
            audio_path: Path to audio file

        Yields:
            AudioChunk with numpy array and timing metadata
        """
        signal, sr, _ = self.load_audio(audio_path)

        chunk_samples = int(self.config.chunk_duration_seconds * sr)
        hop_samples = int((self.config.chunk_duration_seconds - self.config.overlap_seconds) * sr)

        position = 0
        chunk_index = 0

        while position < len(signal):
            chunk_end = min(position + chunk_samples, len(signal))
            chunk_data = signal[position:chunk_end]

            # Pad if needed for final chunk
            if len(chunk_data) < chunk_samples:
                chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)))

            yield AudioChunk(
                data=chunk_data,
                start_time=position / sr,
                end_time=chunk_end / sr,
                sample_rate=sr,
                chunk_index=chunk_index
            )

            position += hop_samples
            chunk_index += 1

    def get_audio_info(self, audio_path: str) -> AudioInfo:
        """Get information about an audio file without loading it fully."""
        _, sr, info = self.load_audio(audio_path)
        return info

    def chunk_to_temp_file(self, chunk: AudioChunk) -> str:
        """
        Save audio chunk to a temporary WAV file for BirdNET processing.

        Args:
            chunk: AudioChunk to save

        Returns:
            Path to temporary WAV file
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="birdnet_")

        temp_path = Path(self._temp_dir) / f"chunk_{chunk.chunk_index:06d}.wav"
        sf.write(str(temp_path), chunk.data, chunk.sample_rate)
        return str(temp_path)

    def cleanup_temp_files(self):
        """Remove temporary files created during processing."""
        if self._temp_dir is not None:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mm timestamp."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
