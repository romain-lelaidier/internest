"""Detection tracking and deduplication."""

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, field

from .config import Detection, FilterConfig


@dataclass
class SpeciesStats:
    """Statistics for a detected species."""
    common_name: str
    scientific_name: str
    count: int = 0
    total_confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    detections: List[Detection] = field(default_factory=list)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all detections."""
        return self.total_confidence / self.count if self.count > 0 else 0.0

    def add_detection(self, detection: Detection):
        """Add a detection to the stats."""
        self.count += 1
        self.total_confidence += detection.confidence
        self.last_seen = detection.start_time
        if self.count == 1:
            self.first_seen = detection.start_time
        self.detections.append(detection)


class DetectionTracker:
    """
    Tracks detections across chunks to:
    1. Filter by confidence threshold
    2. Deduplicate overlapping detections
    3. Apply species filters
    4. Aggregate statistics
    """

    def __init__(self, config: FilterConfig):
        self.config = config
        self.recent_detections: List[Detection] = []
        self.all_detections: List[Detection] = []
        self.species_stats: Dict[str, SpeciesStats] = {}
        self.dedup_window: float = 2.0  # seconds

    def process_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Process new detections, returning only valid new ones.

        Args:
            detections: List of new detections from a chunk

        Returns:
            List of valid, non-duplicate detections
        """
        valid = []

        for det in detections:
            # Apply confidence threshold
            if det.confidence < self.config.confidence_threshold:
                continue

            # Apply species whitelist
            if self.config.species_whitelist:
                if det.common_name not in self.config.species_whitelist:
                    continue

            # Apply species blacklist
            if det.common_name in self.config.species_blacklist:
                continue

            # Check for duplicates in recent window
            if self._is_duplicate(det):
                continue

            # Valid detection - track it
            valid.append(det)
            self.recent_detections.append(det)
            self.all_detections.append(det)
            self._update_species_stats(det)

        # Prune old detections from recent tracking
        self._prune_old_detections()

        return valid

    def _is_duplicate(self, detection: Detection) -> bool:
        """Check if detection overlaps with recent detection of same species."""
        for recent in self.recent_detections:
            if recent.common_name != detection.common_name:
                continue

            # Check temporal overlap
            time_diff = abs(detection.start_time - recent.start_time)
            if time_diff < self.dedup_window:
                return True

        return False

    def _prune_old_detections(self):
        """Remove old detections from recent tracking window."""
        if not self.recent_detections:
            return

        # Keep only detections within the last dedup_window * 2
        latest_time = max(d.start_time for d in self.recent_detections)
        cutoff_time = latest_time - (self.dedup_window * 2)

        self.recent_detections = [
            d for d in self.recent_detections
            if d.start_time >= cutoff_time
        ]

    def _update_species_stats(self, detection: Detection):
        """Update species statistics with a new detection."""
        name = detection.common_name

        if name not in self.species_stats:
            self.species_stats[name] = SpeciesStats(
                common_name=detection.common_name,
                scientific_name=detection.scientific_name
            )

        self.species_stats[name].add_detection(detection)

    def get_summary(self) -> Dict:
        """Get summary of all detections."""
        return {
            'total_detections': len(self.all_detections),
            'unique_species': len(self.species_stats),
            'species': [
                {
                    'common_name': stats.common_name,
                    'scientific_name': stats.scientific_name,
                    'count': stats.count,
                    'avg_confidence': round(stats.avg_confidence, 3),
                    'first_seen': stats.first_seen,
                    'last_seen': stats.last_seen
                }
                for stats in sorted(
                    self.species_stats.values(),
                    key=lambda s: s.count,
                    reverse=True
                )
            ]
        }

    def get_recent_detections(self, n: int = 10) -> List[Detection]:
        """Get the n most recent detections."""
        return sorted(
            self.all_detections,
            key=lambda d: d.start_time,
            reverse=True
        )[:n]

    def reset(self):
        """Reset all tracking state."""
        self.recent_detections = []
        self.all_detections = []
        self.species_stats = {}
