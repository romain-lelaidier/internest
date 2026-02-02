"""Real-time bird presence tracking with arrival/departure alerts."""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Detection


class EventType(Enum):
    ARRIVAL = "arrival"
    DEPARTURE = "departure"


@dataclass
class BirdPresence:
    """Tracks presence of a bird species."""
    common_name: str
    scientific_name: str
    first_seen: float  # timestamp in audio
    last_seen: float
    detection_count: int = 1
    max_confidence: float = 0.0
    total_confidence: float = 0.0  # Sum for avg calculation

    def update(self, detection: Detection):
        """Update presence with new detection."""
        self.last_seen = detection.start_time
        self.detection_count += 1
        self.max_confidence = max(self.max_confidence, detection.confidence)
        self.total_confidence += detection.confidence

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all detections."""
        if self.detection_count == 0:
            return 0.0
        return self.total_confidence / self.detection_count

    @property
    def total_duration(self) -> float:
        """Total duration this bird has been present."""
        return self.last_seen - self.first_seen


@dataclass
class PresenceEvent:
    """An arrival or departure event."""
    event_type: EventType
    common_name: str
    scientific_name: str
    timestamp: float  # in audio time
    confidence: float = 0.0
    duration_present: float = 0.0  # only for departures


class PresenceTracker:
    """
    Tracks bird presence over time and generates arrival/departure events.

    A bird is considered "present" if detected recently.
    A bird "departs" if not detected for `departure_timeout` seconds.
    """

    def __init__(
        self,
        departure_timeout: float = 30.0,  # seconds without detection = departed
        console: Optional[Console] = None,
        silent: bool = False  # If True, don't print alerts
    ):
        self.departure_timeout = departure_timeout
        self.console = console or Console()
        self.silent = silent

        # Currently present birds: species_name -> BirdPresence
        self.present_birds: Dict[str, BirdPresence] = {}

        # Event history
        self.events: list[PresenceEvent] = []

        # Current audio time
        self.current_time: float = 0.0

    def update(self, detections: list[Detection], current_time: float) -> list[PresenceEvent]:
        """
        Update tracker with new detections and return any events.

        Args:
            detections: New detections from current chunk
            current_time: Current position in audio stream

        Returns:
            List of arrival/departure events
        """
        self.current_time = current_time
        events = []

        # Track which species were detected this update
        detected_species: Set[str] = set()

        # Process new detections
        for det in detections:
            species = det.common_name
            detected_species.add(species)

            if species not in self.present_birds:
                # NEW ARRIVAL
                presence = BirdPresence(
                    common_name=det.common_name,
                    scientific_name=det.scientific_name,
                    first_seen=det.start_time,
                    last_seen=det.start_time,
                    max_confidence=det.confidence,
                    total_confidence=det.confidence
                )
                self.present_birds[species] = presence

                event = PresenceEvent(
                    event_type=EventType.ARRIVAL,
                    common_name=det.common_name,
                    scientific_name=det.scientific_name,
                    timestamp=det.start_time,
                    confidence=det.confidence
                )
                events.append(event)
                self._alert_arrival(event)
            else:
                # Update existing presence
                self.present_birds[species].update(det)

        # Check for departures (birds not seen for too long)
        departed = []
        for species, presence in self.present_birds.items():
            time_since_seen = current_time - presence.last_seen
            if time_since_seen > self.departure_timeout:
                departed.append(species)

                event = PresenceEvent(
                    event_type=EventType.DEPARTURE,
                    common_name=presence.common_name,
                    scientific_name=presence.scientific_name,
                    timestamp=current_time,
                    confidence=presence.max_confidence,
                    duration_present=presence.last_seen - presence.first_seen
                )
                events.append(event)
                self._alert_departure(event)

        # Remove departed birds
        for species in departed:
            del self.present_birds[species]

        self.events.extend(events)
        return events

    def _alert_arrival(self, event: PresenceEvent):
        """Display arrival alert."""
        if self.silent:
            return
        self.console.print(Panel(
            f"[bold green]{event.common_name}[/bold green]\n"
            f"[dim]{event.scientific_name}[/dim]\n"
            f"Confiance: {event.confidence:.0%}",
            title="[green]🟢 ARRIVÉE[/green]",
            border_style="green",
            width=50
        ))

    def _alert_departure(self, event: PresenceEvent):
        """Display departure alert."""
        if self.silent:
            return
        self.console.print(Panel(
            f"[bold red]{event.common_name}[/bold red]\n"
            f"[dim]{event.scientific_name}[/dim]\n"
            f"Présent pendant: {event.duration_present:.0f}s",
            title="[red]🔴 DÉPART[/red]",
            border_style="red",
            width=50
        ))

    def get_present_species(self) -> list[str]:
        """Get list of currently present species."""
        return list(self.present_birds.keys())

    def get_status_display(self) -> Panel:
        """Get a rich panel showing current status."""
        if not self.present_birds:
            content = "[dim]Aucun oiseau détecté[/dim]"
        else:
            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("Espèce", style="green")
            table.add_column("Depuis", style="cyan", justify="right")
            table.add_column("Confiance", style="yellow", justify="right")

            for presence in self.present_birds.values():
                duration = self.current_time - presence.first_seen
                table.add_row(
                    presence.common_name,
                    f"{duration:.0f}s",
                    f"{presence.max_confidence:.0%}"
                )
            content = table

        return Panel(
            content,
            title=f"[bold]Oiseaux présents ({len(self.present_birds)})[/bold]",
            border_style="blue"
        )

    def finalize(self) -> list[PresenceEvent]:
        """Mark all remaining birds as departed at end of stream."""
        events = []
        for species, presence in list(self.present_birds.items()):
            event = PresenceEvent(
                event_type=EventType.DEPARTURE,
                common_name=presence.common_name,
                scientific_name=presence.scientific_name,
                timestamp=self.current_time,
                confidence=presence.max_confidence,
                duration_present=presence.last_seen - presence.first_seen
            )
            events.append(event)
            self._alert_departure(event)

        self.present_birds.clear()
        self.events.extend(events)
        return events

    def get_summary(self) -> dict:
        """Get summary of all events."""
        arrivals = [e for e in self.events if e.event_type == EventType.ARRIVAL]
        departures = [e for e in self.events if e.event_type == EventType.DEPARTURE]

        unique_species = set(e.common_name for e in arrivals)

        return {
            "total_arrivals": len(arrivals),
            "total_departures": len(departures),
            "unique_species": len(unique_species),
            "species_list": list(unique_species),
            "events": [
                {
                    "type": e.event_type.value,
                    "species": e.common_name,
                    "scientific_name": e.scientific_name,
                    "timestamp": e.timestamp,
                    "confidence": e.confidence
                }
                for e in self.events
            ]
        }
