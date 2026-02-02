"""
Live dashboard display for multi-ESP bird detection.

Uses Rich Live display to show real-time status of each ESP
in separate columns with detection events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text


@dataclass
class ESPStatus:
    """Status tracking for a single ESP."""
    esp_id: int
    files_received: int = 0
    buffer_seconds: float = 0.0
    chunks_analyzed: int = 0
    current_birds: Dict[str, float] = field(default_factory=dict)  # species -> confidence
    recent_events: deque = field(default_factory=lambda: deque(maxlen=5))
    last_update: Optional[datetime] = None


class MultiESPDisplay:
    """
    Real-time dashboard for monitoring multiple ESP streams.

    Shows each ESP in a column with:
    - Buffer status
    - Currently detected birds
    - Recent events (arrivals/departures)

    ESPs can be added dynamically as they are discovered.
    """

    def __init__(self, console: Optional[Console] = None):
        self.esp_ids: List[int] = []
        self.console = console or Console()

        # Per-ESP status (created dynamically)
        self.esp_status: Dict[int, ESPStatus] = {}

        # MAC address mapping for display
        self.esp_mac_mapping: Dict[int, str] = {}

        # Global stats
        self.total_detections = 0
        self.start_time = datetime.now()

        # Event log (global)
        self.event_log: deque = deque(maxlen=10)

        # Live display
        self._live: Optional[Live] = None

    def add_esp(self, esp_id: int, mac_address: str = ""):
        """Add a new ESP to the display."""
        if esp_id not in self.esp_status:
            self.esp_ids.append(esp_id)
            self.esp_ids.sort()  # Keep sorted
            self.esp_status[esp_id] = ESPStatus(esp_id=esp_id)
            self.esp_mac_mapping[esp_id] = mac_address
            self._refresh()

    def start(self):
        """Start the live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=2,
            transient=False
        )
        self._live.start()

    def stop(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_buffer(self, esp_id: int, buffer_seconds: float, files_received: int):
        """Update buffer status for an ESP."""
        if esp_id not in self.esp_status:
            self.add_esp(esp_id)
        if esp_id in self.esp_status:
            status = self.esp_status[esp_id]
            status.buffer_seconds = buffer_seconds
            status.files_received = files_received
            status.last_update = datetime.now()
            self._refresh()

    def on_chunk_analyzed(self, esp_id: int):
        """Called when a chunk is analyzed."""
        if esp_id in self.esp_status:
            self.esp_status[esp_id].chunks_analyzed += 1
            self._refresh()

    def on_detection(self, esp_id: int, species: str, confidence: float):
        """Called when a bird is detected."""
        self.total_detections += 1

        if esp_id in self.esp_status:
            status = self.esp_status[esp_id]
            status.current_birds[species] = confidence

            # Add to recent events
            event = f"🐦 {species[:20]} ({confidence:.0%})"
            status.recent_events.append(event)

        self._refresh()

    def on_arrival(self, esp_id: int, species: str, confidence: float):
        """Called when a new bird arrives."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if esp_id in self.esp_status:
            status = self.esp_status[esp_id]
            status.current_birds[species] = confidence

            event = f"[green]▶ {species[:18]}[/green]"
            status.recent_events.append(event)

        # Global log
        self.event_log.append(
            f"[dim]{timestamp}[/dim] [green]▶[/green] ESP{esp_id} [bold]{species}[/bold] ({confidence:.0%})"
        )
        self._refresh()

    def on_departure(self, esp_id: int, species: str, duration: float):
        """Called when a bird departs."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if esp_id in self.esp_status:
            status = self.esp_status[esp_id]
            if species in status.current_birds:
                del status.current_birds[species]

            event = f"[red]◀ {species[:18]}[/red]"
            status.recent_events.append(event)

        # Global log
        self.event_log.append(
            f"[dim]{timestamp}[/dim] [red]◀[/red] ESP{esp_id} [bold]{species}[/bold] ({duration:.0f}s)"
        )
        self._refresh()

    def _refresh(self):
        """Refresh the display."""
        if self._live:
            self._live.update(self._build_display())

    def _build_display(self) -> Panel:
        """Build the complete display layout."""
        # Create ESP columns table
        esp_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            expand=True
        )

        # Add column for each ESP
        for esp_id in self.esp_ids:
            esp_table.add_column(f"ESP {esp_id}", justify="center", min_width=25)

        # Row 1: Buffer status
        buffer_row = []
        for esp_id in self.esp_ids:
            status = self.esp_status[esp_id]
            progress = min(status.buffer_seconds / 30.0, 1.0)  # 30s target
            bar_width = 15
            filled = int(progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            if progress >= 1.0:
                color = "green"
            elif progress >= 0.5:
                color = "yellow"
            else:
                color = "white"

            buffer_row.append(
                f"[{color}]{bar}[/{color}]\n"
                f"[dim]{status.buffer_seconds:.1f}s / 30s[/dim]\n"
                f"[dim]📁 {status.files_received} fichiers[/dim]"
            )
        esp_table.add_row(*buffer_row)

        # Row 2: Current birds
        birds_row = []
        for esp_id in self.esp_ids:
            status = self.esp_status[esp_id]
            if status.current_birds:
                birds_text = "\n".join(
                    f"[green]● {species[:18]}[/green] ({conf:.0%})"
                    for species, conf in status.current_birds.items()
                )
            else:
                birds_text = "[dim]Aucun oiseau[/dim]"
            birds_row.append(birds_text)
        esp_table.add_row(*birds_row)

        # Row 3: Recent events
        events_row = []
        for esp_id in self.esp_ids:
            status = self.esp_status[esp_id]
            if status.recent_events:
                events_text = "\n".join(list(status.recent_events)[-3:])
            else:
                events_text = "[dim]─[/dim]"
            events_row.append(events_text)
        esp_table.add_row(*events_row)

        # Build event log panel
        if self.event_log:
            log_text = "\n".join(list(self.event_log)[-6:])
        else:
            log_text = "[dim]En attente d'événements...[/dim]"

        event_panel = Panel(
            log_text,
            title="[bold]Événements récents[/bold]",
            border_style="dim"
        )

        # Stats line
        elapsed = (datetime.now() - self.start_time).total_seconds()
        stats_text = Text()
        stats_text.append(f"⏱ {elapsed:.0f}s", style="dim")
        stats_text.append("  │  ", style="dim")
        stats_text.append(f"Détections: {self.total_detections}", style="cyan")

        # Combine everything
        content = Group(
            esp_table,
            Text(""),
            event_panel,
            Text(""),
            stats_text
        )

        return Panel(
            content,
            title="[bold cyan]🐦 InterNest - Détection Multi-ESP[/bold cyan]",
            subtitle="[dim]Ctrl+C pour arrêter[/dim]",
            border_style="cyan"
        )

    def print_summary(self):
        """Print final summary after stopping."""
        self.console.print("\n")

        # Summary table
        table = Table(title="Résumé par ESP", show_header=True, header_style="bold")
        table.add_column("ESP", style="cyan")
        table.add_column("Fichiers", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Espèces détectées")

        all_species = set()
        for esp_id in self.esp_ids:
            status = self.esp_status[esp_id]
            species_list = ", ".join(status.current_birds.keys()) if status.current_birds else "-"
            all_species.update(status.current_birds.keys())

            table.add_row(
                f"ESP {esp_id}",
                str(status.files_received),
                str(status.chunks_analyzed),
                species_list
            )

        self.console.print(table)

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.console.print(f"\n[cyan]Durée totale:[/cyan] {elapsed:.0f}s")
        self.console.print(f"[cyan]Détections totales:[/cyan] {self.total_detections}")
        self.console.print(f"[cyan]Espèces uniques:[/cyan] {len(all_species)}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
