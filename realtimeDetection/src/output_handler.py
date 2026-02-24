"""Output handlers for real-time detection display."""

import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from .config import Detection, AnalysisResult
from .detection_tracker import DetectionTracker
from .audio_stream import format_timestamp


class OutputHandler(ABC):
    """Base class for detection output handlers."""

    @abstractmethod
    def on_detection(self, detection: Detection) -> None:
        """Handle a new detection."""
        pass

    @abstractmethod
    def on_progress(self, current_time: float, total_duration: float) -> None:
        """Update progress indicator."""
        pass

    @abstractmethod
    def on_chunk_complete(self, chunk_index: int) -> None:
        """Called when a chunk analysis is complete."""
        pass

    @abstractmethod
    def finalize(self, result: AnalysisResult) -> None:
        """Finalize output (save files, close connections)."""
        pass


class CLIOutputHandler(OutputHandler):
    """Rich terminal dashboard output."""

    def __init__(
        self,
        tracker: DetectionTracker,
        input_file: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ):
        self.console = Console()
        self.tracker = tracker
        self.input_file = Path(input_file).name
        self.latitude = latitude
        self.longitude = longitude
        self.current_time = 0.0
        self.total_duration = 0.0
        self.live: Optional[Live] = None

    def start(self, total_duration: float):
        """Start the live display."""
        self.total_duration = total_duration
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4
        )
        self.live.start()

    def _build_display(self) -> Panel:
        """Build the dashboard display."""
        # Header
        progress_pct = (self.current_time / self.total_duration * 100) if self.total_duration > 0 else 0
        progress_bar = self._build_progress_bar(progress_pct)

        header_text = Text()
        header_text.append(f"Fichier: {self.input_file}", style="cyan")
        header_text.append("   ")
        header_text.append(f"Progression: {format_timestamp(self.current_time)} / {format_timestamp(self.total_duration)}", style="green")

        if self.latitude and self.longitude:
            header_text.append("\n")
            header_text.append(f"Position: {self.latitude:.4f}N, {self.longitude:.4f}E", style="yellow")

        # Recent detections table
        recent_table = Table(title="Detections Recentes", show_header=True, header_style="bold magenta")
        recent_table.add_column("Heure", style="cyan", width=10)
        recent_table.add_column("Espece", style="green", width=25)
        recent_table.add_column("Nom scientifique", style="dim", width=25)
        recent_table.add_column("Confiance", justify="right", style="yellow", width=10)

        for det in self.tracker.get_recent_detections(8):
            conf_style = "green" if det.confidence >= 0.8 else "yellow" if det.confidence >= 0.5 else "red"
            recent_table.add_row(
                format_timestamp(det.start_time),
                det.common_name,
                det.scientific_name,
                Text(f"{det.confidence:.2f}", style=conf_style)
            )

        # Species summary table
        summary = self.tracker.get_summary()
        summary_table = Table(title="Resume par Espece", show_header=True, header_style="bold blue")
        summary_table.add_column("Espece", style="green", width=25)
        summary_table.add_column("Detections", justify="right", style="cyan", width=12)
        summary_table.add_column("Conf. moy.", justify="right", style="yellow", width=12)
        summary_table.add_column("Premiere", style="dim", width=10)
        summary_table.add_column("Derniere", style="dim", width=10)

        for species in summary['species'][:6]:
            summary_table.add_row(
                species['common_name'],
                str(species['count']),
                f"{species['avg_confidence']:.2f}",
                format_timestamp(species['first_seen']),
                format_timestamp(species['last_seen'])
            )

        # Stats line
        stats_text = Text()
        stats_text.append(f"Total: {summary['total_detections']} détections", style="bold cyan")
        stats_text.append("  |  ")
        stats_text.append(f"{summary['unique_species']} espèces uniques", style="bold green")

        # Combine into layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(header_text, title="InterNest - Live detection"), size=5),
            Layout(Panel(progress_bar), size=3),
            Layout(recent_table, size=12),
            Layout(summary_table, size=10),
            Layout(Panel(stats_text), size=3)
        )

        return Panel(layout, border_style="blue")

    def _build_progress_bar(self, percentage: float) -> Text:
        """Build a progress bar."""
        width = 50
        filled = int(width * percentage / 100)
        bar = "[" + "=" * filled + ">" + " " * (width - filled - 1) + "]"
        return Text(f"{bar} {percentage:.1f}%", style="cyan")

    def on_detection(self, detection: Detection) -> None:
        """Handle a new detection."""
        if self.live:
            self.live.update(self._build_display())

    def on_progress(self, current_time: float, total_duration: float) -> None:
        """Update progress indicator."""
        self.current_time = current_time
        self.total_duration = total_duration
        if self.live:
            self.live.update(self._build_display())

    def on_chunk_complete(self, chunk_index: int) -> None:
        """Called when a chunk analysis is complete."""
        if self.live:
            self.live.update(self._build_display())

    def finalize(self, result: AnalysisResult) -> None:
        """Finalize output."""
        if self.live:
            self.live.stop()

        # Print final summary
        self.console.print("\n")
        self.console.print(Panel(
            f"[bold green]Analyse terminée![/bold green]\n\n"
            f"Durée: {format_timestamp(result.duration)}\n"
            f"Détections: {len(result.detections)}\n"
            f"Espèces: {len(set(d.common_name for d in result.detections))}",
            title="Résumé Final",
            border_style="green"
        ))


class JSONFileHandler(OutputHandler):
    """JSON output to file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.detections: List[dict] = []

    def on_detection(self, detection: Detection) -> None:
        """Handle a new detection."""
        self.detections.append(detection.to_dict())

    def on_progress(self, current_time: float, total_duration: float) -> None:
        """Update progress indicator."""
        pass

    def on_chunk_complete(self, chunk_index: int) -> None:
        """Called when a chunk analysis is complete."""
        pass

    def finalize(self, result: AnalysisResult) -> None:
        """Save results to JSON file."""
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': result.duration,
                'total_detections': len(result.detections),
                'unique_species': len(set(d.common_name for d in result.detections))
            },
            'detections': self.detections
        }

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


class CSVFileHandler(OutputHandler):
    """CSV output to file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.detections: List[Detection] = []

    def on_detection(self, detection: Detection) -> None:
        """Handle a new detection."""
        self.detections.append(detection)

    def on_progress(self, current_time: float, total_duration: float) -> None:
        """Update progress indicator."""
        pass

    def on_chunk_complete(self, chunk_index: int) -> None:
        """Called when a chunk analysis is complete."""
        pass

    def finalize(self, result: AnalysisResult) -> None:
        """Save results to CSV file."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'start_time', 'end_time',
                'common_name', 'scientific_name', 'confidence'
            ])

            for det in self.detections:
                writer.writerow([
                    format_timestamp(det.start_time),
                    det.start_time,
                    det.end_time,
                    det.common_name,
                    det.scientific_name,
                    det.confidence
                ])


class SimpleOutputHandler(OutputHandler):
    """Simple console output without live display."""

    def __init__(self):
        self.console = Console()

    def on_detection(self, detection: Detection) -> None:
        """Handle a new detection."""
        conf_style = "green" if detection.confidence >= 0.8 else "yellow" if detection.confidence >= 0.5 else "red"
        self.console.print(
            f"[cyan]{format_timestamp(detection.start_time)}[/cyan] "
            f"[green]{detection.common_name}[/green] "
            f"[dim]({detection.scientific_name})[/dim] "
            f"[{conf_style}]{detection.confidence:.2f}[/{conf_style}]"
        )

    def on_progress(self, current_time: float, total_duration: float) -> None:
        """Update progress indicator."""
        pass

    def on_chunk_complete(self, chunk_index: int) -> None:
        """Called when a chunk analysis is complete."""
        pass

    def finalize(self, result: AnalysisResult) -> None:
        """Finalize output."""
        self.console.print(f"\n[bold]Analyse terminee: {len(result.detections)} detections[/bold]")
