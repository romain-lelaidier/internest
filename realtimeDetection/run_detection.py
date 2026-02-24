#!/usr/bin/env python3
"""
InterNest Bird Detection - Real-time bird species detection using BirdNET-Analyzer.

Usage:
    python run_detection.py -i recording.wav
    python run_detection.py -i recording.wav --lat 48.85 --lon 2.35 --realtime
    python run_detection.py -i recording.wav -o results.json
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel

from src.config import (
    PipelineConfig, AudioConfig, BirdNETConfig,
    StreamingConfig, OutputConfig, FilterConfig,
    Detection, AnalysisResult
)
from src.audio_stream import AudioStreamProcessor
from src.birdnet_analyzer import BirdNETAnalyzerWrapper
from src.detection_tracker import DetectionTracker
from src.output_handler import (
    CLIOutputHandler, JSONFileHandler, CSVFileHandler,
    SimpleOutputHandler, OutputHandler
)
from src.presence_tracker import PresenceTracker

app = typer.Typer(
    name="bird-detect",
    help="Real-time bird detection using BirdNET-Analyzer"
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BirdDetectionPipeline:
    """
    Real-time bird detection pipeline using BirdNET-Analyzer.

    Workflow:
    1. Load audio file
    2. Process in overlapping 3-second chunks
    3. Run BirdNET analysis on each chunk
    4. Aggregate and deduplicate detections
    5. Output results in real-time
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stream_processor = AudioStreamProcessor(
            config.streaming,
            target_sample_rate=config.audio.sample_rate
        )
        self.analyzer = BirdNETAnalyzerWrapper(config.birdnet)
        self.tracker = DetectionTracker(config.filtering)
        self.output_handlers: List[OutputHandler] = []

    def _init_output_handlers(self, total_duration: float) -> List[OutputHandler]:
        """Initialize output handlers based on config."""
        handlers = []

        if self.config.output.mode in ('cli', 'all'):
            cli_handler = CLIOutputHandler(
                tracker=self.tracker,
                input_file=self.config.audio.input_path,
                latitude=self.config.birdnet.latitude,
                longitude=self.config.birdnet.longitude
            )
            cli_handler.start(total_duration)
            handlers.append(cli_handler)
        elif self.config.output.mode == 'simple':
            handlers.append(SimpleOutputHandler())

        if self.config.output.json_path:
            handlers.append(JSONFileHandler(self.config.output.json_path))

        if self.config.output.csv_path:
            handlers.append(CSVFileHandler(self.config.output.csv_path))

        return handlers

    def run(self) -> AnalysisResult:
        """Run the complete detection pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Bird Detection Pipeline")
        logger.info(f"Input: {self.config.audio.input_path}")
        logger.info("=" * 60)

        # Get audio duration for progress tracking
        info = self.stream_processor.get_audio_info(self.config.audio.input_path)
        total_duration = info.duration_seconds

        logger.info(f"Audio duration: {total_duration:.1f}s")

        # Initialize output handlers
        self.output_handlers = self._init_output_handlers(total_duration)

        all_detections = []

        try:
            # Process audio in chunks
            for chunk in self.stream_processor.stream_file(self.config.audio.input_path):

                # Optionally simulate real-time by adding delay
                if self.config.streaming.simulate_realtime:
                    delay = (
                        self.config.streaming.chunk_duration_seconds -
                        self.config.streaming.overlap_seconds
                    ) / self.config.streaming.realtime_speed
                    time.sleep(delay)

                # Analyze chunk with BirdNET
                detections = self.analyzer.analyze_chunk(chunk, self.stream_processor)

                # Filter and deduplicate
                new_detections = self.tracker.process_detections(detections)

                # Notify output handlers
                for detection in new_detections:
                    for handler in self.output_handlers:
                        handler.on_detection(detection)

                # Update progress
                for handler in self.output_handlers:
                    handler.on_progress(chunk.end_time, total_duration)
                    handler.on_chunk_complete(chunk.chunk_index)

                all_detections.extend(new_detections)

        finally:
            # Clean up temporary files
            self.stream_processor.cleanup_temp_files()

        # Create result
        result = AnalysisResult(
            detections=all_detections,
            duration=total_duration,
            config=self.config
        )

        # Finalize outputs
        for handler in self.output_handlers:
            handler.finalize(result)

        return result


@app.command()
def analyze(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Audio file to analyze (WAV, MP3, FLAC, OGG)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="JSON configuration file"
    ),
    lat: Optional[float] = typer.Option(
        None, "--lat",
        help="Latitude for geographic species filtering"
    ),
    lon: Optional[float] = typer.Option(
        None, "--lon",
        help="Longitude for geographic species filtering"
    ),
    min_conf: float = typer.Option(
        0.7, "--min-conf",
        help="Minimum confidence for BirdNET (0.0-1.0)"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t",
        help="Confidence threshold for display (0.0-1.0)"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSON file path"
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--csv",
        help="Output CSV file path"
    ),
    realtime: bool = typer.Option(
        False, "--realtime", "-r",
        help="Simulate real-time processing with delays"
    ),
    speed: float = typer.Option(
        1.0, "--speed", "-s",
        help="Playback speed for real-time mode (e.g., 2.0 = 2x faster)"
    ),
    simple: bool = typer.Option(
        False, "--simple",
        help="Use simple output instead of rich dashboard"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Analyze audio file for bird species using BirdNET.

    Examples:

        # Basic analysis
        python run_detection.py -i recording.wav

        # With geographic filtering (improves accuracy)
        python run_detection.py -i recording.wav --lat 48.85 --lon 2.35

        # Real-time simulation mode
        python run_detection.py -i recording.wav --realtime

        # Export results to JSON
        python run_detection.py -i recording.wav -o results.json

        # Higher confidence threshold
        python run_detection.py -i recording.wav --threshold 0.7
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Load config from file or build from arguments
    if config_file and config_file.exists():
        try:
            pipeline_config = PipelineConfig.from_json(str(config_file))
            # Override with CLI arguments if provided
            pipeline_config.audio.input_path = str(input_file)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise typer.Exit(1)
    else:
        pipeline_config = PipelineConfig(
            audio=AudioConfig(input_path=str(input_file)),
            birdnet=BirdNETConfig(
                min_confidence=min_conf,
                latitude=lat,
                longitude=lon,
                date=datetime.now()
            ),
            streaming=StreamingConfig(
                simulate_realtime=realtime,
                realtime_speed=speed
            ),
            output=OutputConfig(
                mode="simple" if simple else "cli",
                json_path=str(output_json) if output_json else None,
                csv_path=str(output_csv) if output_csv else None
            ),
            filtering=FilterConfig(confidence_threshold=threshold)
        )

    # Run pipeline
    try:
        console.print("[cyan]Initializing BirdNET analyzer...[/cyan]")
        pipeline = BirdDetectionPipeline(pipeline_config)
        result = pipeline.run()

        # Print output file locations
        if output_json:
            console.print(f"\n[green]Results saved to: {output_json}[/green]")
        if output_csv:
            console.print(f"[green]CSV saved to: {output_csv}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def watch(
    input_file: Path = typer.Option(
        ..., "--input", "-i",
        help="Audio file to analyze (simulates a stream)"
    ),
    lat: Optional[float] = typer.Option(
        None, "--lat",
        help="Latitude for geographic species filtering"
    ),
    lon: Optional[float] = typer.Option(
        None, "--lon",
        help="Longitude for geographic species filtering"
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t",
        help="Confidence threshold (0.0-1.0)"
    ),
    departure_timeout: float = typer.Option(
        30.0, "--timeout",
        help="Seconds without detection before bird is considered gone"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSON file for events"
    ),
    speed: float = typer.Option(
        1.0, "--speed", "-s",
        help="Playback speed (1.0 = realtime, 2.0 = 2x faster)"
    )
):
    """
    Watch mode: monitor audio stream for bird arrivals and departures.

    Simulates real-time streaming and alerts when:
    - A new bird species ARRIVES (first detection)
    - A bird species DEPARTS (not detected for --timeout seconds)

    Examples:
        python run_detection.py watch -i recording.mp3
        python run_detection.py watch -i recording.mp3 --speed 5
        python run_detection.py watch -i recording.mp3 --timeout 15
    """
    import json

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Initializing BirdNET analyzer...[/cyan]")

    # Initialize components
    streaming_config = StreamingConfig(
        chunk_duration_seconds=30.0,
        overlap_seconds=3.0,
        simulate_realtime=True,
        realtime_speed=speed
    )
    stream_processor = AudioStreamProcessor(streaming_config, target_sample_rate=48000)

    birdnet_config = BirdNETConfig(
        min_confidence=0.1,  # Low threshold, we filter later
        latitude=lat,
        longitude=lon,
        date=datetime.now()
    )
    analyzer = BirdNETAnalyzerWrapper(birdnet_config)

    filter_config = FilterConfig(confidence_threshold=threshold)
    tracker = DetectionTracker(filter_config)

    presence_tracker = PresenceTracker(
        departure_timeout=departure_timeout,
        console=console
    )

    # Get audio info
    info = stream_processor.get_audio_info(str(input_file))
    console.print(f"[green]Audio chargé: {info.duration_seconds:.1f}s[/green]")
    console.print(f"[dim]Vitesse: {speed}x | Timeout départ: {departure_timeout}s | Seuil: {threshold}[/dim]")
    console.print()
    console.print("[bold]Démarrage du monitoring...[/bold]")
    console.print("[dim]Les alertes apparaîtront ci-dessous[/dim]")
    console.print("─" * 50)

    try:
        for chunk in stream_processor.stream_file(str(input_file)):
            # Simulate real-time
            delay = (streaming_config.chunk_duration_seconds - streaming_config.overlap_seconds) / speed
            time.sleep(delay)

            # Analyze chunk
            detections = analyzer.analyze_chunk(chunk, stream_processor)
            filtered = tracker.process_detections(detections)

            # Update presence tracker (this triggers alerts)
            presence_tracker.update(filtered, chunk.end_time)

            # Show current time
            mins = int(chunk.end_time // 60)
            secs = chunk.end_time % 60
            console.print(f"[dim]⏱ {mins:02d}:{secs:04.1f} | Présents: {len(presence_tracker.present_birds)}[/dim]", end="\r")

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrompu[/yellow]")
    finally:
        stream_processor.cleanup_temp_files()

    # Finalize - mark remaining birds as departed
    console.print("\n" + "─" * 50)
    console.print("[bold]Fin du stream[/bold]")
    presence_tracker.finalize()

    # Summary
    summary = presence_tracker.get_summary()
    console.print()
    console.print(Panel(
        f"[cyan]Espèces détectées:[/cyan] {summary['unique_species']}\n"
        f"[green]Arrivées:[/green] {summary['total_arrivals']}\n"
        f"[red]Départs:[/red] {summary['total_departures']}\n\n"
        f"[bold]Espèces:[/bold] {', '.join(summary['species_list']) or 'Aucune'}",
        title="Résumé",
        border_style="blue"
    ))

    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Events saved to: {output_json}[/green]")


@app.command()
def stream(
    watch_dir: Path = typer.Option(
        ..., "--dir", "-d",
        help="Directory to watch for .bin audio files"
    ),
    lat: Optional[float] = typer.Option(
        None, "--lat",
        help="Latitude for geographic species filtering"
    ),
    lon: Optional[float] = typer.Option(
        None, "--lon",
        help="Longitude for geographic species filtering"
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t",
        help="Confidence threshold (0.0-1.0)"
    ),
    departure_timeout: float = typer.Option(
        30.0, "--timeout",
        help="Seconds without detection before bird is considered gone"
    ),
    chunk_duration: float = typer.Option(
        15.0, "--chunk",
        help="Audio chunk duration in seconds for analysis"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output JSON file for events"
    ),
    no_existing: bool = typer.Option(
        False, "--no-existing",
        help="Skip processing existing files in the directory"
    ),
    web: bool = typer.Option(
        False, "--web", "-w",
        help="Enable web interface accessible from the local network"
    ),
    web_port: int = typer.Option(
        5000, "--port", "-p",
        help="Port for web interface (default: 5000)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Real-time stream mode: monitor directory for ESP32 .bin audio files.

    Watches a directory for new .bin files from ESP32 devices,
    buffers audio per device, and runs BirdNET analysis in pseudo real-time.

    File naming convention: {mac_address}_{unix_timestamp}.bin
    File format: PCM 16-bit signed, 44.1kHz, mono

    ESP IDs are assigned dynamically as new MAC addresses are discovered.

    Examples:
        # Monitor directory
        python run_detection.py stream -d /path/to/audio/

        # With geographic filtering
        python run_detection.py stream -d /path/to/audio/ --lat 48.85 --lon 2.35

        # Higher confidence threshold
        python run_detection.py stream -d /path/to/audio/ --threshold 0.7

        # Enable web interface (accessible from other devices on the network)
        python run_detection.py stream -d /path/to/audio/ --web

        # Web interface on custom port
        python run_detection.py stream -d /path/to/audio/ --web --port 8080
    """
    from src.realtime_pipeline import RealtimeBirdDetector, RealtimePipelineConfig

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Quiet mode for clean display
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger('src').setLevel(logging.WARNING)

    # Validate directory
    if not watch_dir.exists():
        console.print(f"[red]Error: Directory not found: {watch_dir}[/red]")
        raise typer.Exit(1)

    if not watch_dir.is_dir():
        console.print(f"[red]Error: Not a directory: {watch_dir}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Initializing real-time bird detection pipeline...[/cyan]")
    console.print(f"[dim]Watching: {watch_dir}[/dim]")
    console.print(f"[dim]Threshold: {threshold} | Timeout: {departure_timeout}s[/dim]")
    console.print(f"[dim]ESP IDs will be assigned dynamically as devices are discovered[/dim]")

    # Create configuration
    config = RealtimePipelineConfig(
        watch_dir=watch_dir,
        chunk_duration=chunk_duration,
        min_confidence=0.1,  # Low threshold, we filter later
        confidence_threshold=threshold,
        latitude=lat,
        longitude=lon,
        departure_timeout=departure_timeout,
        process_existing=not no_existing
    )

    # Event storage for JSON output
    events = []

    # Create detector and display
    detector = RealtimeBirdDetector(config)

    from src.live_display import MultiESPDisplay
    display = MultiESPDisplay(console=console)

    # Optional web display
    web_display = None
    if web:
        from src.web_display import WebDisplay
        web_display = WebDisplay(host="0.0.0.0", port=web_port)
        # Get local IP for display
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "localhost"
        console.print(f"[green]Web interface enabled: http://{local_ip}:{web_port}[/green]")

    console.print()

    # Per-ESP file counters
    esp_file_counts = {}

    def on_new_esp(esp_id, mac_address):
        display.add_esp(esp_id, mac_address)
        esp_file_counts[esp_id] = 0
        if web_display:
            web_display.on_new_esp(esp_id, mac_address)

    def on_buffer_update(esp_id, buffer_secs, total_files):
        if esp_id not in esp_file_counts:
            esp_file_counts[esp_id] = 0
        esp_file_counts[esp_id] += 1
        display.update_buffer(esp_id, buffer_secs, esp_file_counts[esp_id])
        if web_display:
            web_display.update_buffer(esp_id, buffer_secs, esp_file_counts[esp_id])

    def on_chunk_analyzed(esp_id):
        display.on_chunk_analyzed(esp_id)
        if web_display:
            web_display.on_chunk_analyzed(esp_id)

    def on_detection(event):
        d = event.detection
        display.on_detection(event.esp_id, d.common_name, d.confidence)
        if web_display:
            web_display.on_detection(event.esp_id, d.common_name, d.confidence)
        events.append({
            'type': 'detection',
            'esp_id': event.esp_id,
            'species': d.common_name,
            'scientific_name': d.scientific_name,
            'confidence': d.confidence,
            'timestamp': event.timestamp.isoformat()
        })

    def on_arrival(esp_id, species, confidence):
        display.on_arrival(esp_id, species, confidence)
        if web_display:
            web_display.on_arrival(esp_id, species, confidence)
        events.append({
            'type': 'arrival',
            'esp_id': esp_id,
            'species': species,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    def on_departure(esp_id, species, duration):
        display.on_departure(esp_id, species, duration)
        if web_display:
            web_display.on_departure(esp_id, species, duration)
        events.append({
            'type': 'departure',
            'esp_id': esp_id,
            'species': species,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

    detector.on_new_esp(on_new_esp)
    detector.on_buffer_update(on_buffer_update)
    detector.on_chunk_analyzed(on_chunk_analyzed)
    detector.on_detection(on_detection)
    detector.on_arrival(on_arrival)
    detector.on_departure(on_departure)

    try:
        display.start()
        if web_display:
            web_display.start()
        detector.start()

        # Main loop - just keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        # Get stats before stopping
        stats = detector.get_stats()
        detector.stop()
        display.stop()
        if web_display:
            web_display.stop()

    # Final summary
    display.print_summary()

    # Save to JSON if requested
    if output_json:
        import json
        output_data = {
            'config': {
                'watch_dir': str(watch_dir),
                'threshold': threshold,
                'departure_timeout': departure_timeout
            },
            'esp_mapping': display.esp_mac_mapping,
            'stats': {
                'files_processed': stats['files_processed'],
                'chunks_analyzed': stats['chunks_analyzed'],
                'total_detections': stats['total_detections']
            },
            'events': events
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Events saved to: {output_json}[/green]")


@app.command()
def info():
    """Show information about the BirdNET analyzer."""
    console.print("[cyan]InterNest Bird Detection[/cyan]")
    console.print("Using BirdNET-Analyzer for species identification\n")

    try:
        from birdnetlib.analyzer import Analyzer
        console.print("[green]BirdNET is installed and available[/green]")

        # Try to load the analyzer
        console.print("Loading analyzer model...")
        analyzer = Analyzer()
        console.print(f"[green]Model loaded successfully[/green]")

    except ImportError:
        console.print("[red]birdnetlib is not installed[/red]")
        console.print("Install with: pip install birdnetlib")
    except Exception as e:
        console.print(f"[red]Error loading analyzer: {e}[/red]")


if __name__ == "__main__":
    app()
