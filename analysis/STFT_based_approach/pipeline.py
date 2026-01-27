"""
Pipeline principal de reconstruction de trajectoires.
"""

import numpy as np
import soundfile as sf
import json
from pathlib import Path
from scipy.optimize import least_squares
from ultralytics import YOLO

from config import Config
from detection import (
    generate_spectrogram,
    detect_on_spectrogram,
    cluster_detections,
    compute_global_cluster_map
)
from localization import (
    gcc_phat_band,
    compute_delay_envelope,
    tdoa_residuals,
    find_best_initial_guess,
    compute_position_with_subsets
)
from tracking import TrackManager, filter_confidence_trajectories
from visualization import (
    visualize_spectrograms,
    visualize_trajectories_3d,
    visualize_confidence,
    visualize_confidence_2d
)


class TrajectoryReconstructor:
    """Pipeline complet de reconstruction de trajectoires."""

    def __init__(self, input_dir: str, output_dir: str = None, verbose: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "reconstruction"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.audio_buffers = {}
        self.duration = None
        self.all_detections = {}
        self.clustered_data = {}
        self.global_map = {}
        self.tracker = TrackManager()
        self.debug_data = []
        self.confidence_data = {}

        # Charger le modèle YOLO
        print(f"Chargement du modèle YOLO: {Config.MODEL_PATH}")
        self.model = YOLO(Config.MODEL_PATH)

    def load_audio(self):
        """Charge les fichiers audio de tous les micros."""
        print("\n[1/6] Chargement des fichiers audio...")

        for i in range(len(Config.MICROS)):
            path = self.input_dir / f"mix_mic_{i}.wav"
            if not path.exists():
                raise FileNotFoundError(f"Fichier audio manquant: {path}")

            audio, fs = sf.read(str(path))
            self.audio_buffers[i] = audio

            if self.duration is None:
                self.duration = len(audio) / fs

            print(f"  - Micro {i}: {len(audio)} samples ({self.duration:.2f}s)")

        print(f"  Total: {len(self.audio_buffers)} micros chargés")

    def generate_spectrograms(self):
        """Génère les spectrogrammes pour chaque micro."""
        print("\n[2/6] Génération des spectrogrammes...")

        spectro_dir = self.output_dir / "spectrograms"
        spectro_dir.mkdir(exist_ok=True)

        self.spectro_paths = {}

        for mic_id, audio in self.audio_buffers.items():
            path = spectro_dir / f"mic_{mic_id}.png"
            shape = generate_spectrogram(audio, Config.FS, str(path))
            self.spectro_paths[mic_id] = str(path)
            print(f"  - Micro {mic_id}: {shape[1]}x{shape[0]} pixels")

    def detect_all(self):
        """Exécute la détection YOLO sur tous les spectrogrammes."""
        print("\n[3/6] Détection YOLO sur les spectrogrammes...")

        for mic_id, spectro_path in self.spectro_paths.items():
            detections = detect_on_spectrogram(spectro_path, self.model, self.duration)
            self.all_detections[mic_id] = detections
            print(f"  - Micro {mic_id}: {len(detections)} détections")

        total = sum(len(d) for d in self.all_detections.values())
        print(f"  Total: {total} détections")

    def cluster_all(self):
        """Clusterise les détections par micro puis globalement."""
        print("\n[4/6] Clustering des détections...")

        for mic_id, detections in self.all_detections.items():
            if len(detections) == 0:
                continue

            labels, n_clusters = cluster_detections(detections)

            for i, det in enumerate(detections):
                det['cluster'] = int(labels[i])

            self.clustered_data[mic_id] = {
                'detections': detections,
                'n_clusters': n_clusters,
                'n_noise': np.sum(labels == -1)
            }

            print(f"  - Micro {mic_id}: {n_clusters} clusters, {np.sum(labels == -1)} bruit")

        self.global_map = compute_global_cluster_map(self.clustered_data)
        n_birds = len(set(self.global_map.values()) - {-1})
        print(f"  Oiseaux identifiés: {n_birds}")

    def group_events(self) -> dict:
        """Regroupe les détections en événements (chirps) par oiseau."""
        chirp_events = {}

        for mic_id, dets in self.all_detections.items():
            for det in dets:
                local_cluster = det.get('cluster', -1)
                bird_id = self.global_map.get((mic_id, local_cluster), -1)

                if bird_id == -1:
                    continue

                if bird_id not in chirp_events:
                    chirp_events[bird_id] = []

                t_center = (det['t_start'] + det['t_end']) / 2

                found_event = None
                for event in chirp_events[bird_id]:
                    times = [(d['t_start'] + d['t_end']) / 2 for d in event.values()]
                    avg_time = sum(times) / len(times)

                    if abs(t_center - avg_time) < Config.TIME_TOL:
                        found_event = event
                        break

                if found_event is not None:
                    found_event[mic_id] = det
                else:
                    chirp_events[bird_id].append({mic_id: det})

        return chirp_events

    def compute_trajectories(self):
        """Calcule les positions 3D par TDOA et construit les trajectoires."""
        print("\n[5/6] Calcul des trajectoires (TDOA + Triangulation)...")

        chirp_events = self.group_events()

        for bird_id, events in chirp_events.items():
            events_sorted = sorted(events, key=lambda e: list(e.values())[0]['t_start'])

            for event in events_sorted:
                if len(event) < Config.MIN_MICS:
                    continue

                ref_mic = list(event.keys())[0]
                ref_det = event[ref_mic]

                f_min_avg = np.mean([d['f_min'] for d in event.values()])
                f_max_avg = np.mean([d['f_max'] for d in event.values()])

                all_times = []
                for det in event.values():
                    all_times.append(det['t_start'])
                    all_times.append(det['t_end'])

                t_min = min(all_times) - 0.1
                t_max = max(all_times) + 0.1

                s_common = int(max(0, t_min * Config.FS))
                e_common = int(min(len(self.audio_buffers[0]), t_max * Config.FS))

                if e_common - s_common < 1000:
                    continue

                sig_ref = self.audio_buffers[ref_mic][s_common:e_common]

                delays = np.full(len(Config.MICROS), np.nan)
                delays[ref_mic] = 0.0

                ref_t_center = ref_det['t_center']

                for target_mic, target_det in event.items():
                    if target_mic == ref_mic:
                        continue

                    sig_target = self.audio_buffers[target_mic][s_common:e_common]

                    if len(sig_target) < 100:
                        continue

                    tau0 = target_det['t_center'] - ref_t_center

                    tau1, score1 = gcc_phat_band(
                        sig_target, sig_ref,
                        Config.FS, f_min_avg, f_max_avg
                    )

                    tau2, score2 = compute_delay_envelope(
                        sig_target, sig_ref,
                        Config.FS, f_min_avg, f_max_avg
                    )

                    if abs(tau0) < 0.5:
                        if score1 > Config.MIN_TDOA_SCORE and abs(tau1 - tau0) < 0.05:
                            delays[target_mic] = (tau0 + tau1) / 2
                        elif score2 > 0.1 and abs(tau2 - tau0) < 0.05:
                            delays[target_mic] = (tau0 + tau2) / 2
                        else:
                            delays[target_mic] = tau0
                    elif score2 > 0.1:
                        delays[target_mic] = tau2
                    elif score1 > Config.MIN_TDOA_SCORE:
                        delays[target_mic] = tau1

                n_valid_delays = np.sum(~np.isnan(delays))
                if n_valid_delays >= Config.MIN_MICS:
                    if bird_id in self.tracker.smoothed_last_pos:
                        initial_guess = self.tracker.smoothed_last_pos[bird_id]
                    else:
                        initial_guess = find_best_initial_guess(
                            Config.MICROS, delays, ref_mic
                        )

                    try:
                        res = least_squares(
                            tdoa_residuals, initial_guess,
                            args=(Config.MICROS, delays, ref_mic),
                            bounds=([0, 0, 0], [100, 100, 50]),
                            ftol=1e-4,
                            max_nfev=200
                        )

                        threshold = Config.COST_THRESHOLD * (8 / n_valid_delays)

                        if res.cost < threshold:
                            t_avg = (ref_det['t_start'] + ref_det['t_end']) / 2
                            self.tracker.add_point(res.x, t_avg, bird_id)

                            conf_result = compute_position_with_subsets(
                                delays, ref_mic, res.x,
                                subset_size=5, max_subsets=20
                            )
                            if conf_result is not None:
                                if bird_id not in self.confidence_data:
                                    self.confidence_data[bird_id] = []
                                self.confidence_data[bird_id].append({
                                    't': t_avg,
                                    'main_pos': conf_result['main_pos'],
                                    'subset_positions': conf_result['subset_positions'],
                                    'std_xyz': conf_result['std_xyz'],
                                    'confidence': conf_result['confidence']
                                })

                            if self.verbose:
                                self.debug_data.append({
                                    'bird_id': bird_id,
                                    't': t_avg,
                                    'pos': res.x.tolist(),
                                    'cost': res.cost,
                                    'n_mics': n_valid_delays,
                                    'delays': delays.tolist()
                                })
                    except Exception as e:
                        if self.verbose:
                            print(f"    Erreur triangulation: {e}")

        for bird_id, points in self.tracker.tracks.items():
            print(f"  - Oiseau {bird_id}: {len(points)} positions")

    def save_results(self):
        """Sauvegarde les résultats en JSON."""
        trajectories = self.tracker.get_smoothed_trajectories()

        results = {
            'config': {
                'input_dir': str(self.input_dir),
                'n_micros': len(Config.MICROS),
                'duration': self.duration
            },
            'detections': {
                mic_id: len(dets) for mic_id, dets in self.all_detections.items()
            },
            'n_birds': len(trajectories),
            'trajectories': {
                str(bird_id): data.tolist()
                for bird_id, data in trajectories.items()
            }
        }

        output_path = self.output_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nRésultats sauvegardés: {output_path}")

    def visualize_all(self):
        """Génère toutes les visualisations."""
        # Spectrogrammes avec détections
        print("\n  Génération de la grille des spectrogrammes...")
        spectro_path = self.output_dir / "spectrograms_detections.png"
        visualize_spectrograms(
            self.audio_buffers, self.all_detections,
            self.global_map, self.duration, str(spectro_path)
        )
        print(f"  Sauvegardé: {spectro_path}")

        # Trajectoires 3D
        print("\n[6/6] Génération de la visualisation...")
        traj_path = self.output_dir / "trajectories_3d.png"
        truth_path = self.input_dir / "trajectory_truth.txt"
        visualize_trajectories_3d(
            self.tracker.get_smoothed_trajectories(),
            str(traj_path),
            str(truth_path) if truth_path.exists() else None
        )
        print(f"  Sauvegardé: {traj_path}")

        # Visualisation de confiance
        if self.confidence_data:
            print("\n  Filtrage des trajectoires (confiance + vitesse)...")
            filtered_data = filter_confidence_trajectories(self.confidence_data)

            for bird_id, data in filtered_data.items():
                n_total = len(data['points']) + len(data['rejected'])
                n_kept = len(data['points'])
                n_rejected = len(data['rejected'])
                print(f"  Oiseau {bird_id}: {n_kept}/{n_total} points conservés, {n_rejected} rejetés")

            print("\n  Génération de la visualisation de confiance...")
            conf_path = self.output_dir / "trajectories_confidence.png"
            visualize_confidence(filtered_data, str(conf_path))
            print(f"  Sauvegardé: {conf_path}")

            conf_2d_path = self.output_dir / "trajectories_confidence_2d.png"
            visualize_confidence_2d(filtered_data, str(conf_2d_path))
            print(f"  Sauvegardé: {conf_2d_path}")

            # Stats de confiance
            for bird_id, data in filtered_data.items():
                if len(data['points']) > 0:
                    avg_conf = np.mean([cp['confidence'] for cp in data['points']])
                    avg_std = np.mean([np.mean(cp['std_xyz']) for cp in data['points']])
                    print(f"  Oiseau {bird_id} (filtré): confiance={avg_conf:.2f}, std_moyen={avg_std:.1f}m")

    def run(self):
        """Exécute le pipeline complet."""
        print("=" * 60)
        print("RECONSTRUCTION DE TRAJECTOIRES - YOLO + TDOA")
        print("=" * 60)

        self.load_audio()
        self.generate_spectrograms()
        self.detect_all()
        self.cluster_all()
        self.compute_trajectories()
        self.save_results()
        self.visualize_all()

        print("\n" + "=" * 60)
        print("TERMINÉ")
        print("=" * 60)
