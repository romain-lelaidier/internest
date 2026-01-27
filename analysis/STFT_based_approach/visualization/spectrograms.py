"""
Visualisation des spectrogrammes avec détections.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from config import Config


def visualize_spectrograms(audio_buffers: dict, all_detections: dict,
                           global_map: dict, duration: float,
                           output_path: str):
    """
    Affiche une grille compacte des spectrogrammes avec les détections YOLO.

    Args:
        audio_buffers: {mic_id: audio_array}
        all_detections: {mic_id: [detections]}
        global_map: {(mic_id, local_cluster): global_bird_id}
        duration: Durée totale en secondes
        output_path: Chemin de sauvegarde de l'image
    """
    n_mics = len(audio_buffers)
    n_cols = 4
    n_rows = int(np.ceil(n_mics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.ravel()

    # Palette de couleurs pour les oiseaux
    bird_colors = cm.tab10(np.linspace(0, 1, 10))

    for mic_id in range(n_mics):
        ax = axes[mic_id]
        audio = audio_buffers[mic_id]

        # Calcul STFT
        f, t, Zxx = signal.stft(
            audio, Config.FS,
            nperseg=Config.N_FFT,
            noverlap=Config.N_FFT - Config.HOP_LENGTH
        )
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-9)

        # Affichage spectrogramme (jusqu'à Nyquist/2 pour plus de clarté)
        freq_max_idx = len(f) // 2
        ax.pcolormesh(t, f[:freq_max_idx], S_db[:freq_max_idx],
                      shading='gouraud', cmap='inferno')

        ax.set_title(f"Micro {mic_id}", fontsize=11, fontweight='bold')

        # Dessiner les bounding boxes
        detections = all_detections.get(mic_id, [])
        for det in detections:
            local_cluster = det.get('cluster', -1)
            bird_id = global_map.get((mic_id, local_cluster), -1)

            t_start, t_end = det['t_start'], det['t_end']
            f_min, f_max = det['f_min'], det['f_max']

            # Couleur selon l'oiseau
            if bird_id >= 0:
                color = bird_colors[bird_id % 10]
            else:
                color = (0.5, 0.5, 0.5, 0.5)  # Gris pour le bruit

            rect = Rectangle(
                (t_start, f_min),
                t_end - t_start,
                f_max - f_min,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                alpha=0.9
            )
            ax.add_patch(rect)

            # Label ID oiseau
            if bird_id >= 0:
                ax.text(t_start + 0.02, f_max + 200, f"{bird_id}",
                        color=color, fontsize=8, fontweight='bold',
                        verticalalignment='bottom')

        ax.set_xlabel("Temps (s)", fontsize=9)
        ax.set_ylabel("Fréquence (Hz)", fontsize=9)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, f[freq_max_idx])
        ax.grid(True, alpha=0.2, linestyle='--', color='white')

    # Masquer les axes inutilisés
    for ax in axes[n_mics:]:
        ax.axis('off')

    # Légende globale
    legend_elements = []
    unique_birds = set()
    for (mic_id, local_cluster), bird_id in global_map.items():
        if bird_id >= 0:
            unique_birds.add(bird_id)

    for bird_id in sorted(unique_birds):
        color = bird_colors[bird_id % 10]
        legend_elements.append(
            Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                      label=f'Oiseau {bird_id}')
        )

    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.suptitle("Spectrogrammes STFT avec détections YOLO",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
