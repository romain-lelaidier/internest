"""
Visualisation de la confiance par sous-ensembles de micros.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from config import Config


def visualize_confidence(filtered_data: dict, output_path: str):
    """
    Visualise les trajectoires filtrées avec les nuages de points des sous-ensembles.

    Args:
        filtered_data: {bird_id: {'points': [...], 'rejected': [...]}}
        output_path: Chemin de sauvegarde
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Micros
    ax.scatter(
        Config.MICROS[:, 0], Config.MICROS[:, 1], Config.MICROS[:, 2],
        c='black', marker='^', s=100, label='Micros', zorder=10
    )

    # Palette de couleurs pour les oiseaux
    bird_colors = cm.tab10(np.linspace(0, 1, 10))

    first_rejected_plotted = False

    for bird_id, data in filtered_data.items():
        color = bird_colors[bird_id % 10]
        conf_points = data['points']
        rejected_points = data['rejected']

        if len(conf_points) == 0:
            continue

        # Trajectoire filtrée (ligne épaisse)
        main_positions = np.array([cp['main_pos'] for cp in conf_points])
        times = np.array([cp['t'] for cp in conf_points])

        # Trier par temps
        sort_idx = np.argsort(times)
        main_positions = main_positions[sort_idx]

        ax.plot(
            main_positions[:, 0], main_positions[:, 1], main_positions[:, 2],
            color=color, linewidth=3, alpha=0.9,
            label=f'Oiseau {bird_id} (filtré)'
        )

        # Points filtrés (gros points)
        ax.scatter(
            main_positions[:, 0], main_positions[:, 1], main_positions[:, 2],
            color=color, s=50, alpha=0.8, edgecolors='black', linewidths=0.5
        )

        # Nuages de points des sous-ensembles
        for cp in conf_points:
            subset_pos = cp['subset_positions']
            if len(subset_pos) > 0:
                subset_array = np.array(subset_pos)
                ax.scatter(
                    subset_array[:, 0], subset_array[:, 1], subset_array[:, 2],
                    color=color, s=8, alpha=0.25, marker='.'
                )

        # Points rejetés (croix rouges)
        if len(rejected_points) > 0:
            rejected_positions = np.array([cp['main_pos'] for cp in rejected_points])
            label = 'Points rejetés' if not first_rejected_plotted else ''
            ax.scatter(
                rejected_positions[:, 0], rejected_positions[:, 1], rejected_positions[:, 2],
                color='red', s=60, alpha=0.7, marker='x', linewidths=2,
                label=label
            )
            first_rejected_plotted = True

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 50)
    ax.set_title('Validation par sous-ensembles de micros\n'
                 '(points principaux + nuages des 5-uplets)',
                 fontsize=13)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_confidence_2d(filtered_data: dict, output_path: str):
    """
    Vue de dessus (X-Y) avec ellipses d'incertitude et points rejetés.

    Args:
        filtered_data: {bird_id: {'points': [...], 'rejected': [...]}}
        output_path: Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Micros
    ax.scatter(
        Config.MICROS[:, 0], Config.MICROS[:, 1],
        c='black', marker='^', s=100, label='Micros', zorder=10
    )

    bird_colors = cm.tab10(np.linspace(0, 1, 10))

    for bird_id, data in filtered_data.items():
        color = bird_colors[bird_id % 10]
        conf_points = data['points']
        rejected_points = data['rejected']

        if len(conf_points) == 0:
            continue

        main_positions = np.array([cp['main_pos'] for cp in conf_points])
        times = np.array([cp['t'] for cp in conf_points])
        stds = np.array([cp['std_xyz'] for cp in conf_points])

        sort_idx = np.argsort(times)
        main_positions = main_positions[sort_idx]
        stds = stds[sort_idx]

        # Trajectoire filtrée
        ax.plot(
            main_positions[:, 0], main_positions[:, 1],
            color=color, linewidth=2, alpha=0.8,
            label=f'Oiseau {bird_id}'
        )

        # Points avec ellipses d'incertitude
        for pos, std in zip(main_positions, stds):
            ellipse = mpatches.Ellipse(
                (pos[0], pos[1]),
                width=2 * max(std[0], 1),  # 1 sigma
                height=2 * max(std[1], 1),
                fill=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.15,
                linewidth=0.5
            )
            ax.add_patch(ellipse)

        ax.scatter(
            main_positions[:, 0], main_positions[:, 1],
            color=color, s=30, alpha=0.9, edgecolors='black', linewidths=0.5
        )

        # Points rejetés (croix rouges)
        if len(rejected_points) > 0:
            rejected_positions = np.array([cp['main_pos'] for cp in rejected_points])
            ax.scatter(
                rejected_positions[:, 0], rejected_positions[:, 1],
                color='red', s=50, alpha=0.6, marker='x', linewidths=1.5
            )

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Vue de dessus avec ellipses d'incertitude (1σ)", fontsize=13)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
