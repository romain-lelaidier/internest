"""
Visualisation 3D des trajectoires.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from config import Config


def visualize_trajectories_3d(trajectories: dict, output_path: str,
                              ground_truth_path: str = None):
    """
    Génère la visualisation 3D des trajectoires.

    Args:
        trajectories: {bird_id: array (N x 4) avec [x, y, z, t]}
        output_path: Chemin de sauvegarde
        ground_truth_path: Chemin optionnel vers le fichier de vérité terrain
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Micros
    ax.scatter(
        Config.MICROS[:, 0], Config.MICROS[:, 1], Config.MICROS[:, 2],
        c='black', marker='^', s=100, label='Micros'
    )

    # Ground truth si disponible
    if ground_truth_path:
        try:
            truth = np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)
            true_ids = np.unique(truth[:, 1])
            for tid in true_ids:
                mask = (truth[:, 1] == tid)
                ax.plot(
                    truth[mask, 2], truth[mask, 3], truth[mask, 4],
                    linestyle='--', color='gray', linewidth=1.5, alpha=0.6,
                    label=f'Vérité Oiseau {int(tid)}'
                )
        except Exception as e:
            print(f"  Impossible de charger la vérité terrain: {e}")

    # Trajectoires estimées
    colors = cm.tab10(np.linspace(0, 1, max(1, len(trajectories))))

    for i, (bird_id, data) in enumerate(trajectories.items()):
        if len(data) < 2:
            continue

        color = colors[i]
        ax.plot(
            data[:, 0], data[:, 1], data[:, 2],
            color=color, linewidth=3, alpha=0.9,
            label=f'Estimé Oiseau {bird_id}'
        )
        ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            color=color, s=30, alpha=0.6
        )

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 50)
    ax.set_title('Reconstruction 3D des trajectoires (YOLO + TDOA)', fontsize=14)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
