"""
Triangulation 3D par TDOA.
"""

import numpy as np

from config import Config


def tdoa_residuals(pos: np.ndarray, mic_positions: np.ndarray,
                   delays: np.ndarray, ref_mic_idx: int) -> np.ndarray:
    """
    Fonction de coût pour la triangulation TDOA.

    Args:
        pos: Position candidate (x, y, z)
        mic_positions: Positions des microphones (N x 3)
        delays: Délais mesurés par rapport au micro de référence
        ref_mic_idx: Index du micro de référence

    Returns:
        Résidus (différence entre délais théoriques et mesurés)
    """
    res = []
    d_ref = np.linalg.norm(mic_positions[ref_mic_idx] - pos)

    for i in range(len(mic_positions)):
        if i == ref_mic_idx or np.isnan(delays[i]):
            continue
        d_i = np.linalg.norm(mic_positions[i] - pos)
        d_theo = (d_i - d_ref) / Config.C
        res.append(d_theo - delays[i])

    return np.array(res)


def find_best_initial_guess(mic_positions: np.ndarray, delays: np.ndarray,
                            ref_mic_idx: int) -> np.ndarray:
    """
    Trouve un bon point de départ pour la triangulation via recherche en grille.

    Args:
        mic_positions: Positions des microphones
        delays: Délais TDOA mesurés
        ref_mic_idx: Index du micro de référence

    Returns:
        Position initiale optimale pour l'optimisation
    """
    best_pos = np.array([50., 50., 20.])
    best_cost = float('inf')

    # Grille de recherche grossière
    for x in np.linspace(10, 90, 5):
        for y in np.linspace(10, 90, 5):
            for z in np.linspace(5, 35, 3):
                pos = np.array([x, y, z])
                res = tdoa_residuals(pos, mic_positions, delays, ref_mic_idx)
                cost = np.sum(res**2)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

    return best_pos
