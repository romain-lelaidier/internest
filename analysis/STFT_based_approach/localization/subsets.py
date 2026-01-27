"""
Triangulation par sous-ensembles de micros pour validation.
"""

import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

from config import Config
from .triangulation import tdoa_residuals


def triangulate_with_subset(delays: np.ndarray, ref_mic: int,
                            mic_subset: tuple, initial_guess: np.ndarray) -> tuple:
    """
    Triangule la position en utilisant uniquement un sous-ensemble de micros.

    Args:
        delays: Tableau des délais (8 valeurs, certaines peuvent être nan)
        ref_mic: Index du micro de référence
        mic_subset: Tuple des indices de micros à utiliser
        initial_guess: Position initiale pour l'optimisation

    Returns:
        (position, cost) ou (None, inf) si échec
    """
    # Créer un tableau de délais masqué (nan pour les micros hors subset)
    delays_subset = np.full_like(delays, np.nan)
    for mic_idx in mic_subset:
        delays_subset[mic_idx] = delays[mic_idx]

    # Vérifier qu'on a assez de micros valides
    n_valid = np.sum(~np.isnan(delays_subset))
    if n_valid < Config.MIN_MICS:
        return None, float('inf')

    # Le micro de référence doit être dans le subset
    if ref_mic not in mic_subset:
        return None, float('inf')

    try:
        res = least_squares(
            tdoa_residuals, initial_guess,
            args=(Config.MICROS, delays_subset, ref_mic),
            bounds=([0, 0, 0], [100, 100, 50]),
            ftol=1e-4,
            max_nfev=100
        )
        return res.x, res.cost
    except Exception:
        return None, float('inf')


def compute_position_with_subsets(delays: np.ndarray, ref_mic: int,
                                  initial_guess: np.ndarray,
                                  subset_size: int = 5,
                                  max_subsets: int = 30) -> dict:
    """
    Calcule la position avec différents sous-ensembles de micros pour évaluer la confiance.

    Args:
        delays: Tableau des délais TDOA
        ref_mic: Micro de référence
        initial_guess: Position initiale
        subset_size: Taille des sous-ensembles (5 ou 6 recommandé)
        max_subsets: Nombre max de sous-ensembles à tester

    Returns:
        dict avec:
            - main_pos: position avec tous les micros
            - subset_positions: liste des positions par sous-ensemble
            - std_xyz: écart-type sur chaque axe
            - confidence: score de confiance (0-1)
    """
    # Position principale (tous les micros)
    main_pos, main_cost = triangulate_with_subset(
        delays, ref_mic, tuple(range(8)), initial_guess
    )

    if main_pos is None:
        return None

    # Trouver les micros avec des délais valides
    valid_mics = [i for i in range(8) if not np.isnan(delays[i])]

    if len(valid_mics) < subset_size:
        return {
            'main_pos': main_pos,
            'main_cost': main_cost,
            'subset_positions': [],
            'std_xyz': np.array([0, 0, 0]),
            'confidence': 0.5
        }

    # Générer les sous-ensembles (ref_mic doit toujours être inclus)
    other_mics = [m for m in valid_mics if m != ref_mic]
    all_subsets = list(combinations(other_mics, subset_size - 1))

    # Limiter le nombre de sous-ensembles si nécessaire
    if len(all_subsets) > max_subsets:
        indices = np.random.choice(len(all_subsets), max_subsets, replace=False)
        all_subsets = [all_subsets[i] for i in indices]

    # Calculer les positions pour chaque sous-ensemble
    subset_positions = []
    for subset in all_subsets:
        full_subset = (ref_mic,) + subset
        pos, cost = triangulate_with_subset(delays, ref_mic, full_subset, main_pos)
        if pos is not None and cost < Config.COST_THRESHOLD * 2:
            subset_positions.append(pos)

    # Calculer les statistiques
    if len(subset_positions) >= 3:
        positions_array = np.array(subset_positions)
        std_xyz = np.std(positions_array, axis=0)
        # Confiance = inverse de l'écart-type moyen (normalisé)
        mean_std = np.mean(std_xyz)
        confidence = 1.0 / (1.0 + mean_std / 10.0)  # Normalisation douce
    else:
        std_xyz = np.array([float('inf'), float('inf'), float('inf')])
        confidence = 0.0

    return {
        'main_pos': main_pos,
        'main_cost': main_cost,
        'subset_positions': subset_positions,
        'std_xyz': std_xyz,
        'confidence': confidence
    }
