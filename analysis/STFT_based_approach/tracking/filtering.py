"""
Filtrage des trajectoires par confiance et vitesse.
"""

import numpy as np

from config import Config


def filter_confidence_trajectories(confidence_data: dict) -> dict:
    """
    Filtre les trajectoires basé sur la confiance des sous-ensembles
    et la vitesse maximale physiquement possible.

    Args:
        confidence_data: {bird_id: [{t, main_pos, subset_positions, std_xyz, confidence}, ...]}

    Returns:
        dict: {bird_id: {'points': [...], 'rejected': [...]}}
    """
    filtered_data = {}

    for bird_id, conf_points in confidence_data.items():
        if len(conf_points) == 0:
            continue

        # Trier par temps
        sorted_points = sorted(conf_points, key=lambda x: x['t'])

        # Premier passage : filtre par confiance et std
        candidates = []
        for cp in sorted_points:
            mean_std = np.mean(cp['std_xyz'])
            if cp['confidence'] >= Config.CONFIDENCE_MIN and mean_std <= Config.STD_MAX:
                candidates.append(cp)

        if len(candidates) == 0:
            filtered_data[bird_id] = {'points': [], 'rejected': sorted_points}
            continue

        # Deuxième passage : filtre par vitesse
        filtered = [candidates[0]]  # Premier point accepté
        rejected = []

        for i in range(1, len(candidates)):
            prev = filtered[-1]
            curr = candidates[i]

            dt = curr['t'] - prev['t']
            if dt < 1e-6:
                continue

            dist = np.linalg.norm(
                np.array(curr['main_pos']) - np.array(prev['main_pos'])
            )
            velocity = dist / dt

            if velocity <= Config.V_MAX:
                filtered.append(curr)
            else:
                # Point rejeté car vitesse trop élevée
                curr['reject_reason'] = f'v={velocity:.1f}m/s > {Config.V_MAX}m/s'
                rejected.append(curr)

        # Points rejetés au premier passage (confiance/std)
        rejected_conf = [cp for cp in sorted_points
                        if cp['confidence'] < Config.CONFIDENCE_MIN
                        or np.mean(cp['std_xyz']) > Config.STD_MAX]
        for cp in rejected_conf:
            cp['reject_reason'] = f"conf={cp['confidence']:.2f}, std={np.mean(cp['std_xyz']):.1f}m"

        filtered_data[bird_id] = {
            'points': filtered,
            'rejected': rejected + rejected_conf
        }

    return filtered_data
