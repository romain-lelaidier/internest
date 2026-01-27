"""
Module de localisation : TDOA, triangulation et validation par sous-ensembles.
"""

from .tdoa import gcc_phat_band, compute_delay_envelope, get_envelope_band
from .triangulation import tdoa_residuals, find_best_initial_guess
from .subsets import triangulate_with_subset, compute_position_with_subsets

__all__ = [
    'gcc_phat_band',
    'compute_delay_envelope',
    'get_envelope_band',
    'tdoa_residuals',
    'find_best_initial_guess',
    'triangulate_with_subset',
    'compute_position_with_subsets'
]
