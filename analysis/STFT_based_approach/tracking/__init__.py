"""
Module de tracking : gestion des trajectoires et filtrage.
"""

from .track_manager import TrackManager
from .filtering import filter_confidence_trajectories

__all__ = [
    'TrackManager',
    'filter_confidence_trajectories'
]
