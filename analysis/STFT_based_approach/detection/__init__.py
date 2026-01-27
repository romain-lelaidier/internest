"""
Module de détection : spectrogrammes, YOLO et clustering.
"""

from .spectrogram import generate_spectrogram
from .yolo_detector import detect_on_spectrogram
from .clustering import cluster_detections, compute_global_cluster_map

__all__ = [
    'generate_spectrogram',
    'detect_on_spectrogram',
    'cluster_detections',
    'compute_global_cluster_map'
]
