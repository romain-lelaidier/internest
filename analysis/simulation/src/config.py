"""
Configuration for the bird localization system.
"""
import numpy as np

# Audio parameters
SAMPLE_RATE = 44100  # Hz
SPEED_OF_SOUND = 343.0  # m/s at 20°C

# Synchronization precision
SYNC_PRECISION_US = 50  # microseconds
SYNC_PRECISION_S = SYNC_PRECISION_US * 1e-6  # seconds

# Bird vocalization frequency range (Hz)
BIRD_FREQ_MIN = 1000  # 1 kHz
BIRD_FREQ_MAX = 8000  # 8 kHz

# CWT parameters
CWT_WAVELET = 'morl'  # Morlet wavelet - good for bird songs
CWT_SCALES_MIN = 5
CWT_SCALES_MAX = 50

# Detection parameters
DETECTION_THRESHOLD = 0.3  # Relative threshold for event detection
MIN_EVENT_DURATION_MS = 20  # Minimum duration to consider as bird call

# Simulation parameters
AMBIENT_NOISE_LEVEL = 0.05  # Relative amplitude of background noise
