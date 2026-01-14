"""
Visualization tools for the bird localization system.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict

from .microphone import MicrophoneArray
from .bird import Bird
from .localization import LocalizationResult


def plot_microphone_array(mic_array: MicrophoneArray,
                         ax: Optional[plt.Axes] = None,
                         show_ids: bool = True) -> plt.Axes:
    """
    Plot microphone positions in 3D.

    Args:
        mic_array: Microphone array to plot
        ax: Existing 3D axes (creates new if None)
        show_ids: Whether to show microphone IDs

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    positions = mic_array.positions

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c='blue',
        s=100,
        marker='^',
        label='Microphones'
    )

    if show_ids:
        for mic in mic_array.microphones:
            ax.text(
                mic.position[0],
                mic.position[1],
                mic.position[2] + 2,
                f'M{mic.id}',
                fontsize=8
            )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    return ax


def plot_bird_position(bird: Bird,
                      ax: plt.Axes,
                      color: str = 'red',
                      marker: str = 'o',
                      label: str = None) -> plt.Axes:
    """Plot bird position on existing 3D axes."""
    label = label or f'Bird {bird.id}'
    ax.scatter(
        [bird.position[0]],
        [bird.position[1]],
        [bird.position[2]],
        c=color,
        s=150,
        marker=marker,
        label=label
    )
    return ax


def plot_localization_result(result: LocalizationResult,
                            ax: plt.Axes,
                            color: str = 'green',
                            label: str = 'Estimated') -> plt.Axes:
    """Plot localization result on existing 3D axes."""
    pos = result.position
    ax.scatter(
        [pos[0]],
        [pos[1]],
        [pos[2]],
        c=color,
        s=200,
        marker='*',
        label=label
    )

    # Draw error ellipse if covariance available
    if result.covariance is not None:
        # Simplified: just show position
        pass

    return ax


def plot_scenario(mic_array: MicrophoneArray,
                 birds: List[Bird],
                 results: List[LocalizationResult] = None,
                 title: str = "Bird Localization Scenario") -> plt.Figure:
    """
    Create a complete 3D visualization of the scenario.

    Args:
        mic_array: Microphone array
        birds: List of birds (ground truth)
        results: List of localization results (estimates)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot microphones
    plot_microphone_array(mic_array, ax)

    # Plot birds
    colors = plt.cm.Reds(np.linspace(0.5, 1, len(birds)))
    for i, bird in enumerate(birds):
        plot_bird_position(bird, ax, color=colors[i], label=f'Bird {bird.id} (true)')

    # Plot results if provided
    if results:
        colors = plt.cm.Greens(np.linspace(0.5, 1, len(results)))
        for i, result in enumerate(results):
            plot_localization_result(
                result, ax, color=colors[i],
                label=f'Estimate {i} (err={result.position_error:.1f}m)' if result.position_error else f'Estimate {i}'
            )

            # Draw line from true to estimated
            if i < len(birds):
                true_pos = birds[i].position
                est_pos = result.position
                ax.plot(
                    [true_pos[0], est_pos[0]],
                    [true_pos[1], est_pos[1]],
                    [true_pos[2], est_pos[2]],
                    'k--', alpha=0.5, linewidth=1
                )

    ax.set_title(title)
    ax.legend(loc='upper left')

    # Set equal aspect ratio
    positions = mic_array.positions
    max_range = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ) / 2

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range * 1.5, mid_x + max_range * 1.5)
    ax.set_ylim(mid_y - max_range * 1.5, mid_y + max_range * 1.5)
    ax.set_zlim(0, max_range * 2)

    plt.tight_layout()
    return fig


def plot_scalogram(scalogram: np.ndarray,
                  times: np.ndarray,
                  frequencies: np.ndarray,
                  title: str = "CWT Scalogram") -> plt.Figure:
    """
    Plot CWT scalogram (time-frequency representation).

    Args:
        scalogram: 2D array of CWT coefficients
        times: Time axis
        frequencies: Frequency axis
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot scalogram
    im = ax.pcolormesh(
        times,
        frequencies,
        scalogram,
        shading='gouraud',
        cmap='viridis'
    )

    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_yscale('log')

    plt.colorbar(im, ax=ax, label='Power')
    plt.tight_layout()

    return fig


def plot_signals(signals: Dict[int, np.ndarray],
                sample_rate: int,
                title: str = "Microphone Signals") -> plt.Figure:
    """
    Plot signals from multiple microphones.

    Args:
        signals: Dictionary mapping mic_id to signal
        sample_rate: Audio sample rate
        title: Plot title

    Returns:
        Matplotlib figure
    """
    n_channels = len(signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, (mic_id, signal) in enumerate(sorted(signals.items())):
        t = np.arange(len(signal)) / sample_rate
        axes[i].plot(t, signal, linewidth=0.5)
        axes[i].set_ylabel(f'Mic {mic_id}')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title(title)

    plt.tight_layout()
    return fig


def plot_cross_correlation(correlation: np.ndarray,
                          lags: np.ndarray,
                          sample_rate: int,
                          title: str = "Cross-Correlation") -> plt.Figure:
    """
    Plot cross-correlation result.

    Args:
        correlation: Correlation values
        lags: Lag values in samples
        sample_rate: Audio sample rate
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    lag_times = lags / sample_rate * 1000  # Convert to milliseconds
    ax.plot(lag_times, correlation)

    # Mark peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lag_times[peak_idx]
    ax.axvline(peak_lag, color='r', linestyle='--', label=f'Peak: {peak_lag:.2f} ms')

    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
