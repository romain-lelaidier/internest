#!/usr/bin/env python3
"""
Demo script for the Bird Localization System.

This script demonstrates the complete pipeline:
1. Create microphone array
2. Place a bird in 3D space
3. Simulate bird call propagation
4. Detect the call using CWT
5. Estimate TDOA between microphone pairs
6. Localize the bird using multilateration
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SAMPLE_RATE
from src.microphone import MicrophoneArray
from src.bird import Bird, CallType
from src.propagation import SimulationEnvironment
from src.detection import CWTDetector, MultiChannelDetector
from src.tdoa import TDOAEstimator
from src.localization import Multilaterator
from src.visualization import (
    plot_scenario, plot_signals, plot_scalogram, plot_cross_correlation
)


def run_single_bird_demo():
    """
    Demo: Localize a single stationary bird.
    """
    print("=" * 60)
    print("Bird Localization System Demo")
    print("=" * 60)

    # 1. Create microphone array (6 mics, optimal placement)
    print("\n[1] Creating microphone array...")
    mic_array = MicrophoneArray.create_optimal_array(
        num_mics=6,
        radius=50.0,          # 50m radius
        height_variation=15.0  # Heights vary from 2m to 17m
    )

    print(f"    Array: {mic_array}")
    print(f"    Coplanar check: {mic_array.is_coplanar()}")
    print(f"    Coverage: {mic_array.coverage_volume()}")

    for mic in mic_array.microphones:
        print(f"    Mic {mic.id}: {mic.position}")

    # 2. Create a bird
    print("\n[2] Creating bird...")
    bird = Bird(
        id=0,
        position=np.array([20.0, 30.0, 12.0]),  # Inside the array
        call_type=CallType.CHIRP,
        frequency_range=(2500, 4000),  # Hz
        amplitude=1.0
    )
    print(f"    Bird: {bird}")
    print(f"    Position: {bird.position}")

    # 3. Set up simulation environment
    print("\n[3] Setting up simulation...")
    env = SimulationEnvironment(
        mic_array=mic_array,
        sample_rate=SAMPLE_RATE,
        noise_level=0.05  # Low background noise
    )

    # 4. Simulate bird call
    print("\n[4] Simulating bird call...")
    call_duration = 0.15  # 150ms call
    total_duration = 1.0  # 1 second recording

    received = env.simulate_call(bird, call_duration, total_duration)

    print(f"    Call duration: {call_duration}s")
    print(f"    Recording duration: {total_duration}s")
    for mic_id, sig in received.items():
        print(f"    Mic {mic_id}: arrival={sig.arrival_time:.4f}s, SNR={sig.snr:.1f}dB")

    # 5. Detect bird call using CWT
    print("\n[5] Detecting bird call (CWT)...")
    detector = CWTDetector(sample_rate=SAMPLE_RATE, threshold=0.2)

    # Detect in first microphone as reference
    events = detector.detect_events(received[0].signal)
    print(f"    Detected {len(events)} event(s)")

    if events:
        event = events[0]
        print(f"    Event: {event.start_time:.3f}s - {event.end_time:.3f}s")
        print(f"    Peak frequency: {event.peak_frequency:.0f} Hz")
        print(f"    Confidence: {event.confidence:.2f}")

    # 6. Estimate TDOA
    print("\n[6] Estimating TDOA (GCC-PHAT)...")
    signals = {mic_id: sig.signal for mic_id, sig in received.items()}

    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)
    tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals, method='gcc_phat')

    # Get theoretical TDOA for comparison
    theoretical_tdoa = env.get_theoretical_tdoa(bird)

    print(f"    Estimated {len(tdoa_estimates)} TDOA pairs:")
    for est in tdoa_estimates:
        theoretical = theoretical_tdoa.get((est.mic_i, est.mic_j), 0)
        error_us = (est.tdoa_seconds - theoretical) * 1e6
        print(f"    ({est.mic_i},{est.mic_j}): {est.tdoa_seconds*1000:.3f}ms "
              f"(theory: {theoretical*1000:.3f}ms, error: {error_us:.1f}µs, "
              f"conf: {est.confidence:.2f})")

    # 7. Localize bird
    print("\n[7] Localizing bird (Multilateration)...")
    localizer = Multilaterator(mic_array)

    # Filter low-confidence estimates
    good_estimates = tdoa_estimator.filter_by_confidence(tdoa_estimates, min_confidence=0.3)
    print(f"    Using {len(good_estimates)}/{len(tdoa_estimates)} high-confidence estimates")

    result = localizer.localize(good_estimates)

    # Calculate error
    true_pos = bird.position
    est_pos = result.position
    error = np.linalg.norm(est_pos - true_pos)
    result.position_error = error

    print(f"\n    True position:      ({true_pos[0]:.1f}, {true_pos[1]:.1f}, {true_pos[2]:.1f})")
    print(f"    Estimated position: ({est_pos[0]:.1f}, {est_pos[1]:.1f}, {est_pos[2]:.1f})")
    print(f"    Localization error: {error:.2f} m")
    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    Residual: {result.residual:.6f}")

    # 8. Visualize results
    print("\n[8] Generating visualizations...")

    # Plot 3D scenario
    fig1 = plot_scenario(mic_array, [bird], [result], title="Bird Localization Result")
    fig1.savefig('demo_3d_scenario.png', dpi=150)
    print("    Saved: demo_3d_scenario.png")

    # Plot signals
    fig2 = plot_signals(signals, SAMPLE_RATE, title="Received Signals at Microphones")
    fig2.savefig('demo_signals.png', dpi=150)
    print("    Saved: demo_signals.png")

    # Plot scalogram for first microphone
    scalogram, times, frequencies = detector.compute_scalogram(signals[0])
    fig3 = plot_scalogram(scalogram, times, frequencies, title="CWT Scalogram (Mic 0)")
    fig3.savefig('demo_scalogram.png', dpi=150)
    print("    Saved: demo_scalogram.png")

    plt.close('all')

    print("\n" + "=" * 60)
    print(f"Demo complete! Localization error: {error:.2f} m")
    print("=" * 60)

    return error


def run_moving_bird_demo():
    """
    Demo: Track a moving bird.
    """
    print("\n" + "=" * 60)
    print("Moving Bird Tracking Demo")
    print("=" * 60)

    # Create microphone array
    mic_array = MicrophoneArray.create_optimal_array(num_mics=6, radius=50.0)

    # Create bird with velocity
    bird = Bird(
        id=0,
        position=np.array([0.0, 0.0, 15.0]),
        call_type=CallType.SONG,
        frequency_range=(3000, 5000),
        is_moving=True
    )
    bird.set_velocity(np.array([5.0, 3.0, 0.0]))  # 5 m/s in x, 3 m/s in y

    env = SimulationEnvironment(mic_array)
    localizer = Multilaterator(mic_array)
    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)

    # Track bird over multiple calls
    positions_true = []
    positions_estimated = []
    times = []

    print("\nTracking bird over 5 calls...")
    for i in range(5):
        t = i * 0.5  # Call every 0.5 seconds
        times.append(t)

        # Move bird
        bird.move(0.5)
        positions_true.append(bird.position.copy())

        # Simulate call
        received = env.simulate_call(bird, 0.1, 0.5)
        signals = {mic_id: sig.signal for mic_id, sig in received.items()}

        # Estimate TDOA and localize
        tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals)
        result = localizer.localize(tdoa_estimates)
        positions_estimated.append(result.position.copy())

        error = np.linalg.norm(result.position - bird.position)
        print(f"  t={t:.1f}s: True=({bird.position[0]:.1f}, {bird.position[1]:.1f}, {bird.position[2]:.1f}), "
              f"Est=({result.position[0]:.1f}, {result.position[1]:.1f}, {result.position[2]:.1f}), "
              f"Error={error:.2f}m")

    # Plot trajectory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = mic_array.positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='blue', s=100, marker='^', label='Microphones')

    positions_true = np.array(positions_true)
    positions_estimated = np.array(positions_estimated)

    ax.plot(positions_true[:, 0], positions_true[:, 1], positions_true[:, 2],
            'r-o', linewidth=2, markersize=8, label='True trajectory')
    ax.plot(positions_estimated[:, 0], positions_estimated[:, 1], positions_estimated[:, 2],
            'g-*', linewidth=2, markersize=10, label='Estimated trajectory')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Moving Bird Tracking')
    ax.legend()

    plt.tight_layout()
    plt.savefig('demo_tracking.png', dpi=150)
    print("\nSaved: demo_tracking.png")
    plt.close()


def run_noise_robustness_test():
    """
    Test localization accuracy at different noise levels.
    """
    print("\n" + "=" * 60)
    print("Noise Robustness Test")
    print("=" * 60)

    mic_array = MicrophoneArray.create_optimal_array(num_mics=6, radius=50.0)

    bird = Bird(
        id=0,
        position=np.array([25.0, -15.0, 10.0]),
        call_type=CallType.CHIRP,
        frequency_range=(2000, 3500)
    )

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    errors = []

    print(f"\nBird position: {bird.position}")
    print("\nTesting different noise levels...")

    for noise in noise_levels:
        env = SimulationEnvironment(mic_array, noise_level=noise)
        received = env.simulate_call(bird, 0.15, 1.0)
        signals = {mic_id: sig.signal for mic_id, sig in received.items()}

        tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)
        tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals)

        localizer = Multilaterator(mic_array)
        result = localizer.localize(tdoa_estimates)

        error = np.linalg.norm(result.position - bird.position)
        errors.append(error)

        avg_snr = np.mean([sig.snr for sig in received.values()])
        print(f"  Noise={noise:.2f}, Avg SNR={avg_snr:.1f}dB, Error={error:.2f}m")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(noise_levels, errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Localization Error (m)')
    ax.set_title('Noise Robustness Test')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('demo_noise_test.png', dpi=150)
    print("\nSaved: demo_noise_test.png")
    plt.close()


if __name__ == '__main__':
    # Run all demos
    run_single_bird_demo()
    run_moving_bird_demo()
    run_noise_robustness_test()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
