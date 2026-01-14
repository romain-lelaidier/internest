#!/usr/bin/env python3
"""
Test script for the Bird Localization System (no visualization).

This validates the complete pipeline without matplotlib dependency.
"""
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SAMPLE_RATE
from src.microphone import MicrophoneArray
from src.bird import Bird, CallType
from src.propagation import SimulationEnvironment
from src.detection import CWTDetector
from src.tdoa import TDOAEstimator
from src.localization import Multilaterator


def test_single_bird():
    """
    Test: Localize a single stationary bird.
    """
    print("=" * 60)
    print("Bird Localization System - Test")
    print("=" * 60)

    # 1. Create microphone array (6 mics, optimal placement)
    print("\n[1] Creating microphone array...")
    np.random.seed(42)  # For reproducibility
    mic_array = MicrophoneArray.create_optimal_array(
        num_mics=6,
        radius=50.0,
        height_variation=15.0
    )

    print(f"    Array: {mic_array}")
    print(f"    Coplanar check: {mic_array.is_coplanar()}")

    for mic in mic_array.microphones:
        print(f"    Mic {mic.id}: [{mic.position[0]:.1f}, {mic.position[1]:.1f}, {mic.position[2]:.1f}]")

    # 2. Create bird
    print("\n[2] Creating bird...")
    bird = Bird(
        id=0,
        position=np.array([20.0, 30.0, 12.0]),
        call_type=CallType.CHIRP,
        frequency_range=(2500, 4000),
        amplitude=1.0
    )
    print(f"    Bird position: {bird.position}")
    print(f"    Call type: {bird.call_type.value}")

    # 3. Simulation
    print("\n[3] Simulating bird call...")
    env = SimulationEnvironment(mic_array, sample_rate=SAMPLE_RATE, noise_level=0.05)

    received = env.simulate_call(bird, call_duration=0.15, total_duration=1.0)

    print("    Signal received at microphones:")
    for mic_id, sig in received.items():
        print(f"      Mic {mic_id}: arrival={sig.arrival_time*1000:.2f}ms, SNR={sig.snr:.1f}dB")

    # 4. CWT Detection
    print("\n[4] Detecting bird call (CWT)...")
    detector = CWTDetector(sample_rate=SAMPLE_RATE, threshold=0.2)
    events = detector.detect_events(received[0].signal)

    print(f"    Detected {len(events)} event(s)")
    if events:
        event = events[0]
        print(f"    Event: {event.start_time*1000:.1f}ms - {event.end_time*1000:.1f}ms")
        print(f"    Peak frequency: {event.peak_frequency:.0f} Hz")

    # 5. TDOA Estimation using full signal with GCC-PHAT
    # Note: For large mic arrays with 200ms+ arrival time differences,
    # using full signal works better than windowed estimation
    print("\n[5] Estimating TDOA (GCC-PHAT on full signal)...")
    signals = {mic_id: sig.signal for mic_id, sig in received.items()}

    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)
    tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals, method='gcc_phat')

    # Theoretical TDOA for comparison
    theoretical_tdoa = env.get_theoretical_tdoa(bird)

    print(f"    TDOA estimates (15 pairs):")
    total_error = 0
    for est in tdoa_estimates:
        theoretical = theoretical_tdoa.get((est.mic_i, est.mic_j), 0)
        error_us = abs(est.tdoa_seconds - theoretical) * 1e6
        total_error += error_us
        conf_marker = "✓" if est.confidence > 0.5 else "✗"
        print(f"      ({est.mic_i},{est.mic_j}): {est.tdoa_seconds*1000:+.3f}ms "
              f"[theory: {theoretical*1000:+.3f}ms, err: {error_us:.0f}µs, conf: {est.confidence:.2f}] {conf_marker}")

    avg_tdoa_error = total_error / len(tdoa_estimates)
    print(f"\n    Average TDOA error: {avg_tdoa_error:.1f} µs")

    # 6. Localization
    print("\n[6] Localizing bird (Multilateration)...")
    localizer = Multilaterator(mic_array)

    # Filter estimates by physical validity and confidence
    valid_estimates = tdoa_estimator.filter_physically_valid(tdoa_estimates)
    good_estimates = [e for e in valid_estimates if e.confidence > 0.5]
    if len(good_estimates) < 4:
        good_estimates = sorted(valid_estimates, key=lambda e: e.confidence, reverse=True)[:10]
    print(f"    Using {len(good_estimates)}/{len(tdoa_estimates)} valid+high-confidence estimates")

    result = localizer.localize(good_estimates)

    true_pos = bird.position
    est_pos = result.position
    error = np.linalg.norm(est_pos - true_pos)

    print(f"\n    ┌─────────────────────────────────────────┐")
    print(f"    │ RESULTS                                 │")
    print(f"    ├─────────────────────────────────────────┤")
    print(f"    │ True position:      ({true_pos[0]:6.1f}, {true_pos[1]:6.1f}, {true_pos[2]:6.1f}) │")
    print(f"    │ Estimated position: ({est_pos[0]:6.1f}, {est_pos[1]:6.1f}, {est_pos[2]:6.1f}) │")
    print(f"    │ Localization error: {error:6.2f} m              │")
    print(f"    │ Confidence:         {result.confidence:6.2f}               │")
    print(f"    └─────────────────────────────────────────┘")

    return error


def test_multiple_positions():
    """
    Test localization at multiple positions.
    """
    print("\n" + "=" * 60)
    print("Testing Multiple Positions")
    print("=" * 60)

    np.random.seed(42)
    mic_array = MicrophoneArray.create_optimal_array(num_mics=6, radius=50.0)
    env = SimulationEnvironment(mic_array, noise_level=0.05)
    localizer = Multilaterator(mic_array)
    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)

    # Test positions
    test_positions = [
        np.array([0.0, 0.0, 10.0]),      # Center
        np.array([30.0, 20.0, 15.0]),    # Inside array
        np.array([-20.0, 40.0, 8.0]),    # Inside array
        np.array([60.0, 10.0, 20.0]),    # Near edge
        np.array([80.0, 50.0, 12.0]),    # Outside array
    ]

    print("\n    Position                  | Estimated               | Error")
    print("    " + "-" * 70)

    errors = []
    for i, pos in enumerate(test_positions):
        bird = Bird(id=i, position=pos, call_type=CallType.CHIRP)

        received = env.simulate_call(bird, 0.15, 1.0)
        signals = {mic_id: sig.signal for mic_id, sig in received.items()}

        tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals)

        # Filter by physical validity and confidence
        valid_estimates = tdoa_estimator.filter_physically_valid(tdoa_estimates)
        good_estimates = [e for e in valid_estimates if e.confidence > 0.5]
        if len(good_estimates) < 4:
            good_estimates = sorted(valid_estimates, key=lambda e: e.confidence, reverse=True)[:10]

        result = localizer.localize(good_estimates)

        error = np.linalg.norm(result.position - pos)
        errors.append(error)

        print(f"    ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:5.1f}) | "
              f"({result.position[0]:6.1f}, {result.position[1]:6.1f}, {result.position[2]:5.1f}) | "
              f"{error:5.2f} m")

    print("    " + "-" * 70)
    print(f"    Average error: {np.mean(errors):.2f} m")
    print(f"    Max error:     {np.max(errors):.2f} m")

    return errors


def test_noise_levels():
    """
    Test accuracy at different noise levels.
    """
    print("\n" + "=" * 60)
    print("Testing Noise Robustness")
    print("=" * 60)

    np.random.seed(42)
    mic_array = MicrophoneArray.create_optimal_array(num_mics=6, radius=50.0)
    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)
    localizer = Multilaterator(mic_array)

    bird = Bird(
        id=0,
        position=np.array([25.0, -15.0, 10.0]),
        call_type=CallType.CHIRP
    )

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]

    print(f"\n    Bird position: {bird.position}")
    print("\n    Noise Level | Avg SNR (dB) | Error (m)")
    print("    " + "-" * 45)

    for noise in noise_levels:
        env = SimulationEnvironment(mic_array, noise_level=noise)
        received = env.simulate_call(bird, 0.15, 1.0)
        signals = {mic_id: sig.signal for mic_id, sig in received.items()}

        tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals)

        # Filter by physical validity and confidence
        valid_estimates = tdoa_estimator.filter_physically_valid(tdoa_estimates)
        good_estimates = [e for e in valid_estimates if e.confidence > 0.5]
        if len(good_estimates) < 4:
            good_estimates = sorted(valid_estimates, key=lambda e: e.confidence, reverse=True)[:10]

        result = localizer.localize(good_estimates)

        error = np.linalg.norm(result.position - bird.position)
        avg_snr = np.mean([sig.snr for sig in received.values()])

        print(f"    {noise:11.2f} | {avg_snr:12.1f} | {error:9.2f}")


def test_call_types():
    """
    Test detection and localization with different call types.
    """
    print("\n" + "=" * 60)
    print("Testing Different Call Types")
    print("=" * 60)

    np.random.seed(42)
    mic_array = MicrophoneArray.create_optimal_array(num_mics=6, radius=50.0)
    env = SimulationEnvironment(mic_array, noise_level=0.05)
    localizer = Multilaterator(mic_array)
    detector = CWTDetector(sample_rate=SAMPLE_RATE, threshold=0.2)
    tdoa_estimator = TDOAEstimator(mic_array, SAMPLE_RATE)

    position = np.array([15.0, 25.0, 12.0])
    call_types = [CallType.CHIRP, CallType.SONG, CallType.ALARM, CallType.TRILL]

    print(f"\n    Bird position: {position}")
    print("\n    Call Type | Error (m) | Detection Confidence")
    print("    " + "-" * 50)

    for call_type in call_types:
        bird = Bird(
            id=0,
            position=position,
            call_type=call_type,
            frequency_range=(2500, 4000)
        )

        received = env.simulate_call(bird, 0.15, 1.0)
        signals = {mic_id: sig.signal for mic_id, sig in received.items()}

        # Detect
        events = detector.detect_events(signals[0])
        conf = events[0].confidence if events else 0.0

        # Localize
        tdoa_estimates = tdoa_estimator.estimate_all_pairs(signals)

        # Filter by physical validity and confidence
        valid_estimates = tdoa_estimator.filter_physically_valid(tdoa_estimates)
        good_estimates = [e for e in valid_estimates if e.confidence > 0.5]
        if len(good_estimates) < 4:
            good_estimates = sorted(valid_estimates, key=lambda e: e.confidence, reverse=True)[:10]

        result = localizer.localize(good_estimates)

        error = np.linalg.norm(result.position - position)

        print(f"    {call_type.value:9} | {error:9.2f} | {conf:.2f}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("    BIRD LOCALIZATION SYSTEM")
    print("    Test Suite")
    print("=" * 60)

    # Run all tests
    single_error = test_single_bird()
    position_errors = test_multiple_positions()
    test_noise_levels()
    test_call_types()

    print("\n" + "=" * 60)
    print("    TEST SUMMARY")
    print("=" * 60)
    print(f"    Single bird localization: {single_error:.2f} m error")
    print(f"    Multiple positions avg:   {np.mean(position_errors):.2f} m error")
    print("=" * 60)
    print("    All tests completed successfully!")
    print("=" * 60)
