"""
Test de detect_frequency — gammes réalistes [1000,1100], [1100,1200], etc.
Buffers de 20s, précision attendue < 100 µs.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')
from server.buzzer_analysis.analyzer_buzz import detect_frequency

SAMPLE_RATE = 48000
BUFFER_DURATION = 20.0  # 20 secondes

def make_signal(freq_hz, onset_s, duration_s, amplitude=0.5):
    n_total = int(BUFFER_DURATION * SAMPLE_RATE)
    n_onset = int(onset_s * SAMPLE_RATE)
    n_tone = int(duration_s * SAMPLE_RATE)
    signal = np.zeros(n_total, dtype=np.float32)
    t = np.arange(n_tone) / SAMPLE_RATE
    signal[n_onset:n_onset + n_tone] = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return (signal * 32767).astype(np.int16)


def test(name, buffer, f_min, f_max, expected_freq, expected_onset_s):
    t0 = time.perf_counter()
    result = detect_frequency(buffer, f_min, f_max, sample_rate=SAMPLE_RATE)
    dt = time.perf_counter() - t0

    if result is None:
        print(f"  FAIL {name}: rien detecte  ({dt*1000:.1f} ms)")
        return

    freq, idx = result
    onset_us = idx / SAMPLE_RATE * 1e6
    expected_us = expected_onset_s * 1e6
    error_us = abs(onset_us - expected_us)

    freq_ok = abs(freq - expected_freq) < 10
    time_ok = error_us < 100
    status = "OK" if freq_ok and time_ok else "FAIL"
    print(f"  {status} {name}")
    print(f"       freq: {freq:.1f} Hz (attendu {expected_freq} Hz)")
    print(f"       onset: {onset_us:.1f} µs (attendu {expected_us:.1f} µs, erreur {error_us:.1f} µs)")
    print(f"       temps calcul: {dt*1000:.1f} ms")


print("=== Test 1 : 1050 Hz dans [1000, 1100] onset=5.0s ===")
buf = make_signal(1050, onset_s=5.0, duration_s=2.0)
test("1050Hz", buf, 1000, 1100, 1050, 5.0)

print("\n=== Test 2 : 1150 Hz dans [1100, 1200] onset=3.0s ===")
buf = make_signal(1150, onset_s=3.0, duration_s=2.0)
test("1150Hz", buf, 1100, 1200, 1150, 3.0)

print("\n=== Test 3 : 1500 Hz dans [1500, 1600] onset=10.0s ===")
buf = make_signal(1500, onset_s=10.0, duration_s=2.0)
test("1500Hz", buf, 1500, 1600, 1500, 10.0)

print("\n=== Test 4 : 1350 Hz faible amplitude onset=8.0s ===")
buf = make_signal(1350, onset_s=8.0, duration_s=2.0, amplitude=0.05)
test("1350Hz faible", buf, 1300, 1400, 1350, 8.0)

print("\n=== Test 5 : hors gamme (1050 Hz, cherche [1300,1400]) ===")
buf = make_signal(1050, onset_s=2.0, duration_s=2.0)
t0 = time.perf_counter()
result = detect_frequency(buf, 1300, 1400, sample_rate=SAMPLE_RATE)
dt = time.perf_counter() - t0
status = "OK" if result is None else "FAIL"
print(f"  {status} hors gamme: result={result}  ({dt*1000:.1f} ms)")

print("\n=== Test 6 : silence ===")
buf = np.zeros(int(BUFFER_DURATION * SAMPLE_RATE), dtype=np.int16)
t0 = time.perf_counter()
result = detect_frequency(buf, 1000, 1100, sample_rate=SAMPLE_RATE)
dt = time.perf_counter() - t0
status = "OK" if result is None else "FAIL"
print(f"  {status} silence: result={result}  ({dt*1000:.1f} ms)")

print("\n=== Test 7 : 1200 Hz onset=12.3456s (précision) ===")
buf = make_signal(1200, onset_s=12.3456, duration_s=2.0)
test("1200Hz precis", buf, 1100, 1200, 1200, 12.3456)
