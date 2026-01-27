#!/usr/bin/env python3
"""
Point d'entrée pour la reconstruction de trajectoires d'oiseaux.

Usage:
    python main.py --input ./simu_output_multiple
    python main.py -i ./simu_output_overlap -v
"""

import argparse
import numpy as np

from pipeline import TrajectoryReconstructor


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction de trajectoires d'oiseaux par YOLO + TDOA"
    )
    parser.add_argument(
        "--input", "-i",
        default="./simu_output_multiple",
        help="Dossier contenant les fichiers mix_mic_*.wav"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Dossier de sortie (défaut: input/reconstruction)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose avec debug TDOA"
    )

    args = parser.parse_args()

    # Exécution du pipeline
    reconstructor = TrajectoryReconstructor(
        args.input,
        args.output,
        verbose=args.verbose
    )
    reconstructor.run()

    # Afficher debug si verbose
    if args.verbose and reconstructor.debug_data:
        print("\n--- DEBUG TDOA ---")
        for d in reconstructor.debug_data[:10]:
            print(f"Bird {d['bird_id']} t={d['t']:.2f}s: "
                  f"pos=({d['pos'][0]:.1f}, {d['pos'][1]:.1f}, {d['pos'][2]:.1f}) "
                  f"cost={d['cost']:.3f}")
            delays_str = [f"{x*1000:.2f}ms" if not np.isnan(x) else "nan"
                         for x in d['delays']]
            print(f"  Delays: {delays_str}")


if __name__ == "__main__":
    main()
