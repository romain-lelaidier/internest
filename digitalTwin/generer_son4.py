#!/usr/bin/env python3
"""
generer_son4.py - Simulation de propagation sonore multi-oiseaux avec fichiers audio réels.

Usage:
    python generer_son4.py merle:bird_samples/Merle_noir_chant.mp3 mesange:bird_samples/Mésange_charbonnière_chant.mp3

    Chaque argument est de la forme  nom:chemin_audio
    Les formats .wav et .mp3 sont supportés.

    Options:
        --duree         Durée de la simulation en secondes (défaut: auto = durée du plus long son)
        --trajectoire   Fichier JSON décrivant les trajectoires (optionnel)
        --out           Dossier de sortie (défaut: sim_files)
"""

import argparse
import numpy as np
import soundfile as sf
import os
import json
from scipy.signal import resample_poly
from math import gcd

# ── Configuration par défaut ──────────────────────────────────────────────
FS = 48000
VITESSE_SON = 343.0
BLOC_SIZE = 256

# Positions des micros (cube 10 m)
MICROS = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    [10.0, 10.0, 10.0],
])

# Trajectoires par défaut : lignes droites réparties dans le cube
TRAJECTOIRES_DEFAUT = [
    {"start": [3.0, 3.0, 1.0], "end": [3.0, 3.0, 9.0]},   # montée verticale
    {"start": [1.0, 8.0, 2.0], "end": [9.0, 9.0, 2.0]},   # diagonale basse
    {"start": [9.0, 1.0, 8.0], "end": [1.0, 9.0, 2.0]},   # diagonale descendante
    {"start": [5.0, 1.0, 5.0], "end": [5.0, 9.0, 5.0]},   # traversée Y
    {"start": [1.0, 5.0, 8.0], "end": [9.0, 5.0, 2.0]},   # diagonale X-Z
    {"start": [8.0, 8.0, 1.0], "end": [2.0, 2.0, 9.0]},   # diagonale inverse
]


# ── Chargement audio ─────────────────────────────────────────────────────
def charger_audio(chemin: str, fs_cible: int) -> np.ndarray:
    """Charge un fichier audio (wav/mp3/flac/ogg) et le resample à fs_cible. Retourne un array mono."""
    try:
        data, sr = sf.read(chemin, dtype="float64")
    except RuntimeError:
        # soundfile ne gère pas le mp3 nativement sur certaines installations ;
        # on tente via pydub + ffmpeg en fallback.
        from pydub import AudioSegment
        import io, tempfile
        seg = AudioSegment.from_file(chemin)
        seg = seg.set_channels(1)
        # Export en wav temporaire pour relire avec soundfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        seg.export(tmp.name, format="wav")
        data, sr = sf.read(tmp.name, dtype="float64")
        os.unlink(tmp.name)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample si nécessaire
    if sr != fs_cible:
        g = gcd(fs_cible, sr)
        data = resample_poly(data, fs_cible // g, sr // g)

    # Normalisation crête à 0.8
    mx = np.max(np.abs(data))
    if mx > 0:
        data = data / mx * 0.8

    return data


# ── Trajectoire ───────────────────────────────────────────────────────────
def faire_trajectoire(start, end, duree_trajet):
    """Retourne une fonction t -> position 3D (interpolation linéaire)."""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    def position(t):
        alpha = min(t / duree_trajet, 1.0)
        return start + (end - start) * alpha

    return position


# ── Programme principal ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simulation de propagation sonore multi-oiseaux.",
        epilog="Exemple : python generer_son4.py merle:bird_samples/Merle_noir_chant.mp3 mesange:bird_samples/Mésange_charbonnière_chant.mp3",
    )
    parser.add_argument(
        "oiseaux",
        nargs="+",
        help="Liste d'oiseaux au format nom:fichier_audio (ex: merle:son.wav)",
    )
    parser.add_argument("--duree", type=float, default=None, help="Durée simulation (s). Par défaut = durée du plus long son.")
    parser.add_argument("--out", default="sim_files", help="Dossier de sortie")
    parser.add_argument(
        "--trajectoire",
        default=None,
        help="Fichier JSON avec les trajectoires (clés = noms d'oiseaux, valeurs = {start, end})",
    )
    args = parser.parse_args()

    dossier_out = args.out

    # ── Parse des oiseaux ──
    oiseaux = []
    for spec in args.oiseaux:
        if ":" not in spec:
            parser.error(f"Format invalide '{spec}'. Attendu : nom:chemin_audio")
        nom, chemin = spec.split(":", 1)
        if not os.path.isfile(chemin):
            parser.error(f"Fichier introuvable : {chemin}")
        oiseaux.append({"nom": nom, "chemin": chemin})

    nb_oiseaux = len(oiseaux)
    print(f"--- {nb_oiseaux} oiseau(x) chargé(s) ---")

    # ── Chargement des trajectoires ──
    trajectoires_json = None
    if args.trajectoire and os.path.isfile(args.trajectoire):
        with open(args.trajectoire) as f:
            trajectoires_json = json.load(f)

    # ── Chargement audio + affectation trajectoire ──
    sources = []
    for i, oiseau in enumerate(oiseaux):
        print(f"  Chargement de '{oiseau['nom']}' depuis {oiseau['chemin']}...")
        sig = charger_audio(oiseau["chemin"], FS)
        duree_son = len(sig) / FS
        print(f"    duree = {duree_son:.1f} s  ({len(sig)} samples)")

        # Trajectoire : la durée du trajet = durée du son de cet oiseau
        if trajectoires_json and oiseau["nom"] in trajectoires_json:
            traj = trajectoires_json[oiseau["nom"]]
            pos_fn = faire_trajectoire(traj["start"], traj["end"], duree_son)
        else:
            traj_def = TRAJECTOIRES_DEFAUT[i % len(TRAJECTOIRES_DEFAUT)]
            pos_fn = faire_trajectoire(traj_def["start"], traj_def["end"], duree_son)

        sources.append({
            "nom": oiseau["nom"],
            "signal": sig,
            "n_samples": len(sig),
            "position": pos_fn,
        })

    # ── Durée de la simulation = max des durées audio (ou --duree si fourni) ──
    duree_max_audio = max(src["n_samples"] for src in sources) / FS
    duree = args.duree if args.duree is not None else duree_max_audio
    # On s'assure de couvrir au moins tous les sons
    duree = max(duree, duree_max_audio)
    n_samples = int(FS * duree)
    print(f"  Durée simulation : {duree:.1f} s")

    # ── Mixage & propagation ──
    print("--- CALCUL DE PROPAGATION ---")
    os.makedirs(dossier_out, exist_ok=True)

    mic_buffers = [np.zeros(n_samples) for _ in range(len(MICROS))]
    nb_blocs = n_samples // BLOC_SIZE

    # Ground truth
    ground_truth = {src["nom"]: {"x": [], "y": [], "z": []} for src in sources}
    gt_interval = max(1, int(0.1 * FS / BLOC_SIZE))  # toutes les ~0.1 s

    for b in range(nb_blocs):
        idx_start = b * BLOC_SIZE
        idx_end = idx_start + BLOC_SIZE
        t_now = idx_start / FS

        if b % 1000 == 0:
            print(f"  bloc {b}/{nb_blocs}  (t = {t_now:.1f} s)")

        for src in sources:
            # Cet oiseau a fini de chanter : on ne contribue plus
            if idx_start >= src["n_samples"]:
                continue

            # Le chunk peut être plus court que BLOC_SIZE à la fin du son
            end_sample = min(idx_end, src["n_samples"])
            chunk = src["signal"][idx_start:end_sample]
            chunk_len = len(chunk)

            pos = src["position"](t_now)

            # Ground truth
            if b % gt_interval == 0:
                ground_truth[src["nom"]]["x"].append(float(pos[0]))
                ground_truth[src["nom"]]["y"].append(float(pos[1]))
                ground_truth[src["nom"]]["z"].append(float(pos[2]))

            # Propagation vers chaque micro
            for m in range(len(MICROS)):
                dist = np.linalg.norm(MICROS[m] - pos)
                delay = int((dist / VITESSE_SON) * FS)
                attenuation = 1.0 / max(dist, 0.5)

                dest_start = idx_start + delay
                dest_end = dest_start + chunk_len
                if dest_end < n_samples:
                    mic_buffers[m][dest_start:dest_end] += chunk * attenuation

    # ── Sauvegarde ground truth ──
    gt_path = os.path.join(dossier_out, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  Ground truth -> {gt_path}")

    # ── Export WAV ──
    print("--- SAUVEGARDE DES FICHIERS WAV ---")
    for i in range(len(MICROS)):
        sig = mic_buffers[i]
        mx = np.max(np.abs(sig))
        if mx > 0:
            sig = sig / mx * 0.9

        filename = os.path.join(dossier_out, f"sim_mic_{i}.wav")
        sf.write(filename, sig, FS)
        print(f"  {filename}")

    print("--- TERMINÉ ---")


if __name__ == "__main__":
    main()
