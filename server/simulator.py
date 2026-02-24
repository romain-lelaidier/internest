"""
Simulateur TDOA — genere des signaux synthetiques avec delais de propagation
realistes, puis verifie que le pipeline localiser() retrouve la bonne position.

Usage :
    cd server && python simulator.py           # GUI par defaut (v2)
    cd server && python simulator.py --no-gui  # console uniquement
    cd server && python simulator.py --pipeline v1
"""

import sys
import types
import numpy as np

from config import CONFIG

# ---------------------------------------------------------------------------
#  Constantes
# ---------------------------------------------------------------------------

SPEED_OF_SOUND = 343.0  # m/s
SAMPLE_RATE = CONFIG.SAMPLE_RATE
WINDOW_US = CONFIG.WINDOW_SIZE_VAD_US  # 2s en microsecondes
WINDOW_S = WINDOW_US / 1e6

BIP_FREQ = 2000    # Hz
BIP_DURATION = 0.3  # secondes
SNR_DB = 30

ESP_POSITIONS = {
    "esp0": np.array([0.0, 0.0, 0.0]),
    "esp1": np.array([0.0, 5.0, 0.0]),
    "esp2": np.array([7.0, 0.0, 0.0]),
    "esp3": np.array([6.0, 5.0, 2.3]),
    "esp4": np.array([0.0, 0.5, 2.3]),
}


ESP_LABELS = {
    "esp0": "ESP0 (vert)",
    "esp1": "ESP1 (rouge)",
    "esp2": "ESP2 (jaune buz)",
    "esp3": "ESP3 (bleu)",
    "esp4": "ESP4 (jaune)",
}


# ---------------------------------------------------------------------------
#  MockESP — imite l'interface utilisee par localiser()
# ---------------------------------------------------------------------------

class MockESP:
    def __init__(self, mac, position, audio_signal):
        self.mac = mac
        self.position = np.array(position, dtype=float)
        self._signal = np.array(audio_signal, dtype=np.int16)

    def read_window(self, t1, t2):
        """Retourne (t1, t2, signal_int16) — toute la fenetre."""
        return int(t1), int(t2), self._signal.copy()


# ---------------------------------------------------------------------------
#  Generation du signal synthetique
# ---------------------------------------------------------------------------

def make_bip(freq, duration, sample_rate):
    """Bip sinusoidal avec enveloppe de Hanning."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.hanning(n)
    return envelope * np.sin(2 * np.pi * freq * t)


def generate_signals(source_pos, esp_positions, sample_rate, window_s,
                     bip_freq, bip_duration, snr_db):
    """
    Pour chaque ESP, genere le signal recu (bip retarde + bruit).
    Retourne un dict { mac: signal_int16 }.
    """
    n_samples = int(window_s * sample_rate)
    bip = make_bip(bip_freq, bip_duration, sample_rate)
    bip_len = len(bip)

    # Position du bip au centre de la fenetre (sans delai)
    center = n_samples // 2 - bip_len // 2

    signals = {}
    for mac, esp_pos in esp_positions.items():
        dist = np.linalg.norm(source_pos - esp_pos)
        delay_s = dist / SPEED_OF_SOUND
        delay_samples = int(round(delay_s * sample_rate))

        buf = np.zeros(n_samples, dtype=np.float64)
        start = center + delay_samples
        end = start + bip_len
        if 0 <= start and end <= n_samples:
            buf[start:end] = bip

        # Bruit blanc
        sig_power = np.mean(bip ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), n_samples)
        buf += noise

        # Normalisation → int16
        peak = np.max(np.abs(buf))
        if peak > 0:
            buf = buf / peak * 30000
        signals[mac] = buf.astype(np.int16)

    return signals


# ---------------------------------------------------------------------------
#  Simulation (calcul)
# ---------------------------------------------------------------------------

class MockSample:
    """Remplace sample.Sample pour eviter de charger BirdNET."""
    def __init__(self, origin, s):
        self.origin = origin
        self.s = s
        self.species = []


def _mock_sample_module():
    """Injecte un faux module sample pour eviter de charger BirdNET."""
    mock_sample = types.ModuleType("sample")
    mock_sample.Sample = MockSample
    sys.modules["sample"] = mock_sample


def run_simulation(pipeline="v2"):
    """Lance la simulation et retourne (trajectory_real, trajectory_est, errors, label)."""
    _mock_sample_module()

    if pipeline == "v1":
        from postproc import localiser
        label = "postproc.py (v1)"
    else:
        from postproc_2 import localiser
        label = "postproc_2.py (v2)"

    n_steps = 10
    start = np.array([1.0, 1.0, 1.0])
    end = np.array([5.0, 4.0, 1.0])
    trajectory = [start + (end - start) * i / (n_steps - 1) for i in range(n_steps)]

    trajectory_real = []
    trajectory_est = []
    errors = []

    t1 = 1_000_000
    t2 = t1 + int(WINDOW_US)

    for step, source_pos in enumerate(trajectory):
        sigs = generate_signals(
            source_pos, ESP_POSITIONS, SAMPLE_RATE, WINDOW_S,
            BIP_FREQ, BIP_DURATION, SNR_DB,
        )
        esps = {mac: MockESP(mac, pos, sigs[mac]) for mac, pos in ESP_POSITIONS.items()}
        has_activity, positions_3d = localiser(esps, t1, t2)

        trajectory_real.append(source_pos.copy())

        if not has_activity or len(positions_3d) == 0:
            trajectory_est.append(None)
            errors.append(None)
            print(f"  Pas {step:2d}  |  reel = {source_pos}  |  AUCUNE DETECTION")
            continue

        estimated = positions_3d[0]
        err = np.linalg.norm(estimated - source_pos)
        trajectory_est.append(estimated.copy())
        errors.append(err)

        print(
            f"  Pas {step:2d}  "
            f"|  reel = [{source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}]  "
            f"|  estime = [{estimated[0]:.2f}, {estimated[1]:.2f}, {estimated[2]:.2f}]  "
            f"|  erreur = {err:.3f} m"
        )

    valid = [e for e in errors if e is not None]
    print()
    print("-" * 70)
    if valid:
        print(f"  Erreur moyenne : {np.mean(valid):.3f} m")
        print(f"  Erreur min     : {np.min(valid):.3f} m")
        print(f"  Erreur max     : {np.max(valid):.3f} m")
    else:
        print("  Aucune detection.")
    print("-" * 70)

    return trajectory_real, trajectory_est, errors, label


# ---------------------------------------------------------------------------
#  Simulation IHM (temps reel dans le navigateur)
# ---------------------------------------------------------------------------

FAKE_SPECIES = [
    "Merle noir", "Mesange bleue", "Pinson des arbres",
    "Rouge-gorge familier", "Fauvette a tete noire",
]


def run_simulation_ihm():
    """Lance la simulation pas a pas, alimente l'IHM Flask en temps reel."""
    import time as _time
    from types import SimpleNamespace

    _mock_sample_module()
    from postproc_2 import localiser
    import postproc_2 as pp2
    from ihm_postproc2 import start_ihm_postproc2

    # Enregistrer les ESPs dans l'IHM
    fake_esps = {}
    for mac, pos in ESP_POSITIONS.items():
        fake_esps[mac] = SimpleNamespace(position=pos, mac=mac)
    pp2.ihm_esps = fake_esps

    # Lancer le serveur Flask
    start_ihm_postproc2()
    print()
    print(f"  IHM ouverte sur http://localhost:8010")
    print(f"  Simulation en cours... (1 pas / seconde)")
    print()

    # Trajectoire en boucle
    n_steps = 20
    center = np.array([3.5, 2.5, 1.0])
    t1 = 1_000_000
    t2 = t1 + int(WINDOW_US)
    step = 0

    while True:
        # Trajectoire circulaire
        angle = step * 0.3
        radius = 2.0
        source_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.3 * np.sin(angle * 0.5)
        ])

        sigs = generate_signals(
            source_pos, ESP_POSITIONS, SAMPLE_RATE, WINDOW_S,
            BIP_FREQ, BIP_DURATION, SNR_DB,
        )
        esps = {mac: MockESP(mac, pos, sigs[mac]) for mac, pos in ESP_POSITIONS.items()}
        has_activity, positions_3d = localiser(esps, t1, t2)

        now = _time.time()

        if has_activity and positions_3d:
            est = positions_3d[0]
            err = np.linalg.norm(est - source_pos)
            pp2.ihm_positions.append({
                'x': float(est[0]), 'y': float(est[1]), 'z': float(est[2]),
                'time': now
            })
            print(f"  Pas {step:3d} | estime [{est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f}] | err {err:.3f}m")
        else:
            print(f"  Pas {step:3d} | pas de detection")

        step += 1
        _time.sleep(1.0)


# ---------------------------------------------------------------------------
#  GUI matplotlib
# ---------------------------------------------------------------------------

def show_gui(trajectory_real, trajectory_est, errors, label):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d  # noqa: F401

    real = np.array(trajectory_real)
    # Separer les pas detectes / non detectes
    est_ok = [(i, e) for i, e in enumerate(trajectory_est) if e is not None]
    est_pts = np.array([e for _, e in est_ok]) if est_ok else np.empty((0, 3))
    est_idx = [i for i, _ in est_ok]

    esp_pts = np.array(list(ESP_POSITIONS.values()))
    esp_names = list(ESP_LABELS.values())

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Simulateur TDOA — {label}", fontsize=13, fontweight="bold")

    # ---- Vue 3D ----
    ax3d = fig.add_subplot(121, projection="3d")

    # ESPs
    ax3d.scatter(*esp_pts.T, c="grey", s=90, marker="^", zorder=5, depthshade=False)
    for pos, name in zip(esp_pts, esp_names):
        ax3d.text(pos[0], pos[1], pos[2] + 0.25, name, fontsize=7, ha="center",
                  color="grey")

    # Trajectoire reelle
    ax3d.plot(*real.T, "o-", color="#2196F3", label="Trajectoire reelle", markersize=5)
    ax3d.plot(*real[0], "s", color="#2196F3", markersize=9)   # depart
    ax3d.plot(*real[-1], "D", color="#2196F3", markersize=9)  # arrivee

    # Trajectoire estimee
    if len(est_pts):
        ax3d.plot(*est_pts.T, "x--", color="#F44336", label="Trajectoire estimee",
                  markersize=8, markeredgewidth=2)
        # Lignes d'erreur reel→estime
        for i, ep in zip(est_idx, est_pts):
            rp = real[i]
            ax3d.plot([rp[0], ep[0]], [rp[1], ep[1]], [rp[2], ep[2]],
                      color="#FFAB00", linewidth=0.8, alpha=0.7)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(fontsize=8, loc="upper left")
    ax3d.set_title("Vue 3D")

    # ---- Vue 2D (dessus, X-Y) + barplot erreur ----
    gs = fig.add_gridspec(2, 2, left=0.55, right=0.97, hspace=0.35, wspace=0.3)

    # -- X-Y --
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.scatter(esp_pts[:, 0], esp_pts[:, 1], c="grey", s=60, marker="^", zorder=5)
    for pos, name in zip(esp_pts, esp_names):
        ax_xy.annotate(name, (pos[0], pos[1]), fontsize=6, ha="center",
                       xytext=(0, 7), textcoords="offset points", color="grey")
    ax_xy.plot(real[:, 0], real[:, 1], "o-", color="#2196F3", markersize=4, label="Reel")
    if len(est_pts):
        ax_xy.plot(est_pts[:, 0], est_pts[:, 1], "x--", color="#F44336", markersize=7,
                   markeredgewidth=2, label="Estime")
        for i, ep in zip(est_idx, est_pts):
            rp = real[i]
            ax_xy.plot([rp[0], ep[0]], [rp[1], ep[1]], color="#FFAB00", lw=0.8, alpha=0.7)
    ax_xy.set_xlabel("X (m)", fontsize=8)
    ax_xy.set_ylabel("Y (m)", fontsize=8)
    ax_xy.legend(fontsize=7)
    ax_xy.set_title("Vue dessus (X-Y)", fontsize=9)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    # -- X-Z --
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.scatter(esp_pts[:, 0], esp_pts[:, 2], c="grey", s=60, marker="^", zorder=5)
    ax_xz.plot(real[:, 0], real[:, 2], "o-", color="#2196F3", markersize=4, label="Reel")
    if len(est_pts):
        ax_xz.plot(est_pts[:, 0], est_pts[:, 2], "x--", color="#F44336", markersize=7,
                   markeredgewidth=2, label="Estime")
    ax_xz.set_xlabel("X (m)", fontsize=8)
    ax_xz.set_ylabel("Z (m)", fontsize=8)
    ax_xz.legend(fontsize=7)
    ax_xz.set_title("Vue laterale (X-Z)", fontsize=9)
    ax_xz.grid(True, alpha=0.3)

    # -- Barplot erreur --
    ax_err = fig.add_subplot(gs[1, :])
    valid_errors = [e if e is not None else 0 for e in errors]
    colors = ["#F44336" if e is not None else "#BDBDBD" for e in errors]
    bars = ax_err.bar(range(len(errors)), valid_errors, color=colors, edgecolor="white",
                      linewidth=0.5)
    for i, e in enumerate(errors):
        if e is not None:
            ax_err.text(i, e + 0.005, f"{e:.2f}", ha="center", fontsize=7)
    valid = [e for e in errors if e is not None]
    if valid:
        ax_err.axhline(np.mean(valid), color="#FF6F00", linestyle="--", linewidth=1,
                       label=f"Moyenne = {np.mean(valid):.3f} m")
        ax_err.legend(fontsize=7)
    ax_err.set_xlabel("Pas", fontsize=8)
    ax_err.set_ylabel("Erreur (m)", fontsize=8)
    ax_err.set_title("Erreur par pas", fontsize=9)
    ax_err.set_xticks(range(len(errors)))
    ax_err.grid(True, axis="y", alpha=0.3)

    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.90)
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulateur TDOA")
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v2",
                        help="v1 = postproc.py, v2 = postproc_2.py (defaut)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Pas d'affichage graphique")
    parser.add_argument("--ihm", action="store_true",
                        help="Lance l'IHM web sur localhost:8010 avec simulation temps reel")
    args = parser.parse_args()

    print("=" * 70)
    print("  Simulateur TDOA")
    print("=" * 70)
    print()

    if args.ihm:
        run_simulation_ihm()
    else:
        traj_real, traj_est, errs, lbl = run_simulation(pipeline=args.pipeline)
        if not args.no_gui:
            show_gui(traj_real, traj_est, errs, lbl)
