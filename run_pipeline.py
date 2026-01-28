import os
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import detector_bandenergy

from utils_audio import read_audio, write_wav, slice_audio, clamp_interval
from detector_bandenergy import band_energy_series, detect_events_from_energy
from count_estimator import estimate_num_birds
from birdnet_cli_runner import run_birdnet_analyzer_on_folder

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    os.makedirs(cfg["work_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

def make_event_segments(events, total_s, cfg):
    segs = []
    pad = float(cfg["context_pad_s"])
    seg_len = float(cfg["birdnet_segment_s"])

    for (s, e) in events:
        center = 0.5 * (s + e)
        start = center - 0.5 * seg_len - pad
        end = center + 0.5 * seg_len + pad
        start, end = clamp_interval(start, end, total_s)

        # re-caler exactement à seg_len + 2*pad si possible
        target_len = seg_len + 2*pad
        if (end - start) < target_len:
            # étendre si possible
            extra = target_len - (end - start)
            start = max(0.0, start - extra/2)
            end = min(total_s, end + extra/2)

        segs.append((start, end))
    return segs

def to_jsonable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj

def main(
    audio_path: str,
    cfg_path: str = "config.yaml",
    multichannel: bool = True,
):
    cfg = load_cfg(cfg_path)
    ensure_dirs(cfg)

    sr = int(cfg["sample_rate"])

    # Lecture audio
    if multichannel:
        Y, sr = read_audio(audio_path, target_sr=sr, mono=False)  # (C, N)
        mix = np.mean(Y, axis=0)
        num_channels = Y.shape[0]
    else:
        mix, sr = read_audio(audio_path, target_sr=sr, mono=True)
        Y = None
        num_channels = 1

    total_s = len(mix) / sr

    # Détection légère sur mix
    times, energy = band_energy_series(
        mix, sr,
        win_s=float(cfg["scan_window_s"]),
        hop_s=float(cfg["scan_hop_s"]),
        f_low=int(cfg["band_low_hz"]),
        f_high=int(cfg["band_high_hz"]),
    )

    events = detect_events_from_energy(
        times, energy,
        threshold_quantile=float(cfg["energy_threshold_quantile"]),
        min_event_s=float(cfg["min_event_s"]),
        merge_gap_s=float(cfg["merge_gap_s"]),
    )

    # Générer segments BirdNET
    segs = make_event_segments(events, total_s, cfg)

    work_segments_dir = os.path.join(cfg["work_dir"], "segments")
    os.makedirs(work_segments_dir, exist_ok=True)

    manifest = []
    for idx, (s, e) in enumerate(segs):
        seg_mix = slice_audio(mix, sr, s, e)

        # estimation nb oiseaux (proxy) sur le segment (mix)
        n_est, dbg = estimate_num_birds(
            seg_mix, sr,
            f_low=int(cfg["band_low_hz"]),
            f_high=int(cfg["band_high_hz"]),
        )

        # écrire segment pour BirdNET
        seg_fn = f"seg_{idx:06d}_{s:.2f}_{e:.2f}.wav"
        seg_path = os.path.join(work_segments_dir, seg_fn)
        write_wav(seg_path, seg_mix, sr)

        manifest.append({
            "segment_index": idx,
            "segment_file": seg_fn,
            "segment_path": seg_path,
            "start_s": float(s),
            "end_s": float(e),
            "n_birds_est": int(n_est),
            "count_debug": dbg,
        })

    # Lancer BirdNET sur les segments
    birdnet_out_dir = os.path.join(cfg["work_dir"], "birdnet_out")
    df_bird, csv_path = run_birdnet_analyzer_on_folder(
        input_folder=work_segments_dir,
        output_folder=birdnet_out_dir,
        min_conf=float(cfg["min_conf"]),
        top_n=int(cfg["top_n"]),
        lat=float(cfg["lat"]),
        lon=float(cfg["lon"]),
    )

    # Consolidation : joindre BirdNET CSV avec manifest via nom de fichier
    man_df = pd.DataFrame(manifest)
    print("man_df columns:", man_df.columns.tolist())
    print("nb segments:", len(man_df))
    # BirdNET a souvent une colonne "File" ou "Filename" ; on cherche robustement
    file_col = None
    for cand in ["File", "file", "Filename", "filename", "Path", "path", "Audio file", "Audio", "audio"]:
        if cand in df_bird.columns:
            file_col = cand
            break
    if file_col is None:
        # fallback: première colonne contenant ".wav"
        for c in df_bird.columns:
            if df_bird[c].astype(str).str.contains(".wav", regex=False).any():
                file_col = c
                break
    if file_col is None:
        raise RuntimeError(f"Impossible d'identifier la colonne fichier dans le CSV BirdNET: {df_bird.columns.tolist()}")

    # Normaliser le nom de segment (basename)
    df_bird["segment_file"] = df_bird[file_col].astype(str).apply(lambda p: os.path.basename(p))
    merged = df_bird.merge(man_df, on="segment_file", how="left")

    # Export
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_json = os.path.join(cfg["output_dir"], f"{base}_detections.json")
    out_csv = os.path.join(cfg["output_dir"], f"{base}_detections.csv")

    merged.to_csv(out_csv, index=False)

    payload = {
        "audio_file": audio_path,
        "sample_rate": sr,
        "num_channels": num_channels,
        "segments_extracted": len(manifest),
        "birdnet_csv_source": csv_path,
        "detections": merged.to_dict(orient="records"),
    }
    payload = to_jsonable(payload)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK ✅  segments: {len(manifest)}")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Chemin vers un WAV (mono ou multicanal)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mono", action="store_true", help="Force interprétation mono")
    args = ap.parse_args()

    main(args.audio, cfg_path=args.config, multichannel=(not args.mono))
