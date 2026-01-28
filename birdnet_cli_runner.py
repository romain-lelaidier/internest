import os
import subprocess
import sys
import pandas as pd

DETECTION_COL_CANDIDATES = [
    "Start (s)", "End (s)", "Start", "End",
    "Scientific name", "Common name", "Species", "Confidence", "Score"
]

def _is_detection_csv(df: pd.DataFrame) -> bool:
    cols = set(df.columns.astype(str))
    hits = sum(1 for c in DETECTION_COL_CANDIDATES if c in cols)
    return hits >= 2  # au moins 2 colonnes typiques

def run_birdnet_analyzer_on_folder(
    input_folder: str,
    output_folder: str,
    min_conf: float,
    top_n: int,
    lat: float,
    lon: float,
):
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze",
        input_folder,
        "--output", output_folder,
        "--rtype", "csv",
        "--min_conf", str(min_conf),
        "--top_n", str(top_n),
        "--lat", str(lat),
        "--lon", str(lon),
    ]

    subprocess.run(cmd, check=True)

    # ✅ prendre uniquement les CSV de résultats (pas le params)
    csv_files = [
        os.path.join(output_folder, fn)
        for fn in os.listdir(output_folder)
        if fn.lower().endswith(".birdnet.results.csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV de résultats BirdNET (*.BirdNET.results.csv) trouvé dans {output_folder}."
        )

    csv_files.sort(key=os.path.getmtime, reverse=True)
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    return df, csv_path