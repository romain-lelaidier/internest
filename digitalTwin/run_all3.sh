#!/bin/bash

# run_all3.sh - Lanceur complet v3 : generer_son4.py + triangulation v3 + IHM v2

cleanup() {
    echo -e "\n🛑 Arrêt de tous les services..."
    pkill -P $$
    exit
}

trap cleanup SIGINT

# Vérifier si le port 8080 est déjà utilisé
PORT_BUSY=$(lsof -i :8080)
if [ ! -z "$PORT_BUSY" ]; then
    echo "⚠️ Le port 8080 est déjà utilisé. Tentative de libération..."
    lsof -ti :8080 | xargs kill -9 > /dev/null 2>&1
    sleep 1
fi

echo "🧹 Nettoyage des anciennes données..."
rm -rf output_wavs input_packets sim_files live_positions.csv live_species.json
mkdir -p output_wavs input_packets

echo "🎵 Génération de l'audio de simulation (Aigle royal + Merle noir)..."
python3 generer_son4.py \
    aigle_royal:bird_samples/Aigle_royal_cri.mp3 \
    merle_noir:bird_samples/Merle_noir_chant.mp3 \
    --out sim_files

echo "🚀 Lancement des services..."

# 1. Serveur Web (IHM v2)
python3 viz_server_v2.py &
echo "  [OK] Serveur Web v2 (Port 8080)"
sleep 2
open "http://localhost:8080"

# 2. Triangulation v3 (localisation + BirdNET)
python3 triangulation_mvt_stream_v3.py &
echo "  [OK] Triangulation v3 (localisation + BirdNET)"

# 3. Pont WAV (Conversion Bin -> Wav)
python3 ecriture_fichiers_wav.py > /dev/null 2>&1 &
echo "  [OK] Pont WAV"

sleep 2

echo "-------------------------------------------------------"
echo "🟢 Accédez à l'IHM sur http://localhost:8080"
echo "-------------------------------------------------------"
echo "🚀 Démarrage de la simulation ESP..."

# 4. Simulation ESP
python3 simulate_esp.py

wait
