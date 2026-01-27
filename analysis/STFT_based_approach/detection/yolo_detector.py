"""
Détection YOLO sur spectrogrammes.
"""

import cv2
from ultralytics import YOLO

from config import Config


def detect_on_spectrogram(spectro_path: str, model: YOLO, duration: float) -> list:
    """
    Exécute la détection YOLO sur un spectrogramme.

    Args:
        spectro_path: Chemin vers l'image du spectrogramme
        model: Modèle YOLO chargé
        duration: Durée totale de l'audio (secondes)

    Returns:
        Liste de détections avec coordonnées physiques (temps, fréquence)
    """
    results = model.predict(
        source=spectro_path,
        conf=Config.CONF_THRESHOLD,
        iou=0.45,
        verbose=False
    )

    img = cv2.imread(spectro_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape

    t_max = duration
    f_max = Config.FS / 2  # Nyquist

    detections = []
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Conversion pixel -> physique
        t_start = (x1 / img_w) * t_max
        t_end = (x2 / img_w) * t_max

        # Fréquence (attention au flip vertical)
        y1_stft = img_h - y1
        y2_stft = img_h - y2
        f_start = (y2_stft / img_h) * f_max
        f_end = (y1_stft / img_h) * f_max

        t_center = (t_start + t_end) / 2
        f_center = (f_start + f_end) / 2

        detections.append({
            'class_id': cls,
            'confidence': conf,
            'bbox_pixel': [x1, y1, x2, y2],
            't_start': t_start,
            't_end': t_end,
            'f_min': f_start,
            'f_max': f_end,
            't_center': t_center,
            'f_center': f_center
        })

    return detections
