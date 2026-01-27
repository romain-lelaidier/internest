"""
Clustering des détections (DBSCAN local et global).
"""

import numpy as np
from sklearn.cluster import DBSCAN

from config import Config


def cluster_detections(detections: list) -> tuple:
    """
    Clusterise les détections par DBSCAN sur (f_center, f_min, f_max).

    Args:
        detections: Liste des détections d'un micro

    Returns:
        (labels, n_clusters)
    """
    if len(detections) == 0:
        return np.array([]), 0

    X = np.array([
        [det['f_center'], det['f_min'], det['f_max']]
        for det in detections
    ])

    # Normalisation par eps_freq
    X_scaled = X / Config.EPS_FREQ

    dbscan = DBSCAN(eps=1.0, min_samples=Config.MIN_SAMPLES, metric='euclidean')
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


def compute_global_cluster_map(clustered_data: dict) -> dict:
    """
    Établit la correspondance entre clusters locaux et identités globales.

    Args:
        clustered_data: {mic_id: {'detections': [...], 'n_clusters': int}}

    Returns:
        dict: {(mic_id, local_cluster): global_bird_id}
    """
    cluster_centers = []
    cluster_infos = []

    for mic_id, data in clustered_data.items():
        detections = data['detections']
        n_clusters = data['n_clusters']
        if n_clusters == 0:
            continue

        for cluster_id in range(n_clusters):
            cluster_dets = [d for d in detections if d.get('cluster') == cluster_id]
            if len(cluster_dets) == 0:
                continue

            mean_f_center = np.mean([d['f_center'] for d in cluster_dets])
            mean_f_min = np.mean([d['f_min'] for d in cluster_dets])
            mean_f_max = np.mean([d['f_max'] for d in cluster_dets])

            cluster_centers.append([mean_f_center, mean_f_min, mean_f_max])
            cluster_infos.append({'mic_id': mic_id, 'cluster_id': cluster_id})

    if len(cluster_centers) == 0:
        return {}

    cluster_centers = np.array(cluster_centers)

    # Re-clustering global
    X_scaled = cluster_centers / 5000
    dbscan_global = DBSCAN(eps=0.1, min_samples=2, metric='euclidean')
    global_labels = dbscan_global.fit_predict(X_scaled)

    # Construction du mapping
    global_map = {}
    for idx, info in enumerate(cluster_infos):
        global_map[(info['mic_id'], info['cluster_id'])] = int(global_labels[idx])

    return global_map
