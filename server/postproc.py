"""
Post-traitement audio : detection d'evenements sonores par spectrogramme,
localisation 3D par TDOA (GCC-PHAT + multilateration), puis identification
des especes via BirdNET (dans Sample.analyze()).

Pipeline :
  1. Lecture des buffers audio de chaque ESP
  2. Spectrogramme + seuillage energie → bounding boxes (temps x frequence)
  3. Agregation des boxes entre ESPs (IoU + DBSCAN)
  4. Pour chaque box : TDOA par paires de micros → multilateration → position 3D
  5. Creation d'un Sample (position + signal) → analyse BirdNET
"""

import time
import numpy as np
from scipy import signal, ndimage
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

from utils import micros
from config import CONFIG
from sample import Sample


# ---------------------------------------------------------------------------
#  Utilitaires spectrogramme : dilatation morphologique et filtrage
# ---------------------------------------------------------------------------

def expand_and_filter_true_regions(xy, min_size=3):
    """
    Dilate un masque binaire 2D (spectrogramme seuille) puis supprime
    les regions trop petites.

    1. Dilatation 3x3 pour connecter les pixels voisins
    2. Dilatation horizontale (1x50) pour relier les segments temporels proches
    3. Labellisation des composantes connexes
    4. Suppression des regions dont la taille < min_size

    Parametres :
        xy       : masque binaire 2D (frequence x temps)
        min_size : taille minimale en pixels pour conserver une region
    Retourne :
        masque binaire filtre (memes dimensions que xy)
    """
    # Dilatation 3x3 : connecte les pixels proches dans toutes les directions
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(xy, structure=structure)
    # Dilatation horizontale (1x50) : fusionne les segments temporellement proches
    dilated = ndimage.binary_dilation(dilated, structure=np.array(np.ones(50), dtype=bool).reshape(1, -1))
    # Labellisation des regions connexes et suppression des petites
    labeled_array, num_features = ndimage.label(dilated)
    for i in range(1, num_features + 1):
        if np.sum(labeled_array == i) < min_size:
            dilated[labeled_array == i] = False
    return dilated


# ---------------------------------------------------------------------------
#  Utilitaires boxes : IoU, fusion, agregation par clustering
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """
    Calcule l'Intersection over Union (IoU) entre deux boxes.
    Chaque box = (t_min, t_max, f_min, f_max).
    Retourne un score entre 0 (aucun chevauchement) et 1 (identiques).
    """
    x0_1, x1_1, y0_1, y1_1 = box1
    x0_2, x1_2, y0_2, y1_2 = box2
    # Coordonnees de l'intersection
    x0_inter = max(x0_1, x0_2)
    x1_inter = min(x1_1, x1_2)
    y0_inter = max(y0_1, y0_2)
    y1_inter = min(y1_1, y1_2)
    inter_area = max(0, x1_inter - x0_inter) * max(0, y1_inter - y0_inter)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def merge_boxes(boxes):
    """Fusionne une liste de boxes en une seule englobante (enveloppe convexe rectangulaire)."""
    x0 = min(box[0] for box in boxes)
    x1 = max(box[1] for box in boxes)
    y0 = min(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return (x0, x1, y0, y1)


def aggregate_boxes(boxes):
    """
    Regroupe les boxes de tous les ESPs en clusters via DBSCAN.

    Les boxes qui se chevauchent fortement (IoU > 0.003, soit eps=0.997)
    sont considerees comme le meme evenement sonore et fusionnees.

    Parametres :
        boxes : dict { mac: [(t_min, t_max, f_min, f_max), ...] }
    Retourne :
        liste de boxes fusionnees [(t_min, t_max, f_min, f_max), ...]
    """
    # Rassemble toutes les boxes de tous les ESPs
    all_boxes = []
    for mac, bxs in boxes.items():
        all_boxes.extend(bxs)
    n_boxes = len(all_boxes)
    if n_boxes == 0:
        return []
    # Matrice d'IoU entre toutes les paires de boxes
    overlap_matrix = np.zeros((n_boxes, n_boxes))
    for i in range(n_boxes):
        for j in range(n_boxes):
            overlap_matrix[i, j] = compute_iou(all_boxes[i], all_boxes[j])
    # DBSCAN sur la matrice de distance (1 - IoU)
    # eps=0.997 → deux boxes sont voisines si IoU > 0.003
    distance_matrix = 1 - overlap_matrix
    clustering = DBSCAN(eps=0.997, min_samples=1, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    # Regroupement par cluster et fusion
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # -1 = bruit (ignore)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_boxes[i])
    return [ merge_boxes(boxes) for boxes in clusters.values() ]


# ---------------------------------------------------------------------------
#  TDOA : extraction de patch, GCC-PHAT, multilateration
# ---------------------------------------------------------------------------

def extract_patch(samples, mac, box):
    """
    Extrait et filtre passe-bande un segment audio correspondant a une box.

    Parametres :
        samples : dict { mac: signal_audio_complet }
        mac     : adresse MAC de l'ESP
        box     : (t_min, t_max, f_min, f_max) en secondes / Hz
    Retourne :
        signal filtre dans la bande [f_min, f_max] et la fenetre [t_min, t_max]
    """
    sample = samples[mac]
    tmin, tmax, fmin, fmax = box
    # Conversion temps → indices d'echantillons
    imin = int(tmin * CONFIG.SAMPLE_RATE)
    imax = int(tmax * CONFIG.SAMPLE_RATE)
    # Filtre Butterworth passe-bande ordre 3 sur la bande de frequence de la box
    b, a = signal.butter(3, [fmin, fmax], fs=CONFIG.SAMPLE_RATE, btype='band')
    return signal.lfilter(b, a, sample[imin:imax])


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    Generalized Cross Correlation - Phase Transform (GCC-PHAT).
    Estime le decalage temporel (tau) entre sig et refsig.

    Source : https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py

    Parametres :
        sig     : signal a comparer
        refsig  : signal de reference
        fs      : frequence d'echantillonnage
        max_tau : decalage max autorise (en secondes), None = pas de limite
        interp  : facteur d'interpolation pour affiner la precision
    Retourne :
        (tau, cc) : decalage en secondes, intercorrelation complete
    """
    # Taille FFT >= len(sig) + len(refsig) pour eviter le repliement circulaire
    n = sig.shape[0] + refsig.shape[0]
    # FFT des deux signaux
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    # Intercorrelation normalisee en phase (GCC-PHAT)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    # Limitation du decalage max
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    # Le pic de correlation donne le decalage temporel
    shift = np.argmax(cc) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def box_tdoa(samples, box, mac1, mac2):
    """
    Calcule le TDOA entre deux ESPs pour une box donnee.
    Retourne le decalage temporel (positif = mac2 recoit avant mac1).
    """
    s1 = extract_patch(samples, mac1, box)
    s2 = extract_patch(samples, mac2, box)
    return gcc_phat(s2, s1, fs=CONFIG.SAMPLE_RATE)


def multilateration_residual(source, mic_positions, distance_differences):
    """
    Fonction de cout pour la multilateration par moindres carres.

    Pour chaque paire (i, j), le residu = d(source, mic_i) - d(source, mic_j) + diff_mesuree.
    Si source est a la bonne position, tous les residus → 0.

    Parametres :
        source               : position 3D candidate [x, y, z]
        mic_positions        : dict { mac: np.array([x, y, z]) }
        distance_differences : [(mac_i, mac_j, delta_distance_metres), ...]
    """
    residuals = []
    for i, j, diff in distance_differences:
        d_i = np.linalg.norm(source - mic_positions[i])
        d_j = np.linalg.norm(source - mic_positions[j])
        residuals.append(d_i - d_j + diff)
    return np.array(residuals)


# ---------------------------------------------------------------------------
#  Pipeline principal : spectrogramme → boxes → TDOA → localisation
# ---------------------------------------------------------------------------

def localiser(esps, t1, t2):
    """
    Pipeline complet de localisation sonore.

    Parametres :
        esps : dict { mac: ESP } avec les buffers audio remplis
        t1   : timestamp debut de la fenetre (microsecondes)
        t2   : timestamp fin de la fenetre (microsecondes)
    Retourne :
        liste de Sample (position 3D estimee + signal le plus energetique)
    """
    samples = {}    # { mac: signal_audio_filtre }
    boxes = {}      # { mac: [(t_min, t_max, f_min, f_max), ...] }
    positions = {}  # { mac: np.array([x, y, z]) }

    # --- Etape 1 : lecture audio + spectrogramme + detection de boxes par ESP ---
    for mac, esp in esps.items():
        try:
            t1r, t2r, s = esp.read_window(t1, t2)
            if len(s) > 100:

                # Filtre passe-haut a 800 Hz pour supprimer le bruit basse frequence
                b, a = signal.butter(3, 800, fs=CONFIG.SAMPLE_RATE, btype='hp')
                s = signal.lfilter(b, a, s)

                # Spectrogramme (fenetres de 100 echantillons, FFT 200 points)
                f, t, Sxx = signal.spectrogram(s, CONFIG.SAMPLE_RATE, nperseg=100, nfft=200)
                positions[mac] = esp.position
                samples[mac] = s

                # Seuillage : on garde les pixels dont l'energie > max/30
                i = Sxx > Sxx.max() / 30
                # Masque frequentiel : on ne garde que les frequences > 800 Hz
                i = i & np.repeat([f > 800], len(t), axis=0).T
                # Dilatation + filtrage des petites regions
                ii = expand_and_filter_true_regions(i, min_size=20)
                # Labellisation des composantes connexes → une box par region
                labeled_array, num_features = ndimage.label(ii)
                boxes[mac] = []
                for i in range(num_features):
                    iii = labeled_array == i+1
                    # Coordonnees de la bounding box (indices → temps/frequence)
                    true_indices = np.where(iii)
                    x_min, x_max = np.min(true_indices[1]), np.max(true_indices[1])
                    y_min, y_max = np.min(true_indices[0]), np.max(true_indices[0])
                    t_min, t_max, f_min, f_max = t[x_min], t[x_max], f[y_min], f[y_max]
                    # Ajout d'une duree minimale pour capturer la fin du son
                    boxes[mac].append((t_min, t_max + CONFIG.BIRD_SOUND_MIN_DURATION, f_min, f_max))
        except:
            continue

    # --- Etape 2 : agregation des boxes de tous les ESPs ---
    agg_boxes = aggregate_boxes(boxes)
    # On ne garde que les sons d'au moins 0.3 s
    agg_boxes = list(filter(lambda box: box[1] - box[0] > 0.3, agg_boxes))

    # Point initial pour la multilateration = barycentre des micros
    initial_guess = np.mean(list(positions.values()), axis=0)

    # --- Etape 3 : pour chaque box, TDOA + multilateration ---
    sound_guesses = []
    for box in agg_boxes:
        tdoas = []
        sie_max = 0        # energie max parmi les ESPs (pour garder le meilleur signal)
        sie_max_si = None   # signal correspondant a l'ESP la plus energetique

        for i, (maci, si) in enumerate(samples.items()):
            for j, (macj, sj) in enumerate(samples.items()):
                if i >= j: continue
                # TDOA entre les deux ESPs pour cette box
                tdoa, cc = box_tdoa(samples, box, maci, macj)
                # Coherence geometrique : le TDOA ne peut pas depasser
                # 1.2x la distance entre les deux micros / vitesse du son
                if np.abs(tdoa) > 1.2 * np.linalg.norm(esps[maci].position - esps[macj].position): continue
                tdoas.append((maci, macj, tdoa))

            # Calcul de l'energie du signal pour cette ESP sur la duree de la box
            t = np.linspace(box[0], box[1], len(si))
            sie = np.trapz(np.abs(si)**2, t) / len(si)
            if sie > sie_max:
                sie_max = sie
                sie_max_si = si

        # Conversion TDOA (secondes) → differences de distance (metres, v_son = 343 m/s)
        distance_differences = [(i, j, tdoa * 343) for i, j, tdoa in tdoas]
        # Multilateration par moindres carres
        result = least_squares(
            multilateration_residual,
            initial_guess,
            args=(positions, distance_differences),
        )
        estimated_sound_origin = result.x
        # Creation du Sample avec la position estimee et le signal le plus energetique
        sound_guesses.append(Sample(estimated_sound_origin, sie_max_si))

    return sound_guesses


# ---------------------------------------------------------------------------
#  Routine principale (tourne en thread)
# ---------------------------------------------------------------------------

def routine_postproc(esps):
    """
    Boucle infinie de post-traitement.
    A chaque iteration :
      1. Definit la fenetre temporelle a analyser (en tenant compte du delai buffer)
      2. Lance le pipeline localiser() → liste de Samples
      3. Analyse chaque Sample avec BirdNET (sample.analyze())
      4. Attend COMPUTE_INTERVAL_US avant la prochaine iteration
    """
    while True:
        t = micros()
        # Fenetre d'analyse : on recule de BUFFER_DELAY pour laisser le buffer se remplir
        target_t2 = t - CONFIG.BUFFER_DELAY_US
        target_t1 = target_t2 - CONFIG.WINDOW_SIZE_VAD_US
        try:
            samples = localiser(esps, target_t1, target_t2)
            # Analyse BirdNET sur chaque son localise
            for sample in samples:
                sample.analyze()

        finally:
            time.sleep(CONFIG.COMPUTE_INTERVAL_US / 1e6)
