import time
import numpy as np
from scipy import signal, ndimage
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from scipy.io.wavfile import write

from utils import micros
from config import CONFIG

def expand_and_filter_true_regions(xy, min_size=3):
    # Dilatation
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(xy, structure=structure)
    dilated = ndimage.binary_dilation(dilated, structure=np.array(np.ones(50), dtype=bool).reshape(1, -1))
    # Filtering of small regions
    labeled_array, num_features = ndimage.label(dilated)
    for i in range(1, num_features + 1):
        if np.sum(labeled_array == i) < min_size:
            dilated[labeled_array == i] = False
    return dilated

def compute_iou(box1, box2):
    # return intersection area / union area
    x0_1, x1_1, y0_1, y1_1 = box1
    x0_2, x1_2, y0_2, y1_2 = box2
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
    x0 = min(box[0] for box in boxes)
    x1 = max(box[1] for box in boxes)
    y0 = min(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return (x0, x1, y0, y1)

def aggregate_boxes(boxes):
    all_boxes = []
    for mac, bxs in boxes.items():
        all_boxes.extend(bxs)  # boxes_i is the list of boxes for group i
    n_boxes = len(all_boxes)
    if n_boxes == 0:
        return []
    overlap_matrix = np.zeros((n_boxes, n_boxes))
    for i in range(n_boxes):
        for j in range(n_boxes):
            overlap_matrix[i, j] = compute_iou(all_boxes[i], all_boxes[j])
    distance_matrix = 1 - overlap_matrix
    clustering = DBSCAN(eps=0.99, min_samples=1, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise (label = -1)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_boxes[i])
    return [ merge_boxes(boxes) for boxes in clusters.values() ]

def extract_patch(samples, mac, box):
    sample = samples[mac]
    tmin, tmax, fmin, fmax = box
    imin = int(tmin * CONFIG.SAMPLE_RATE)
    imax = int(tmax * CONFIG.SAMPLE_RATE)
    b, a = signal.butter(3, [fmin, fmax], fs=CONFIG.SAMPLE_RATE, btype='band')
    return signal.lfilter(b, a, sample[imin:imax])
    # return sample[imin:imax]

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    # https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    # find max cross correlation index
    shift = np.argmax(cc) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc

def box_tdoa(samples, box, mac1, mac2):
    s1 = extract_patch(samples, mac1, box)
    s2 = extract_patch(samples, mac2, box)
    return gcc_phat(s2, s1, fs=CONFIG.SAMPLE_RATE)

def multilateration_residual(source, mic_positions, distance_differences):
    residuals = []
    for i, j, diff in distance_differences:
        d_i = np.linalg.norm(source - mic_positions[i])
        d_j = np.linalg.norm(source - mic_positions[j])
        residuals.append(d_i - d_j + diff)
    return np.array(residuals)

nth = 0

def localiser(esps, t1, t2):
    global nth
    samples = {}
    boxes = {}
    positions = {}

    # reading every esp's latest data and computing spectrogram
    for mac, esp in esps.items():
        positions[mac] = esp.position

        t1r, t2r, s = esp.read_window(t1, t2)
        # print("READING", t1r, t2r)
        samples[mac] = s
        f, t, Sxx = signal.spectrogram(s, CONFIG.SAMPLE_RATE, nperseg=100, nfft=200) # f, t, Sxx

        # threshold filtering
        i = Sxx > Sxx.max() / 20                                # keeping only high-energy values
        i = i & np.repeat([f > 800], len(t), axis=0).T          # keeping only frequencies above 800 Hz
        # building boxes
        ii = expand_and_filter_true_regions(i, min_size=20)     # aggregating those values and removing the only ones
        labeled_array, num_features = ndimage.label(ii)
        boxes[mac] = []
        for i in range(num_features):
            iii = labeled_array == i+1
            # retrieving box boundaries
            true_indices = np.where(iii)
            x_min, x_max, y_min, y_max = np.min(true_indices[1]), np.max(true_indices[1]), np.min(true_indices[0]), np.max(true_indices[0])
            boxes[mac].append((t[x_min], t[x_max], f[y_min], f[y_max]))

    # aggregating boxes from all spectrograms together
    agg_boxes = aggregate_boxes(boxes)
    # keeping only boxes whose time length is superior to 0.3 s
    agg_boxes = list(filter(lambda box: box[1] - box[0] > 0.3, agg_boxes))
    # print("BOXES:", agg_boxes)

    initial_guess = np.mean(list(positions.values()), axis=0)

    sound_guesses = []
    for box in agg_boxes:
        tdoas = []
        sie_max = 0
        sie_max_si = None
        for i, (maci, si) in enumerate(samples.items()):
            for j, (macj, sj) in enumerate(samples.items()):
                if i >= j: continue
                # computing tdoa
                tdoa, cc = box_tdoa(samples, box, maci, macj)
                # removing values that are incoherent with the geometry
                if np.abs(tdoa) > 1.2 * np.linalg.norm(esps[maci].position - esps[macj].position): continue
                tdoas.append((maci, macj, tdoa))
            
            t = np.linspace(box[0], box[1], len(si))
            sie = np.trapz(np.abs(si)**2, t) / len(si)
            if sie > sie_max:
                sie_max = sie
                sie_max_si = si

        distance_differences = [(i, j, tdoa * 343) for i, j, tdoa in tdoas]
        result = least_squares(
            multilateration_residual,
            initial_guess,
            args=(positions, distance_differences),
        )
        estimated_sound_origin = result.x
        # print(f"saving {nth}")
        write(f"./out/{nth}.wav", CONFIG.SAMPLE_RATE, np.int16(sie_max_si / np.max(np.abs(sie_max_si)) * 32767))
        print(f" ! DETECTION ! {nth} (origin: {estimated_sound_origin})")
        nth += 1
        sound_guesses.append((estimated_sound_origin, box))

    # print("GUESSES:", sound_guesses)

def routine_localiser(esps):
    while True:
        t = micros()
        target_t2 = t - CONFIG.BUFFER_DELAY_US
        target_t1 = target_t2 - CONFIG.WINDOW_SIZE_VAD_US
        try:
            localiser(esps, target_t1, target_t2)
        finally:
            time.sleep(CONFIG.COMPUTE_INTERVAL_US / 1e6)