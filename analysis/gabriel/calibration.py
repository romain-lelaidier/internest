import numpy as np
import os
import time
import librosa
import scipy.signal as signal
from sklearn.manifold import MDS
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. CONFIGURATION ---
INPUT_WAV_DIR = "./output_wavs" # Dossier où le script watchdog écrit
VITESSE_SON = 343.0
N_MICROS = 5
FREQUENCES_BUZZERS = [1000, 2000, 3000, 4000, 5000] # Hz (Indicatif pour l'instant)

# Seuil pour détecter le "Beep" (Amplitude relative 0.0 à 1.0)
SEUIL_DETECTION_BEEP = 0.3 

# --- 2. FONCTIONS DE SYNCHRONISATION ---

def attendre_sync_fichiers(dossier, timeout=10):
    """ Vérifie que les 5 fichiers micros sont présents et récents """
    print("⏳ Attente de la synchronisation des 5 micros...")
    start = time.time()
    
    while (time.time() - start) < timeout:
        # On cherche le batch le plus récent
        batches = sorted([d for d in os.listdir(dossier) if d.startswith('batch_')], 
                         key=lambda x: int(x.split('_')[1]), reverse=True)
        
        if not batches:
            time.sleep(0.5); continue
            
        latest_batch = os.path.join(dossier, batches[0])
        fichiers = [f for f in os.listdir(latest_batch) if f.endswith('.wav')]
        
        if len(fichiers) == N_MICROS:
            # Vérifier qu'ils ont été modifiés récemment (< 2s)
            timestamps = [os.path.getmtime(os.path.join(latest_batch, f)) for f in fichiers]
            if time.time() - max(timestamps) < 3.0:
                print(f"✅ Synchronisation OK sur {latest_batch}")
                return latest_batch
        
        time.sleep(0.5)
    
    print("❌ Timeout: Les micros ne semblent pas envoyer de données.")
    return None

def envoyer_commande_buzzer(mic_id):
    """ 
    PLACEHOLDER : C'est ici que vous mettriez votre code réseau (UDP/HTTP) 
    pour dire à l'ESP numéro mic_id d'activer son buzzer.
    """
    print(f"\n📢 ACTION : Veuillez faire bipper le MICRO {mic_id} (Freq {FREQUENCES_BUZZERS[mic_id]}Hz) !")
    # requests.get(f"http://192.168.1.10{mic_id}/buzz") # Exemple

def analyser_temps_vol(dossier_batch, emetteur_id):
    """
    Analyse les fichiers audio pour trouver les distances.
    Principe : T0 = Moment où le Mic Emetteur sature.
    """
    print(f"   -> Analyse acoustique du tir {emetteur_id}...")
    
    # 1. Charger le son de l'émetteur pour trouver le T0
    f_emetteur = os.path.join(dossier_batch, f"mic_{emetteur_id}.wav")
    y_ref, fs = librosa.load(f_emetteur, sr=None)
    
    # On cherche le pic d'énergie (le buzzer est fort sur son propre micro)
    envelope = np.abs(y_ref)
    pic_idx = np.argmax(envelope)
    
    if envelope[pic_idx] < SEUIL_DETECTION_BEEP:
        print(f"⚠️  Pas de beep détecté sur le Mic {emetteur_id} (Trop faible).")
        return None
        
    t0_index = pic_idx
    distances = np.zeros(N_MICROS)
    
    # 2. Calculer le retard sur les autres micros
    for i in range(N_MICROS):
        if i == emetteur_id:
            distances[i] = 0.0
            continue
            
        f_recepteur = os.path.join(dossier_batch, f"mic_{i}.wav")
        y_rec, _ = librosa.load(f_recepteur, sr=None)
        
        # Corrélation croisée autour du pic pour trouver le décalage
        # On prend une fenêtre autour du T0 théorique
        win_size = int(fs * 0.05) # 50ms de fenêtre
        start = max(0, t0_index - win_size)
        end = min(len(y_rec), t0_index + win_size*4) # On regarde surtout après
        
        segment_ref = y_ref[start:end]
        segment_rec = y_rec[start:end]
        
        # Cross-correlation
        corr = signal.correlate(segment_rec, segment_ref, mode='same')
        lag = signal.correlation_lags(len(segment_rec), len(segment_ref), mode='same')
        best_lag = lag[np.argmax(corr)]
        
        # Délai en secondes
        dt = best_lag / fs
        
        # Si dt est négatif, c'est que le son est arrivé "avant" (impossible physiquement, bruit ?)
        # Sauf erreur de synchro d'horloge. On prend la valeur absolue si petite, sinon 0.
        dist = np.abs(dt) * VITESSE_SON
        distances[i] = dist
        
        print(f"      Dist M{emetteur_id}->M{i} : {dist:.3f} m")
        
    return distances


# --- 3. MAIN SEQUENCE ---

def main():
    print("=== CALIBRATION AUTOMATIQUE DES MICROS ===")
    
    # A. Attente Synchro
    last_batch = attendre_sync_fichiers(INPUT_WAV_DIR)
    if not last_batch: return

    # B. Demande utilisateur
    choix = input("\nLes 5 micros sont prêts. Voulez-vous lancer la calibration ? (o/n) : ")
    if choix.lower() != 'o': return

    # C. Acquisition de la matrice de distances
    dist_matrix = np.zeros((N_MICROS, N_MICROS))
    
    for i in range(N_MICROS):
        # 1. Déclencher le buzzer i
        envoyer_commande_buzzer(i)
        
        # 2. Attendre que le son soit enregistré (dépend de votre latence watchdog)
        # On demande à l'utilisateur de confirmer ou on attend X secondes
        input(f"Appuyez sur ENTRÉE dès que le bip du Mic {i} a retenti...")
        
        # 3. Récupérer le dernier batch généré (qui contient le bip)
        time.sleep(1.0) # Sécurité pour laisser l'écriture finir
        batch_bip = attendre_sync_fichiers(INPUT_WAV_DIR, timeout=5)
        
        # 4. Analyser
        dists = analyser_temps_vol(batch_bip, i)
        
        if dists is not None:
            dist_matrix[i, :] = dists
        else:
            print("❌ Échec sur ce micro. On met des distances par défaut.")
    
    # Symétrisation de la matrice (Moyenne des mesures A->B et B->A)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    print("\n--- MATRICE DE DISTANCES MESURÉE (m) ---")
    print(np.round(dist_matrix, 2))
    
    # --- 4. LE SCRIPT MDS QUE VOUS AVEZ FOURNI ---
    print("\n--- RECONSTRUCTION GÉOMÉTRIQUE (MDS) ---")
    
    mds = MDS(n_components=3, metric=True, dissimilarity='precomputed', random_state=42)
    positions_mds = mds.fit_transform(dist_matrix)

    # Fonction d'alignement (Votre code)
    def align_to_mic0(points):
        t = points[0]
        points_centered = points - t
        p1 = points_centered[1]
        u = p1 / np.linalg.norm(p1)
        p2 = points_centered[2]
        w = np.cross(u, p2)
        w = w / np.linalg.norm(w)
        v = np.cross(w, u)
        R = np.array([u, v, w])
        points_aligned = np.dot(points_centered, R.T)
        if points_aligned[4][2] < 0:
            points_aligned[:, 2] *= -1
        return points_aligned

    try:
        positions_finales = align_to_mic0(positions_mds)
        
        print("\n--- POSITIONS CALCULÉES ---")
        for idx, pos in enumerate(positions_finales):
            print(f"Mic {idx} : {np.round(pos, 3)}")
            
        # Sauvegarde
        np.savetxt("positions_calibrees.csv", positions_finales, delimiter=",", header="X,Y,Z")
        print("\n💾 Positions sauvegardées dans 'positions_calibrees.csv'")

        # --- 5. VISUALISATION ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, pos in enumerate(positions_finales):
            ax.scatter(pos[0], pos[1], pos[2], s=100, label=f'Mic {i}')
            ax.text(pos[0], pos[1], pos[2]+0.1, f"M{i}", fontsize=10)
        
        # Axes
        max_val = np.max(np.abs(positions_finales))
        ax.set_xlim(-max_val, max_val); ax.set_ylim(-max_val, max_val); ax.set_zlim(0, max_val)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Erreur MDS : {e}")
        print("Vérifiez que la matrice de distance n'est pas remplie de zéros.")

if __name__ == "__main__":
    main()
