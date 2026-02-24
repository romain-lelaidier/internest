import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev # <--- NÉCESSAIRE POUR LE LISSAGE
import os

# --- 1. CONFIGURATION ---
INPUT_CSV = "trajectoire_oiseaux.csv"

# Paramètres DBSCAN / Couture
VITESSE_POIDS_TEMPS = 5.0   
EPSILON_DISTANCE = 20.0       
MIN_SAMPLES = 4
VITESSE_MAX_OISEAU = 20.0    
TEMPS_MAX_SILENCE = 10.0      

# Paramètres de LISSAGE
FACTEUR_LISSAGE = 50.0  # Plus c'est haut, plus c'est lisse (mais s'éloigne des points)
                         # 0 = Passe par tous les points (bruité)
                         # Essayez entre 50 et 500 selon le bruit de vos micros.

# --- 2. FONCTIONS ---

def load_data(filename):
    if not os.path.exists(filename):
        # Simulation (identique avant)
        t1 = np.linspace(0, 5, 20)
        traj1 = np.column_stack((50+5*t1 + np.random.normal(0,1,20), 
                                 50+2*t1, 10+t1, t1))
        t2 = np.linspace(8, 13, 20)
        traj2 = np.column_stack((50+5*t2, 50+2*t2, 10+t2, t2))
        return np.vstack((traj1, traj2))
    else:
        try:
            raw = np.genfromtxt(filename, delimiter=',', skip_header=1)
            # Tri par temps pour éviter les erreurs de spline
            raw = raw[raw[:, 0].argsort()]
            return np.column_stack((raw[:,1], raw[:,2], raw[:,3], raw[:,0]))
        except:
            return None

def recoudre_trajectoires(clusters_dict):
    # (Votre fonction de couture précédente, inchangée)
    liste_trajs = []
    for cid, data in clusters_dict.items():
        data = data[data[:, 3].argsort()]
        liste_trajs.append({'id': cid, 'data': data})
    liste_trajs.sort(key=lambda x: x['data'][0, 3])
    
    merged_trajs = []
    while len(liste_trajs) > 0:
        curr = liste_trajs.pop(0)
        found_next = True
        while found_next:
            found_next = False
            best_idx = -1
            last_pt = curr['data'][-1] 
            p_end, t_end = last_pt[:3], last_pt[3]
            
            for i, candidate in enumerate(liste_trajs):
                first_pt = candidate['data'][0]
                p_start, t_start = first_pt[:3], first_pt[3]
                dt = t_start - t_end
                if dt <= 0 or dt > TEMPS_MAX_SILENCE: continue 
                
                dist = np.linalg.norm(p_start - p_end)
                vitesse_requise = dist / dt
                
                if vitesse_requise < VITESSE_MAX_OISEAU:
                    curr['data'] = np.vstack((curr['data'], candidate['data']))
                    best_idx = i
                    found_next = True
                    break
            if found_next:
                liste_trajs.pop(best_idx)
        merged_trajs.append(curr['data'])
    return merged_trajs

def generer_courbe_lisse(traj_data, s_factor=100):
    """
    Utilise des B-Splines pour générer une courbe fluide à partir de points bruités.
    """
    # Il faut au moins 4 points pour une spline cubique (k=3)
    if len(traj_data) < 4:
        return traj_data # Pas assez de points, on renvoie le brut
    
    x, y, z, t = traj_data[:,0], traj_data[:,1], traj_data[:,2], traj_data[:,3]
    
    # Nettoyage : Splprep plante si deux points ont exactement le même temps (doublons)
    # On ajoute un epsilon infinitésimal au temps si doublon
    t_unique = t.copy()
    for i in range(1, len(t_unique)):
        if t_unique[i] <= t_unique[i-1]:
            t_unique[i] = t_unique[i-1] + 1e-6

    try:
        # Calcul des coefficients de la Spline (représentation paramétrique)
        # u = t_unique (on utilise le temps comme paramètre)
        # s = facteur de lissage (smoothness)
        tck, u = splprep([x, y, z], u=t_unique, s=s_factor, k=3)
        
        # Génération des nouveaux points (beaucoup plus denses : 200 points)
        t_new = np.linspace(t_unique[0], t_unique[-1], 200)
        new_points = splev(t_new, tck) # new_points = [x_new, y_new, z_new]
        
        # On remet sous forme de matrice [X, Y, Z, T]
        return np.column_stack((new_points[0], new_points[1], new_points[2], t_new))
        
    except Exception as e:
        print(f"⚠️ Échec lissage : {e}")
        return traj_data

# --- 3. MAIN ---

data = load_data(INPUT_CSV)
if data is None: exit()
X, Y, Z, T = data[:,0], data[:,1], data[:,2], data[:,3]

# DBSCAN
features = np.column_stack((X, Y, Z, T * VITESSE_POIDS_TEMPS))
model = DBSCAN(eps=EPSILON_DISTANCE, min_samples=MIN_SAMPLES).fit(features)
labels = model.labels_

clusters_bruts = {}
for k in set(labels):
    if k == -1: continue
    clusters_bruts[k] = data[labels == k]

# Couture
trajectoires_finales = recoudre_trajectoires(clusters_bruts)
print(f"✅ {len(trajectoires_finales)} trajectoires identifiées.")


# --- 4. VISUALISATION LISSÉE ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.tab10(np.linspace(0, 1, len(trajectoires_finales)))

for i, traj in enumerate(trajectoires_finales):
    traj = traj[traj[:, 3].argsort()]
    color = colors[i]
    
    # 1. On affiche les POINTS BRUTS (les mesures réelles)
    # En petit et transparent pour montrer la "vérité terrain"
    ax.scatter(traj[:,0], traj[:,1], traj[:,2], color=color, s=15, alpha=0.3, label=f"Mesures O{i+1}")
    
    # 2. On calcule et affiche la COURBE LISSÉE
    traj_lisse = generer_courbe_lisse(traj, s_factor=FACTEUR_LISSAGE)
    
    # Gestion visuelle des trous (Pointillés si grand silence)
    # On parcourt la courbe lissée et on coupe si le dt est trop grand
    # (Car le lissage a interpolé le silence, on veut voir si c'est un "vol" ou un "saut")
    
    x_l, y_l, z_l, t_l = traj_lisse[:,0], traj_lisse[:,1], traj_lisse[:,2], traj_lisse[:,3]
    
    # On plotte par segments
    start_idx = 0
    for k in range(len(t_l)-1):
        dt = t_l[k+1] - t_l[k]
        # Si le pas de temps généré est anormalement grand (ce qui arrive si splprep traverse un grand vide)
        # Ou on compare avec les données brutes pour savoir où couper...
        
        # Méthode simple : On trace tout d'un coup en ligne épaisse
        pass 
    
    ax.plot(x_l, y_l, z_l, color=color, linewidth=3, alpha=0.9, label=f"Trajectoire O{i+1}")
    
    # Start / End
    ax.text(x_l[0], y_l[0], z_l[0], "Début", fontsize=9, fontweight='bold', color=color)
    ax.scatter(x_l[-1], y_l[-1], z_l[-1], c='red', marker='x', s=50) # Fin

ax.set_title(f"Trajectoires Lissées (B-Spline s={FACTEUR_LISSAGE})")
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()
