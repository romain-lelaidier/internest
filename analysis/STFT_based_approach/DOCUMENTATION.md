# Localisation acoustique d'oiseaux par vision et TDOA

@author : Noé Daniel.

## Le problème

On veut localiser des oiseaux en 3D à partir de leurs chants captés par un réseau de microphones. L'idée est de combiner :
- **Computer Vision** : pour détecter les vocalisations sur des spectrogrammes
- **Acoustique** : pour trianguler la position à partir des différences de temps d'arrivée (TDOA)

---

## Vue d'ensemble du pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  8 microphones captent le chant d'un oiseau                         │
│  → Le son arrive à des instants légèrement différents sur chaque    │
│     micro (car la distance oiseau-micro varie)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  On transforme chaque signal audio en image (spectrogramme STFT)    │
│  → Le temps en X, la fréquence en Y, l'intensité en couleur         │
│  → Les chants d'oiseaux apparaissent comme des motifs visuels       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Un réseau de neurones (YOLO) détecte ces motifs                    │
│  → Bounding boxes avec coordonnées (temps, fréquence)               │
│  → Chaque détection = un "chirp" d'oiseau                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  On regroupe les détections pour identifier chaque oiseau           │
│  → Clustering par signature fréquentielle                           │
│  → Correspondance entre micros (même oiseau = même fréquence)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  On calcule les délais entre micros (TDOA)                          │
│  → Différence des temps de détection YOLO                           │
│  → Affinement par corrélation croisée (GCC-PHAT)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  On triangule la position 3D                                        │
│  → Résolution d'un système d'équations géométriques                 │
│  → Optimisation par moindres carrés                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  On lisse et valide les trajectoires                                │
│  → Filtrage des outliers (vitesse max, confiance)                   │
│  → Lissage temporel (Savitzky-Golay)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Les idées clés

### 1. Pourquoi transformer l'audio en image ?

Le son d'un oiseau est un signal 1D qui varie dans le temps. En le transformant en spectrogramme (via STFT - Short-Time Fourier Transform), on obtient une image 2D où :
- L'axe X représente le temps
- L'axe Y représente la fréquence
- L'intensité des pixels représente l'énergie

**Avantage** : Les chants d'oiseaux deviennent des **motifs visuels reconnaissables** (des "chirps" en forme de trait incliné). On peut alors utiliser des techniques de détection d'objets comme YOLO, qui excellent sur les images.

### 2. Pourquoi YOLO plutôt qu'une détection classique ?

Les méthodes classiques (seuillage, détection d'énergie) détectent tout ce qui dépasse un certain niveau sonore. Elles confondent facilement :
- Le bruit de fond
- Les chevauchements de plusieurs oiseaux
- Les échos

**YOLO apprend à reconnaître la forme caractéristique** d'un chirp d'oiseau. Même avec du bruit ou des chevauchements, il peut isoler les vrais chants.

### 3. Comment identifier quel oiseau chante ?

Chaque espèce (ou individu) a une **signature fréquentielle** caractéristique :
- Fréquence minimale du chant
- Fréquence maximale
- Fréquence centrale

On utilise **DBSCAN** (clustering par densité) pour regrouper les détections qui ont des signatures similaires. Ainsi :
- Toutes les détections avec f_min ≈ 3500 Hz et f_max ≈ 6500 Hz → Oiseau A
- Toutes les détections avec f_min ≈ 5000 Hz et f_max ≈ 9500 Hz → Oiseau B

Ensuite, on fait un **clustering global** pour associer les clusters de chaque micro entre eux (le cluster "Oiseau A" du micro 0 correspond au cluster "Oiseau A" du micro 3, etc.).

### 4. Comment calculer les délais (TDOA) ?

#### L'idée physique

Quand un oiseau chante à la position (x, y, z), le son met un certain temps à atteindre chaque micro :
```
temps_arrivée = distance / vitesse_du_son
```

Si l'oiseau est plus proche du micro 0 que du micro 1, le son arrive d'abord au micro 0. La différence de temps d'arrivée (TDOA) entre les deux micros dépend de la position de l'oiseau.

#### Comment mesurer ce délai ?

**Méthode 1 : Temps de détection YOLO**

YOLO détecte le chirp sur chaque spectrogramme et donne un temps central (`t_center`). Ce temps inclut déjà le délai de propagation ! La différence :
```
TDOA = t_center(micro_cible) - t_center(micro_référence)
```
donne directement le délai.

**Méthode 2 : Corrélation croisée (GCC-PHAT)**

On prend les signaux audio des deux micros et on cherche le décalage qui maximise leur ressemblance. C'est plus précis mais sensible au bruit.

**Notre stratégie** : On utilise principalement les temps YOLO (robustes), et on affine avec GCC-PHAT quand les deux méthodes sont cohérentes.

### 5. Comment trianguler en 3D ?

#### Le problème mathématique

On a 8 micros aux positions connues. Pour chaque micro i, on mesure le TDOA par rapport à un micro de référence. Ce TDOA correspond à une différence de distance :
```
distance(oiseau, micro_i) - distance(oiseau, micro_ref) = TDOA × vitesse_son
```

Chaque équation définit un **hyperboloïde** dans l'espace 3D (l'ensemble des points qui satisfont cette différence de distance). L'intersection de plusieurs hyperboloïdes donne la position de l'oiseau.

#### La résolution

On ne résout pas analytiquement (trop compliqué). On utilise une **optimisation par moindres carrés** :
1. On propose une position candidate (x, y, z)
2. On calcule les TDOA théoriques que cette position donnerait
3. On compare aux TDOA mesurés
4. On ajuste la position pour minimiser l'erreur

Pour éviter les minima locaux, on commence par une **recherche en grille** grossière avant l'optimisation fine.

### 6. Pourquoi valider avec des sous-ensembles de micros ?

Avec 8 micros et seulement 4 nécessaires pour trianguler, on a de la redondance. On peut donc :
1. Calculer la position avec tous les 8 micros
2. Recalculer avec chaque combinaison de 5 micros (C(8,5) = 56 combinaisons)
3. Comparer les résultats

**Si tous les sous-ensembles donnent des positions proches** → Haute confiance 

**Si les sous-ensembles divergent** → Un ou plusieurs micros ont des mesures erronées

C'est similaire à **RANSAC** : on détecte et élimine les outliers en vérifiant la cohérence.

### 7. Comment filtrer les trajectoires aberrantes ?

Un oiseau ne peut pas se téléporter. Entre deux mesures consécutives, on vérifie :

**Critère de vitesse** :
```
vitesse = distance / temps_écoulé
si vitesse > 25 m/s → point rejeté (impossible pour un oiseau)
```

**Critère de confiance** :
```
si confiance < 0.5 ou écart_type > 10m → point rejeté
```

Les points rejetés sont affichés en rouge sur les visualisations pour inspection manuelle.

---

## Configuration géométrique

Les 8 microphones sont placés aux coins d'un cube de 100m × 100m × 40m :

```
        6 ──────────── 7          z=40m (niveau haut)
       /│            /│
      / │           / │
     4 ──────────── 5  │
     │  │          │  │
     │  2 ─────────│── 3         z=0m (niveau sol)
     │ /           │ /
     │/            │/
     0 ──────────── 1

     (0,0)        (100,0)
           x →
```

Cette configuration cubique offre une bonne géométrie pour la triangulation 3D car elle couvre les trois dimensions de l'espace.

---

## Paramètres importants

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `FS` | 44100 Hz | Fréquence d'échantillonnage audio |
| `C` | 343 m/s | Vitesse du son dans l'air |
| `N_FFT` | 2048 | Taille de la fenêtre FFT (résolution fréquentielle) |
| `CONF_THRESHOLD` | 0.25 | Confiance minimum pour une détection YOLO |
| `MIN_MICS` | 4 | Minimum de micros pour trianguler |
| `V_MAX` | 25 m/s | Vitesse max d'un oiseau (~90 km/h) |
| `TIME_TOL` | 0.25 s | Tolérance pour regrouper les détections d'un même chirp |

---

## Limitations et améliorations possibles

### Limitations actuelles

1. **Chevauchements fréquentiels** : Si deux oiseaux chantent dans la même bande de fréquence, le clustering peut les confondre.

2. **Réverbération** : Les échos peuvent créer des fausses détections ou perturber les TDOA.

3. **Synchronisation** : Les micros doivent être parfaitement synchronisés (échantillonnage commun).

### Améliorations possibles

1. **Beamforming** : Utiliser la formation de voies pour améliorer le rapport signal/bruit avant détection.

2. **Tracking multi-hypothèses** : Maintenir plusieurs hypothèses de trajectoire et choisir la plus cohérente.

3. **Apprentissage des signatures** : Apprendre à reconnaître les individus, pas seulement les espèces.
