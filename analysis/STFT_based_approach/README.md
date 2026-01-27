## Dossier avec les codes pour l'approche via STFT + Computer Vision

L'idée est d'exploiter les spectrogrammes des fichiers receuillis sur chaque micro. Ensuite on applique un modèle entrainé de type YOLO pour repérer les motifs spectraux.
Ensuite avec des algos de clustering et de lissage, on peut faire de la GCC et remonter aux trajectoires.

<img width="1440" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/bbad8ce5-926b-48df-955f-9990dd6771bf" />

<img width="1450" height="1484" alt="trajectories_3d" src="https://github.com/user-attachments/assets/f1d6a744-a1fb-4274-9fa1-6763d1b4bc8d" />


### Potentiels problèmes avec des fausses détections.

Le problème de cet algorithme de détection est qu'il est très robuste à partir du moment où le CNN est lui même très robuste. Mais on ne dispose pas d'un dataset avec beaucoup de données ni de grosse puissance de calcul pour pouvoir l'entrainer.

Donc dans certains cas les trajectoires peuvent dévier de la vérité. 

<img width="1450" height="1484" alt="trajectories_3d" src="https://github.com/user-attachments/assets/a4b9a465-4a34-47b6-847c-9d2a4f8d85c1" />

On ajoute donc une vérification qui se base sur une approche quelque peu similaire au RANSAC. C'est-à-dire que si on dispose de $N$ micros, on sait que l'on a besoin d'au moins $p \geq 4$ micros pour avoir une bonne reconstituion. Donc on peut faire des vérification en prenant des $p$-uplets de micros et reconstruire la trajectoire avec ces $p$-uplets et après moyenner pour eviter qu'un micro (s'il est defectueux) ne nous nuise.

Le deuxième problème c'est que cette méthode peut créer des hallucinations sur les points, donc on ajoute un filtre de "smoothing" assez puissant pour éliminer les points complètement invraisemblables. _(Ils sont marqués d'une croix rouge sur la figure ci-dessous)._

<img width="1422" height="1485" alt="trajectories_confidence" src="https://github.com/user-attachments/assets/bc35429b-6cab-4c8b-b79b-661bd8d4c675" />

<img width="1490" height="1483" alt="trajectories_confidence_2d" src="https://github.com/user-attachments/assets/a842e754-8822-4dc9-8e3e-d285ef7ce011" />



