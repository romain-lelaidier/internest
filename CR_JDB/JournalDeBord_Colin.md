#### 14/01/2026

- Set raspberry as wifi access point : menu wifi en haut à droite (sur l'interface graphique)
- mqtt to exchange messages using python from raspberry to computer. Mqtt : for time synchronisation
- For sound data transfer, websocket seems better. Using python package websocket and asyncio. Works. How to transfer from esp then ?


#### 16/01/2026

- Working on websocket audio streaming
- (Romain) use raspap to create wifi access point with raspberry pi. https://raspap.com/quick-start/

#### 19/01/2026

- nouveau protocole de communication : udp, probablement plus adapté pour du streaming audio continu
- recherches rapides sur l'alimentation des cartes par batterie et panneau solaire. Not straightforward. Demander à Nemo demain.
- Il faut discuter du format des données (IN) et (OUT)

#### 21/01/2025

- Algo qui transforme le signal binaire en série temporelle de flottant (version beta de simulation)
- Discussion avec Nemo sur le pb panneau solaire / solar charge controller. Solution = commander d'autres panneaux solaires, et des solar charge controller
--> Mais d'abord, étudier aussi précisément que possible la consommation de l'installation et fixer l'autonomie désirée.
- recherche sur comment on charge la batterie

#### 23/01/2026

- serveur sur la raspberry, et conversion presque finie
- recherches sur l'aspect énergétique. Décision de ne pas s'intéresser aux panneaux solaires pour le moment.
- Réflexion : changer la rasp par un appareil plus light ? Par ex : rasp pico 2
- Pour recharger les batteries et faire le montage, aller au fablab. Lundi matin, tester direct en branchant la rasp et les esp sur les batteries.

#### 30/01/2026

- fait des trucs inutiles pcq Romain l'avait déjà fait ou l'a fait derrière moi. (serveur rasp qui reçoit de manière asynchrone; nouvelle manière de stocker, élaboration de la levture; pllay audio)
- regardé  pour lorawan : on peut pas streamer l'audio en continu mais (raisonnement en terme de débit):
    - on peut streamer les extraits d'intérêt (en compressant si besoin, si on n'a pas l'intention de les traiter, mais juste les écouter)  [NOTE : même extraits c'est chaud]
    - si le traitement est fait en local, on peut streamer la position et l'identité en temps réel. 
- justement, expliciter les filtres (quand il y a un signal et quand il n'y en a pas) ? **Réunion**
- voir photo tableau: autres trucs à discuter pour la prochaine fois (lundi)

#### 02/02/2025

- travaillé sur LoRa WaN toute la journée :
La communication fonctionne. Maintenant je réfléchis à quelle sera la forme du signal d'entrée et comment le transférer, comment l'encoder. Quelles données exactement ?

#### 04/02/2025

- travaillé sur LoRa Wan toute la journée.
ça avance bien ! j'arrive à communiquer les données
RESTE A FAIRE :
- Réception systématique et décodage des données
- Voir comment ça se passe vrmt, avec les autres : si on veut du temps réel : watchdog. Voir ce que Noé a déjà fait (en gros il fait ce que je fais mais avec ssh)
PROBLEMES :
- overflow avec les to bytes _ok_
- delay : récupérer tous les paquets (quitte à avoir des doublons) _ok mais mtn gérer les doublons_
- quelle fréquence d'envoi est possible ?? Comment s'organiser pour atteindre un débit maximum tout en se prémunissant des pertes (et si les pertes sont fatales, comment les détecter ?) ça il doit y avoir bcp de doc dessus.

Files : test.ino for Lora 1
lora 2 : sketch receiver


#### 06/02/2026

- struggling with wsl and on the Serial communication. Tried serial python module, but it doesn't work. Having trouble with coolterms too
- serial works now. The problem was wsl with port detection. Consider using dual boot instead of wsl ?

NOW :
0) passe proprement sur windows vs code pour ce programme (drop wsl for now) _ok_
1) supprime les doublons _ok_
2) decode message pas en temps réel. Vérifie que la transmission se passe bien en plottant les données / printant
Ensuite : on verra. Time to eat. _on it_

SO now : solve minor problems :
pbs : 
- position is badly decoded
- so are start end time
- so is birdnet_id

#### 09/02/2026
Difficult morning, only starting at 12

To do today :
1) **slides**
2) schéma api pour lora
3) leur montrer, leur demander ce qu'ils en pensent. Discuter aussi de comment ça va de leur côté
4) Faire l'api tout propre

Ensuite on verra en fonction des autres.

Bilan séance :
- Slides à finir chez moi (prendre en photo batterie)
- encore pbs avec lora, mercredi j'arrive en avance pcq ils auront peut-être besoin de moi pour autre chose après. Ou bien mardi soir juste après les stats comme je devrais sortir un peu en avance.

Pour Lora :
- bin to csv
- lire pour voir à quoi ça ressemble et mieux comprendre l'erreur.


#### 11/02/2026

- worked on the slides.
- LoRaWAN : it works well. Some packets are lost, but not a lot.

This afternoon:
- watchdog -> tout dans un csv donc pas forcément utile.
- API : fonctions utiles. Tu la lances et ça fait quoi exactement ? Pour la démo
    - sur rasp, watchdog pour scanner les données et les envoyer (adapter le send_pos_csv, ou plutôt watchdog pour l'appliquer au fur et à mesure (voir mistral))
    - sur PC, recevoir en continu les données, les enregistrer dans des fichiers bin, puis en parallèle watchdog qui décrypte et... fait quoi ? Fenêtre qui montre la progression en réel (position, cost, ...). HTML --> demander ce qu'il fait déjà leur truc, et avec quelles données ? 
- Could add a byte : packet id (number) to track which ones are lost. Take off a byte from time is ok
- Download the dictionnary code -> specie name and use it
- 

#### 12/02/2026

for some reason, il faut que je lance l'ide pour activer 
Non c'est bon, c'est pcq j'avais mis (while !Serial), donc tant que j'allume pas le moniteur in any way, il attend et fait rien.

------------------------

!!! Potentiellement problème avec l'alimentation batterie !!!

## Les slides :
Batteries et LoRa

__Batteries__ : il faut acheter des composants spécifiques et adaptés aux cartes. Il faut des charges controller si on veut que ça se passe bien. En théorie, même pour les esp, un transformateur DC-DC ce serait bien, pour gérer la chute de tension. Mais batteries au Li chutent peu en tension. Entre 80% et 20% de charge, très stable à 3.7V

J'ai eu bcp de mal avec la doc. On trouve difficilement les bonnes infos.

panneau solaire, c'est galère, et ça augmente pas tant l'autonomie.

Petite batterie simple. Piste : grosse pour rasp comme on a besoin de pas mal de puissance de calcul (à dimensionner encore)
On arrivera pas à plus de qqs heures à part grosse installation avec gros panneaux solaires, mais risque de troubler les oiseaux (réflexions par ex.). Bref, piste. Piste : passer à Pi Zero ? réduire les calculs, pas de transmission de données, mais plus d'autonomie. Ou transmission de méta données de base. Ou tout petit traitement.

------
BATTERIE, AUTONOMIE

Intérêt autonomie. Possibilité de déployer dans les espaces cibles.
On a travaillé sur les batteries, mais pas trop approfondi :
- manque de temps
- d'autres choses plus intéressantes et plus en lien avec le thème du trimestre (LoRaWAN)
- Possibilité de faire une démo avec la raspberry sur secteur
- manque de compétence en électronique + difficulté de trouver des infos fiables (forums) :
    - problème de tension : branché esp sur 3.7V alors que tension nominale = 3.3V. Des qui disent que ça marche, d'autres non. Différence alim pin et usb c. Ou bien histoire de puissance. Technique, je vais pas demander à Nemo de faire toute une partie du projet.

Présentation de la solution actuelle
ESP sur batterie (caractéristiques batteries, autonomie)
Mais (puissance) plus compliqué pour la rasp

Perspectives :
- Solution de Nemo, 3-4 h. Bien mais pas ouf.
- Panneaux solaires (regarder puissance nécessaire pour donner les dimensions)
|-> pb = pour la rasp, très grand. Fragile (faune, chute de branche) + perturbation des oiseaux
- Prendre un modèle qui consomme moins pour la rasp pi (zero, mais puissance ? -> moins de traitement sur place, juste stocker les données bruts ?)

Conclusion sur le problème de l'autonomie énergétique :
Pas mal de pistes, mais pas trop développé

------

__LoRa__ : Long Range Wide Area Network, protocole de communication
pourquoi c'est intéressant (2 - 15 km (ville / campagne))
expliquer toute la chaîne de communication, les problèmes majeurs et intéressants
Pistes : possibilités d'envoyer les audios d'intérêt / communiquer avec la rasp pour la contrôler à distance

------

Expliquer schéma

Expliquer chacune des étapes :
- I2C : type de communication filaire entre plusieurs appareils (dvper vite  fait)
Directement sur les pin rasp et arduino
- LORA : vu en cours, expliquer avantages, inconvénients. Nécessité de traitement en réel.
- Série : juste python se branche sur le port et écoute.

Transmission en temps direct, 10 paquets par seconde ok suffisant
Transmission par paquets
Quel format ces paquets ? Binaire, convertissage des données en entiers (ou en char) tq l'utilisation des octets est optimisé. concat, en donnant un nombre fixe d'octet à chaque données. Par ex, time sur 8 o. Pos sur 2
Correspond au niveau de précision qu'on veut

Problèmes :
perte des paquets (?), plusieurs réceptions du même paquet (pas grave, unicité)
I2c : start byte

Perspectives :
Commander : lui dire de s'éteindre en cas de mauvaises conditions (météo); 
Surveillance : usage batterie, tests basiques pour vérifier que tout va bien (fréquence de réception des données pour chaque esp, est ce que le traitement se fait bien)

