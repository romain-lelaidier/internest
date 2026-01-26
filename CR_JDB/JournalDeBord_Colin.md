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

- Algo qui transforme le signal binaire entre série temporelle de flottant (version beta de simulation)
- Discussion avec Nemo sur le pb panneau solaire / solar charge controller. Solution = commander d'autres panneaux solaires, et des solar charge controller
--> Mais d'abord, étudier aussi précisément que possible la consommation de l'installation et fixer l'autonomie désirée.
- recherche sur comment on charge la batterie

#### 23/01/2026

- serveur sur la raspberry, et conversion presque finie
- recherches sur l'aspect énergétique. Décision de ne pas s'intéresser aux panneaux solaires pour le moment.
- Réflexion : changer la rasp par un appareil plus light ? Par ex : rasp pico 2
- Pour recharger les batteries et faire le montage, aller au fablab. Lundi matin, tester direct en branchant la rasp et les esp sur les batteries.