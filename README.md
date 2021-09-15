# Challenge Mathématiques et Entreprise

Ce dossier comporte l'implémentation d'une méthode de reconstitution de trajectoires, développée dans le cadre du Challenge Mathématiques Entreprises (https://challenge-maths.sciencesconf.org/) sur le sujet proposé par l'entreprise Eurecam (https://eurecam.net/fr/).
Le travail a été réalisé par Olympio Hacquard, Etienne Lasalle et Vadim Lebovici.

## Résumé

La méthode repose sur du transport optimal avec bord. Il permet d'appairer les personnes d'une image à l'autre pour reconstruire les trajectoires de proche en proche. Elle inclut une phase de pré-traitement pour retirer les données abérrantes et recalculer des nouvelles coordonnées de détections, de sorte qu'il n'y ait plus qu'une détection par personne. Une phase de post-traitement est également implémentée, dans le but de recoller certaines trajectoires et de nettoyer les trajectoires aberrantes.

## Organisation du dossier

Le fichier principal `main.py` contient le code permettant de calculer les trajectoires sur les dix jeux de données fournis par Eurecam. Ces données sont stockées dans le dossier `./data_detection/`. Le fichier principal fait appel à des fonctions implémentées dans les différents fichiers : `tools_for_preprocessing.py`, `tools_for_reconstruction.py`, `tools_for_postprocessing.py` et `tools_for_visualization.py`. Le fichier `best_parameters.csv` contient les paramètres adaptés pour chacun des jeux de données. Le dossier `./csv/` reçoit les données de détection une fois le pré-traitement effectué. Le fichier principal enregistre par défaut les images contenant les trajectoires reconstituées dans le dossier `./save/`. Vous pouvez retrouver les vidéos de nos reconstitutions dans le dossier `./videos/`.

## Pour utiliser ce code

Pour installer les librairies nécessaires et exécuter le fichier `main.py`, on propose d'utiliser un nouvel environnement conda. Une fois un terminal ouvert et localisé dans le dossier principal, vous pouvez taper les lignes suivantes :

```
conda create -n challenge_eurecam python=3.8
conda activate challenge_eurecam
python -m pip install pot scikit-image matplotlib pandas imageio
python main.py
```
