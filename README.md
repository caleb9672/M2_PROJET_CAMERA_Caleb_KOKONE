# Projet APEKE : Détection, Alertes et Suivi Vidéo

Ce projet est une solution complète d'analyse vidéo divisée en trois étapes clés : la détection d'objets, la surveillance de zones d'alerte et le suivi (tracking) de personnes spécifiques.

# Pour commencer veuillez charger toutes les vidéos dans le dossier vide "Videos"


## Techniques Implémentées

- **Détection d'Objets (YOLOv8)** : Utilisation du modèle YOLOv8 pour identifier des objets dans les vidéos.
- **Optimisation OpenVINO** : Exportation et utilisation de modèles OpenVINO pour des performances accrues sur CPU.
- **Traitement Efficace** : Utilisation du mode `stream=True` pour minimiser l'utilisation de la RAM et `multiprocessing` pour traiter plusieurs vidéos en parallèle.
- **Zones d'Alerte Interactives** : Interface permettant de définir graphiquement des zones de surveillance (polygones).
- **Suivi Multi-Objets (ByteTrack)** : Algorithme de tracking robuste pour suivre les individus même en cas d'occlusion.
- **Re-Identification (Re-ID)** : Utilisation de ResNet50 pour extraire des caractéristiques visuelles et retrouver une personne spécifique dans différentes vidéos via la similarité cosinus.

## Installation

1. Clonez le dépôt ou téléchargez les fichiers.
2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

## Exécution du Projet

Le projet se déroule en 3 étapes distinctes. Assurez-vous que vos vidéos sont placées dans le dossier `videos/`.

### Étape 1 : Détection Globale
Cette étape analyse toutes les vidéos du dossier source pour détecter les objets et enregistrer leurs coordonnées dans des fichiers CSV.

- **Fichier à exécuter** : `step1_detection/src/process_videos.py`
- **Configuration** : Modifiez `step1_detection/config.yaml` pour ajuster les paramètres (modèle, classes cibles, nombre de workers).
- **Commande** :
  ```bash
  python step1_detection/src/process_videos.py
  ```

### Étape 2 : Surveillance de Zone (Alertes)
Cette étape permet de définir une zone sensible et de capturer automatiquement les visages/silhouettes des personnes qui y pénètrent.

1. **Définir la zone** : Exécutez `zone_picker.py` pour cliquer sur les points du polygone. Appuyez sur 'q' pour sauvegarder.
   ```bash
   python step2_alerts/src/zone_picker.py
   ```
2. **Lancer la surveillance** :
   ```bash
   python step2_alerts/src/alert_monitor.py
   ```
- **Sorties** : Les captures d'écran et un journal CSV sont enregistrés dans `results_alerts/`.

### Étape 3 : Suivi de Personne Spécifique (Tracking Re-ID)
Cette étape recherche une personne précise dans une vidéo à partir d'images de référence.

- **Fichier à exécuter** : `step3_tracking/src/person_tracker.py`
- **Configuration** : Ajoutez les chemins de vos images de référence dans `step3_tracking/config.yaml`.
- **Commande** :
  ```bash
  python step3_tracking/src/person_tracker.py
  ```
- **Sorties** : Une vidéo annotée et un fichier CSV de tracking dans `step3_tracking/results/`.

## Structure du Projet

- `step1_detection/` : Scripts et config pour la détection initiale.
- `step2_alerts/` : Outils de définition de zone et monitoring.
- `step3_tracking/` : Système de Re-ID pour le suivi de cibles.
- `videos/` : Dossier contenant les vidéos sources.
- `requirements.txt` : Liste des dépendances Python.
