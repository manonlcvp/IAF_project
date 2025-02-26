# Projet AIF

## Description

Ce projet est une application web basée sur Flask et Gradio qui permet de :
1. Prédire le genre d'un film à partir de son affiche tout en détectant d'éventuelles anomalies dans les données.
2. Recommander des films similaires à partir d'une affiche.
3. Recommander des films similaires à partir d'une description de l'intrigue.

---

## Technologies Utilisées

### Modèles de Machine Learning
- **PyTorch** : Pour l'entraînement et l'inférence des modèles.
- **DistilBERT** : Pour générer des embeddings basés sur le texte.
- **Annoy** : Pour une recherche rapide de similarités entre embeddings.

### Frameworks et Outils
- **Flask** : Backend API pour servir les prédictions et recommandations.
- **Gradio** : Interface utilisateur interactive et simple à utiliser.
- **Torchvision** : Pour prétraiter les images et utiliser des datasets.
- **Pillow** : Manipulation d'images.

### Autres
- **Pickle** : Sauvegarde des embeddings et des métadonnées.
- **Annoy Index** : Recherche rapide de similarités entre affiches et descriptions.

---

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/manonlcvp/IAF_project.git
   cd IAF_project
   ```

2. **Créer les poids** :
   ```bash
   cd Part_1
   python train1.py
   python train_anomaly_model.py
   ```
   ```bash
   cd Part_2
   python train2.py
   python generate_annoy_part2.py
   ```
   ```bash
   cd Part_3
   python generate_bow_distilbert.py
   ```

3. **Lancer Docker** :
   ```bash
   docker-compose up --build
   ```