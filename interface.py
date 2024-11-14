import gradio as gr
import requests

# URL de l'API Flask
API_URL = "http://127.0.0.1:5002/predict"

# Fonction pour envoyer l'image à l'API et obtenir la prédiction
def predict_genre(image):
    # Convertir l'image pour l'envoyer en tant que fichier dans une requête POST
    with open(image, "rb") as file:  # Utilisation du gestionnaire de contexte
        files = {"image": file}
        response = requests.post(API_URL, files=files)
    
    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Récupérer le genre prédit
        return response.json().get("predicted_genre", "Genre non reconnu")
    else:
        # Gérer les erreurs d'API
        return f"Erreur dans la prédiction (Code: {response.status_code})"

# Création de l'interface Gradio
interface = gr.Interface(
    fn=predict_genre,  # Fonction appelée pour la prédiction
    inputs=gr.Image(type="filepath"),  # Entrée de type image
    outputs="text",  # Sortie de type texte (le genre prédit)
    title="Prédiction de Genre de Film",
    description="Charge une affiche de film pour prédire son genre."
)

# Lancer l'interface
interface.launch()
