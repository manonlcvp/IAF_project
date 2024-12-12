import gradio as gr
import requests

# URL de l'API
API_URL_PREDICT = "http://127.0.0.1:5002/predict"
API_URL_RECOMMEND = "http://127.0.0.1:5002/recommend"

def predict_genre(image):
    with open(image, "rb") as file:
        response = requests.post(API_URL_PREDICT, files={"image": file})
    if response.status_code == 200:
        return response.json().get("predicted_genre", "Erreur dans la prédiction")
    return f"Erreur API (Code {response.status_code})"

def recommend_movies(image):
    with open(image, "rb") as file:
        response = requests.post(API_URL_RECOMMEND, files={"image": file})
    if response.status_code == 200:
        posters = response.json()
        return [poster["path"] for poster in posters]
    return f"Erreur API (Code {response.status_code})"

# Interface Gradio pour prédiction de genre
genre_interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Prédiction de Genre de Film",
    description="Charge une affiche de film pour prédire son genre."
)

# Interface Gradio pour recommandation de films
recommend_interface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Gallery(label="Affiches similaires"),
    title="Recommandation de Films Similaires",
    description="Charge une affiche de film pour trouver des affiches similaires."
)

# Interface tabulée
gr.TabbedInterface(
    [genre_interface, recommend_interface],
    ["Prédiction de Genre", "Recommandation de Films"]
).launch()
