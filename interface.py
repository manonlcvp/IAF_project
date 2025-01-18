import gradio as gr
import requests

# URLs de l'API
API_URL_PREDICT = "http://model_api:5002/predict"
API_URL_RECOMMEND = "http://model_api:5002/recommend"
API_URL_RECOMMEND_PLOT = "http://model_api:5002/recommend_plot"

def predict_genre_and_anomaly(image):
    """Prédit le genre du film et détecte les anomalies à partir d'une affiche."""
    try:
        with open(image, "rb") as file:
            response = requests.post(API_URL_PREDICT, files={"image": file})
        if response.status_code == 200:
            data = response.json()
            genre = data.get("predicted_genre", "Genre inconnu")
            anomaly = data.get("anomaly_detected", False)
            reconstruction_error = data.get("reconstruction_error", "N/A")

            # Formater le résultat
            anomaly_text = "Anomalie détectée" if anomaly else "Aucune anomalie détectée"
            result = (
                f"**Genre prédit :** {genre}\n"
                f"**Anomalie :** {anomaly_text}\n"
                f"**Erreur de reconstruction :** {reconstruction_error:.4f}"
            )
            return result
        return f"Erreur API (Code {response.status_code}) : {response.text}"
    except Exception as e:
        return f"Erreur : {str(e)}"

def recommend_movies(image):
    """Recommande des films similaires à partir d'une affiche."""
    try:
        with open(image, "rb") as file:
            response = requests.post(API_URL_RECOMMEND, files={"image": file})
        if response.status_code == 200:
            posters = response.json()
            return [poster["path"] for poster in posters]
        return f"Erreur API (Code {response.status_code}) : {response.text}"
    except Exception as e:
        return f"Erreur : {str(e)}"

def recommend_movies_by_plot(plot, method):
    """Recommande des films similaires à partir d'une description d'intrigue."""
    payload = {"plot": plot, "method": method}
    try:
        response = requests.post(API_URL_RECOMMEND_PLOT, json=payload)
        if response.status_code == 200:
            movies = response.json()
            return [[movie['title'], movie['distance']] for movie in movies]
        return f"Erreur API (Code {response.status_code}) : {response.text}"
    except Exception as e:
        return f"Erreur : {str(e)}"

# Interface Gradio pour la prédiction de genre et la détection d'anomalies
genre_anomaly_interface = gr.Interface(
    fn=predict_genre_and_anomaly,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Prédiction de Genre et Détection d'Anomalies",
    description="Charge une affiche de film pour prédire son genre et détecter des anomalies éventuelles."
)

# Interface Gradio pour la recommandation de films (affiches)
recommend_interface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Gallery(label="Affiches similaires"),
    title="Recommandation de Films Similaires",
    description="Charge une affiche de film pour trouver des affiches similaires."
)

# Interface Gradio pour la recommandation basée sur le contenu
plot_recommend_interface = gr.Interface(
    fn=recommend_movies_by_plot,
    inputs=[
        gr.Textbox(lines=5, placeholder="Entrez la description de l'intrigue du film."),
        gr.Dropdown(choices=["bow", "distilbert"],
                    label="Technique d'embedding",
                    value="bow")
    ],
    outputs=gr.List(label="Films similaires"),
    title="Recommandation Basée sur le Contenu",
    description="Entrez une description d'intrigue pour trouver des films similaires."
)

# Interface tabulée
gr.TabbedInterface(
    [genre_anomaly_interface, recommend_interface, plot_recommend_interface],
    ["Prédiction de Genre et Anomalies", "Recommandation (Affiches)", "Recommandation (Intrigue)"]
).launch(server_name="0.0.0.0")