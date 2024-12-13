import gradio as gr
import requests

# URLs de l'API
API_URL_PREDICT = "http://127.0.0.1:5002/predict"
API_URL_RECOMMEND = "http://127.0.0.1:5002/recommend"
API_URL_RECOMMEND_PLOT = "http://127.0.0.1:5002/recommend_plot"

def predict_genre(image):
    """Prédit le genre du film à partir d'une affiche."""
    try:
        with open(image, "rb") as file:
            response = requests.post(API_URL_PREDICT, files={"image": file})
        if response.status_code == 200:
            return response.json().get("predicted_genre", "Erreur dans la prédiction")
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

# Interface Gradio pour la prédiction de genre
genre_interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Prédiction de Genre de Film",
    description="Charge une affiche de film pour prédire son genre."
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
    [genre_interface, recommend_interface, plot_recommend_interface],
    ["Prédiction de Genre", "Recommandation (Affiches)", "Recommandation (Intrigue)"]
).launch()
