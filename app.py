from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import torch
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertModel
import pickle

# Charger les modèles et données
app = Flask(__name__)

# Configuration : device, modèles et Annoy Index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modèle pour la prédiction du genre
from Part_1.model1 import load_model as load_genre_model
num_classes = 10
genre_model = load_genre_model(num_classes, device)
genre_model.load_state_dict(torch.load("Part_1/prediction_model.pth", map_location=device))
genre_model.eval()

# Modèle d'autoencodeur pour la détection d'anomalies
from Part_1.model_anomaly import Autoencoder
anomaly_model = Autoencoder().to(device)
anomaly_model.load_state_dict(torch.load("Part_1/anomaly_detector.pth", map_location=device))
anomaly_model.eval()

# Seuil pour détecter les anomalies (valeur à ajuster selon votre modèle)
anomaly_threshold = 1.5

# Modèle pour les embeddings de recommandation
def load_recommendation_model():
    from Part_2.model2 import MovieRecommendationModel
    embedding_size = 256
    model = MovieRecommendationModel(embedding_size).to(device)
    model.load_state_dict(torch.load("Part_2/recommendation_model.pth", map_location=device))
    model.eval()
    return model

recommendation_model = load_recommendation_model()

# Annoy Index pour les affiches
poster_annoy_index = AnnoyIndex(256, "angular")
poster_annoy_index.load("Part_2/movie_posters.ann")

# Charger les chemins des affiches
with open("Part_2/poster_paths.txt", "r") as f:
    poster_paths = f.read().splitlines()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger les embeddings et Annoy Index pour les synopsis
with open("Part_3/bag_of_words_embeddings.pkl", "rb") as f:
    bow_embeddings = pickle.load(f)

with open("Part_3/distilbert_embeddings.pkl", "rb") as f:
    distilbert_embeddings = pickle.load(f)

bow_annoy_index = AnnoyIndex(len(bow_embeddings[0]), "angular")
bow_annoy_index.load("Part_3/bow_annoy.ann")

distilbert_annoy_index = AnnoyIndex(len(distilbert_embeddings[0]), "angular")
distilbert_annoy_index.load("Part_3/distilbert_annoy.ann")

# Charger les métadonnées
with open("Part_3/movies_metadata.pkl", "rb") as f:
    movies_metadata = pickle.load(f)

# Initialisation de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict_genre_and_anomaly():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Chargement de l'image
    image = request.files['image']
    img = Image.open(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Prédiction du genre
    with torch.no_grad():
        genre_outputs = genre_model(input_tensor)
        _, predicted = torch.max(genre_outputs, 1)
        class_names = ["Action", "Animation", "Comedy", "Documentary", "Drama", 
                       "Fantasy", "Horror", "Romance", "Science Fiction", "Thriller"]
        genre = class_names[predicted.item()]

    # Détection d'anomalie
    with torch.no_grad():
        reconstructed = anomaly_model(input_tensor)
        reconstruction_error = torch.nn.functional.mse_loss(reconstructed, input_tensor).item()
        is_anomaly = reconstruction_error > anomaly_threshold

    # Résultat
    return jsonify({
        'predicted_genre': genre,
        'anomaly_detected': is_anomaly,
        'reconstruction_error': reconstruction_error
    })

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img = Image.open(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = recommendation_model.get_embedding(input_tensor).squeeze().cpu().numpy()

    similar_indices = poster_annoy_index.get_nns_by_vector(embedding, 5, include_distances=True)
    similar_posters = [{"path": poster_paths[i], "distance": d} for i, d in zip(*similar_indices)]

    return jsonify(similar_posters)

@app.route('/recommend_plot', methods=['POST'])
def recommend_based_on_plot():
    data = request.json
    if not data or "plot" not in data or "method" not in data:
        return jsonify({'error': 'Invalid input'}), 400

    plot = data["plot"].strip()
    if not plot:
        return jsonify({'error': 'Empty plot description'}), 400

    method = data["method"]  # "bow" ou "distilbert"

    try:
        if method == "bow":
            input_vector = bow_embeddings.mean(axis=0)  # Approche simplifiée
            index = bow_annoy_index
        elif method == "distilbert":
            inputs = tokenizer(plot, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                input_vector = distilbert_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            index = distilbert_annoy_index
        else:
            return jsonify({'error': 'Invalid method'}), 400

        similar_indices = index.get_nns_by_vector(input_vector, 5, include_distances=True)
        recommendations = [{"title": movies_metadata.iloc[i]["title"], "distance": d} for i, d in zip(*similar_indices)]

        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5002)
