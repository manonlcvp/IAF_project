from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import torch
from PIL import Image
from torchvision import transforms
from model1 import load_model as load_genre_model
from model2 import MovieRecommendationModel

app = Flask(__name__)

# Configuration : device, modèles et Annoy Index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modèle pour la prédiction du genre
num_classes = 10
genre_model = load_genre_model(num_classes, device)
genre_model.load_state_dict(torch.load("prediction_model.pth", map_location=device))
genre_model.eval()

# Modèle pour les embeddings de recommandation
embedding_size = 256  # Correspond à la taille définie dans MovieRecommendationModel
recommendation_model = MovieRecommendationModel(embedding_size).to(device)
recommendation_model.load_state_dict(torch.load("recommendation_model.pth", map_location=device))
recommendation_model.eval()

# Annoy Index pour la recherche de similarités
annoy_index = AnnoyIndex(embedding_size, "angular")
annoy_index.load("movie_posters.ann")

# Charger les chemins des affiches
with open("poster_paths.txt", "r") as f:
    poster_paths = f.read().splitlines()

# Transformation des images (commune pour les deux routes)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict_genre():
    """
    Prédiction du genre d'un film à partir d'une affiche donnée.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Chargement et prétraitement de l'image
    image = request.files['image']
    img = Image.open(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        outputs = genre_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ["Action", "Animation", "Comedy", "Documentary", "Drama", 
                       "Fantasy", "Horror", "Romance", "Science Fiction", "Thriller"]
        genre = class_names[predicted.item()]

    return jsonify({'predicted_genre': genre})


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    """
    Recommande 5 films similaires à partir d'une affiche donnée.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Chargement et prétraitement de l'image
    image = request.files['image']
    img = Image.open(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Extraction de l'embedding avec le modèle de recommandation
    with torch.no_grad():
        embedding = recommendation_model.get_embedding(input_tensor).squeeze().cpu().numpy()

    # Recherche des 5 affiches les plus proches dans Annoy
    similar_indices = annoy_index.get_nns_by_vector(embedding, 5, include_distances=True)
    similar_posters = [{"path": poster_paths[i], "distance": d} for i, d in zip(*similar_indices)]

    return jsonify(similar_posters)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
