from annoy import AnnoyIndex
import torch
from model2 import MovieRecommendationModel
from torchvision import transforms
from PIL import Image
import os

# Charger le modèle de recommandation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 256  # Taille des embeddings
recommendation_model = MovieRecommendationModel(embedding_size).to(device)
recommendation_model.load_state_dict(torch.load("recommendation_model.pth", map_location=device))
recommendation_model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Créer l'index Annoy
annoy_index = AnnoyIndex(embedding_size, "angular")

# Chemin vers les affiches de films
poster_directory = "data/part 2"
poster_paths = []
index = 0

# Ajouter les embeddings de chaque affiche au fichier Annoy
for poster_path in os.listdir(poster_directory):
    image_path = os.path.join(poster_directory, poster_path)
    poster_paths.append(image_path)

    # Charger et prétraiter l'image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Extraire l'embedding avec le modèle de recommandation
    with torch.no_grad():
        embedding = recommendation_model.get_embedding(input_tensor).squeeze().cpu().numpy()

    # Ajouter l'embedding à l'index Annoy
    annoy_index.add_item(index, embedding)
    index += 1

# Construire l'index (le nombre d'arbres influence la vitesse et la précision)
annoy_index.build(10)  # Le paramètre 10 est le nombre d'arbres

# Sauvegarder l'index dans un fichier
annoy_index.save("movie_posters.ann")

# Sauvegarder les chemins des affiches
with open("poster_paths.txt", "w") as f:
    for path in poster_paths:
        f.write(f"{path}\n")
