import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from model import load_model

# Initialisation de l'application Flask
app = Flask(__name__)

# Définition du device (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement du modèle et configuration du mode évaluation
num_classes = 10
model = load_model(num_classes, device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Transformation de l'image pour la prédiction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Catégories de films
class_names = ["Action", "Animation", "Comedy", "Documentary", "Drama", "Fantasy", "Horror", "Romance", "Science Fiction", "Thriller"]

@app.route('/predict', methods=['POST'])
def predict_genre():
    # Vérifier si une image a été envoyée
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Chargement de l'image
    image = request.files['image']
    img = Image.open(image).convert('RGB')
    
    # Prétraitement de l'image
    img = transform(img).unsqueeze(0)  # Ajouter une dimension pour le batch
    img = img.to(device)  # Déplacer l'image vers le GPU ou CPU
    
    # Prédiction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        genre = class_names[predicted.item()]
    
    # Retourner le genre prédit en JSON
    return jsonify({'predicted_genre': genre})

if __name__ == '__main__':
    app.run(debug=True, port=5002)