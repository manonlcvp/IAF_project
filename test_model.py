import os
import torch
from torchvision import transforms
import argparse
from PIL import Image
from model import load_model

# Configurations
def parse_args():
    parser = argparse.ArgumentParser(description='Test a model for movie genre prediction.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image of the movie poster')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    return parser.parse_args()

# Chargement des arguments (récupération depuis la ligne de commande)
args = parse_args()
image_path = args.image
model_path = args.model_path

# Transformation de l'image pour la prédiction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

# Définition du device (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement du modèle et configuration du mode évaluation
num_classes = len(os.listdir('data'))
model = load_model(num_classes, device) 
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

image = image.to(device)

# Prédiction du genre du film associé à l'image
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    class_names = os.listdir('data')  # Liste des genres
    print(f"Predicted genre: {class_names[predicted.item()]}")
