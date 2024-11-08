import os
import torch
from torchvision import transforms
from PIL import Image
from model import load_model

# Configurations
model_path = "model.pth"
image_path = "images_test/110.jpg"  # Remplace par le chemin d'une image d'affiche de test

# Préparation de l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

# Charger le modèle
num_classes = len(os.listdir('data'))
model = load_model(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # Mettre le modèle en mode évaluation

# Prédire
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    class_names = os.listdir('data')  # Liste des genres
    print(f"Predicted genre: {class_names[predicted.item()]}")